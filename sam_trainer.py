#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer-based fine-tuning for SAM (Segment Anything) with an explicit validation split.
Default dataset: "nielsr/breast-cancer" (HF datasets). Replace with your dataset as needed.
"""

import argparse
from statistics import mean
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import load_dataset, DatasetDict
from transformers import (
    SamProcessor,
    SamModel,
    SamConfig,
    Trainer,
    TrainingArguments,
)

try:
    import monai
except ImportError as e:
    raise SystemExit("This script requires MONAI. Install with `pip install monai`.")


# -------------------------
# Data utilities
# -------------------------
def get_bounding_box(ground_truth_map: np.ndarray):
    """Compute a slightly jittered bounding box around the positive mask region."""
    y_indices, x_indices = np.where(ground_truth_map > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Fallback to full image if mask is empty
        H, W = ground_truth_map.shape
        return [0, 0, W - 1, H - 1]
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W - 1, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H - 1, y_max + np.random.randint(0, 20))
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


class SAMDataset(Dataset):
    """
    Thin wrapper that runs the SAM processor and attaches a ground-truth mask.
    Expects HF examples to have keys: {"image", "label"} with label as a mask-like array.
    """

    def __init__(self, hf_split, processor: SamProcessor):
        self.ds = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        gt_mask = np.array(item["label"], dtype=np.uint8)
        bbox = get_bounding_box(gt_mask)

        enc = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}  # remove added batch dim
        enc["ground_truth_mask"] = gt_mask  # numpy; collator will convert

        return enc


def sam_data_collator(features):
    """
    Collate processor outputs for SAM + attach (B,1,H,W) tensor masks.
    """
    batch = {}
    tensor_keys = ["pixel_values", "input_boxes", "original_sizes", "reshaped_input_sizes"]
    for k in tensor_keys:
        elems = [f[k] for f in features]
        if isinstance(elems[0], torch.Tensor):
            batch[k] = torch.stack(elems)
        else:
            batch[k] = torch.tensor(elems)
    gts = [torch.tensor(f["ground_truth_mask"], dtype=torch.float32) for f in features]
    gts = torch.stack(gts).unsqueeze(1)
    batch["ground_truth_mask"] = gts.unsqueeze(1)
    return batch


# -------------------------
# Trainer subclass + metricsx
# -------------------------
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


def _dice_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    pred_bin = (pred > 0.5).float()
    target = target.float()
    inter = (pred_bin * target).sum(dim=(1, 2, 3))
    union = (pred_bin + target).sum(dim=(1, 2, 3)) - inter
    dice = (2 * inter + eps) / (pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


class SamTrainer(Trainer):
    """
    Custom Trainer that applies DiceCE loss to SAM's pred_masks and computes simple metrics.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("ground_truth_mask")
        outputs = model(**inputs, multimask_output=False)
        logits = outputs.pred_masks  # (B, 1, H, W) – SAM's mask decoder output
        loss = seg_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        labels = inputs.get("ground_truth_mask")
        with torch.no_grad():
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_boxes=inputs["input_boxes"],
                original_sizes=inputs["original_sizes"],
                reshaped_input_sizes=inputs["reshaped_input_sizes"],
                multimask_output=False,
            )
            logits = torch.sigmoid(outputs.pred_masks)  # probs for metrics
            loss = seg_loss(outputs.pred_masks, labels) if labels is not None else None
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits.detach().cpu(), labels.detach().cpu())


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # Ensure shapes (B,1,H,W)
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:
        preds = preds[:, None, :, :]
    if labels.ndim == 3:
        labels = labels[:, None, :, :]
    preds_t = torch.tensor(preds, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    d, i = _dice_iou(preds_t, labels_t)
    return {"dice": d, "iou": i}


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SAM with Hugging Face Trainer and a validation split.")
    p.add_argument("--dataset_name", type=str, default="nielsr/breast-cancer",
                   help="HF dataset name or path.")
    p.add_argument("--dataset_split", type=str, default="train",
                   help="Split to load when the dataset doesn't provide train/val.")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Held-out fraction if the dataset lacks a validation split.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_name", type=str, default="facebook/sam-vit-base")
    p.add_argument("--output_dir", type=str, default="./sam_trainer_runs")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train_bs", type=int, default=2)
    p.add_argument("--eval_bs", type=int, default=2)
    p.add_argument("--fp16", action="store_true", help="Enable fp16 training explicitly.")
    return p.parse_args()


def main():
    args = parse_args()

    # Dataset load + split
    raw = load_dataset(args.dataset_name, split=args.dataset_split)
    if isinstance(raw, DatasetDict):
        raise ValueError("Unexpected DatasetDict from split load. Provide a split that yields a single split dataset.")
    splits = raw.train_test_split(test_size=args.val_ratio, seed=args.seed)
    dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})

    # Processor + PT datasets
    processor = SamProcessor.from_pretrained(args.model_name)
    train_dataset = SAMDataset(dataset["train"], processor)
    val_dataset = SAMDataset(dataset["validation"], processor)

    # Model (freeze encoders; train mask decoder)

    base = SamConfig.from_pretrained(args.model_name)
    base.vision_config.attention_dropout = 0.1 
    model = SamModel.from_pretrained(args.model_name, config=base)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Training args
    use_fp16 = args.fp16 or torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="dice",
        greater_is_better=True,
        fp16=use_fp16,
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = SamTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=sam_data_collator,
        compute_metrics=compute_metrics,
    )

    # Train + evaluate
    train_result = trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()

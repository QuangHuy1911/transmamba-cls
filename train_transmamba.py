# Training Script for TransMamba-Cls
# With separate LRs, warmup scheduler, improved defaults
#
# Usage:
#   python train_transmamba.py --task sst2 --fusion cross_attention --epochs 5
#   python train_transmamba.py --task sst2 --encoder bert-small --epochs 5
#   python train_transmamba.py --task rte --epochs 15 --encoder_lr 2e-5 --decoder_lr 5e-4
#
# Features:
#   - Separate LR for encoder (2e-5) and decoder (5e-4)
#   - Warmup scheduler (10% warmup + cosine decay)
#   - Default 8 Mamba layers (50% of paper's 16L)
#   - Default encoder: bert-small (50% of paper's 8L custom)
#   - Support encoder sizes: bert-tiny, bert-small

import argparse
import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.transmamba_cls import TransMambaClassifier
from data.glue_loader import get_glue_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup + Cosine decay — stable hơn pure cosine."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        output = model(input_ids, attention_mask, labels)
        loss = output["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = output["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Show current LR
        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


@torch.no_grad()
def evaluate(model, loader, device, num_labels):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        output = model(input_ids, attention_mask, labels)
        total_loss += output["loss"].item()
        
        preds = output["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds) if num_labels == 2 else f1_score(all_labels, all_preds, average="macro")
    
    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train TransMamba-Cls")
    
    # Task & Data
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mnli", "rte"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    
    # Model
    parser.add_argument("--encoder", type=str, default="prajjwal1/bert-small",
                        help="Encoder: bert-tiny, bert-small, or HF model name")
    parser.add_argument("--n_mamba_layers", type=int, default=8)    # Gần paper (8L vs paper 16L)
    parser.add_argument("--fusion", type=str, default="cross_attention",
                        choices=["cross_attention", "cross_attention_simple", "additive", "none"])
    parser.add_argument("--freeze_encoder", action="store_true")
    
    # Training — separate LRs
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--encoder_lr", type=float, default=2e-5,
                        help="Learning rate for pretrained encoder (fine-tuning)")
    parser.add_argument("--decoder_lr", type=float, default=5e-4,
                        help="Learning rate for decoder+fusion (train from scratch)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio (10%% of total steps)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"TransMamba-Cls Training")
    print(f"  Task: {args.task.upper()}")
    print(f"  Encoder: {args.encoder}")
    print(f"  Mamba layers: {args.n_mamba_layers}")
    print(f"  Fusion: {args.fusion}")
    print(f"  Encoder LR: {args.encoder_lr} | Decoder LR: {args.decoder_lr}")
    print(f"  Warmup: {args.warmup_ratio*100:.0f}%")
    print(f"  Device: {device}")
    print("=" * 60)
    
    # Data
    train_loader, val_loader, num_labels = get_glue_dataloaders(
        task=args.task, batch_size=args.batch_size, max_length=args.max_length,
    )
    
    # Model
    model = TransMambaClassifier(
        encoder_name=args.encoder,
        n_mamba_layers=args.n_mamba_layers,
        num_labels=num_labels,
        fusion=args.fusion,
        freeze_encoder=args.freeze_encoder,
    ).to(device)
    
    info = model.get_model_info()
    print(f"\nModel: TransMamba-Cls ({args.fusion})")
    print(f"  Total params: {info['total_params']:,}")
    print(f"  Encoder: {info['encoder_params']:,}")
    print(f"  Mamba Decoder: {info['mamba_decoder_params']:,}")
    print(f"  Fusion: {info['fusion_params']:,}")
    print(f"  Classifier: {info['classifier_params']:,}")
    
    # Optimizer — Separate LRs
    param_groups = model.get_param_groups(
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
    )
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Scheduler — Linear warmup + Cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    
    print(f"  Total steps: {total_steps} | Warmup: {warmup_steps}")
    
    # Training
    best_acc = 0
    results = []
    start_time = time.time()
    
    freeze_suffix = "_frozen" if args.freeze_encoder else ""
    output_dir = f"results/transmamba_{args.task}_{args.fusion}{freeze_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip
        )
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, num_labels)
        epoch_time = time.time() - epoch_start
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | F1: {val_f1*100:.2f}%")
        print(f"Time: {epoch_time:.1f}s")
        
        result = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "time": epoch_time,
        }
        results.append(result)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            print(f"  ⭐ New best! (Acc: {best_acc*100:.2f}%)")
    
    total_time = time.time() - start_time
    
    # Save results
    final_results = {
        "model": "TransMamba-Cls",
        "version": "full",
        "task": args.task,
        "fusion": args.fusion,
        "encoder": args.encoder,
        "best_val_acc": best_acc,
        "total_params": info["total_params"],
        "total_time_minutes": total_time / 60,
        "seed": args.seed,
        "epochs": results,
        "config": vars(args),
        "model_info": info,
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Model: TransMamba-Cls ({args.fusion})")
    print(f"  Task: {args.task.upper()}")
    print(f"  Best Val Acc: {best_acc*100:.2f}%")
    print(f"  Total Time: {total_time/60:.1f} minutes")
    print(f"  Results: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

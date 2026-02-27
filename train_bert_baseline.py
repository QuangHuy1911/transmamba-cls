# Training Script for BERT-tiny Baseline (Pure Transformer)
# Usage: python train_bert_baseline.py --task sst2 --epochs 5

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
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import BertForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.glue_loader import get_glue_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, device, num_labels):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds) if num_labels == 2 else f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train BERT Baseline")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mnli", "rte"])
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print(f"BERT Baseline Training")
    print(f"  Model: {args.model_name}")
    print(f"  Task: {args.task.upper()} | Device: {device}")
    print("=" * 60)
    
    train_loader, val_loader, num_labels = get_glue_dataloaders(
        task=args.task, batch_size=args.batch_size, max_length=args.max_length,
    )
    
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)
    
    best_acc = 0
    results = []
    start_time = time.time()
    
    model_short = args.model_name.split("/")[-1]
    output_dir = f"results/bert_{model_short}_{args.task}"
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, num_labels)
        epoch_time = time.time() - epoch_start
        
        print(f"Train: {train_acc*100:.2f}% | Val: {val_acc*100:.2f}% | F1: {val_f1*100:.2f}% | {epoch_time:.0f}s")
        
        results.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "time": epoch_time,
        })
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
    
    total_time = time.time() - start_time
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "model": f"BERT-{model_short}", "task": args.task, "best_val_acc": best_acc,
            "total_params": total_params, "total_time_minutes": total_time / 60,
            "seed": args.seed, "epochs": results, "config": vars(args),
        }, f, indent=2)
    
    print(f"\nDone! Best Acc: {best_acc*100:.2f}% | Time: {total_time/60:.1f}min")


if __name__ == "__main__":
    main()

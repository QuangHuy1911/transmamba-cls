# Ablation Study Runner v2
# Chạy tất cả experiments: fusion types, encoder sizes, frozen/unfrozen

import subprocess
import sys
import os
import json
from datetime import datetime


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

ABLATION_CONFIGS = [
    # === Fusion Ablation (bert-small default, 8L decoder) ===
    {
        "name": "TransMamba-small (cross_attention)",
        "script": "train_transmamba.py",
        "args": ["--fusion", "cross_attention", "--n_mamba_layers", "8"],
        "description": "FULL v2: bert-small + 8L + FeatureProj + CrossAttn (giống paper)",
    },
    {
        "name": "TransMamba-small (cross_attention_simple)",
        "script": "train_transmamba.py",
        "args": ["--fusion", "cross_attention_simple", "--n_mamba_layers", "8"],
        "description": "v1 simple cross-attn (no feature projection, for ablation)",
    },
    {
        "name": "TransMamba-small (additive)",
        "script": "train_transmamba.py",
        "args": ["--fusion", "additive", "--n_mamba_layers", "8"],
        "description": "Additive fusion (no attention, for ablation)",
    },
    {
        "name": "TransMamba-small (none)",
        "script": "train_transmamba.py",
        "args": ["--fusion", "none", "--n_mamba_layers", "8"],
        "description": "No fusion (Mamba output only, for ablation)",
    },
    {
        "name": "TransMamba-small (frozen encoder)",
        "script": "train_transmamba.py",
        "args": ["--fusion", "cross_attention", "--n_mamba_layers", "8", "--freeze_encoder"],
        "description": "Freeze encoder, only train decoder + fusion",
    },
    # === Encoder Scaling (2 sizes, cross_attention fusion, 8L decoder) ===
    {
        "name": "TransMamba-tiny (encoder scaling)",
        "script": "train_transmamba.py",
        "args": ["--encoder", "bert-tiny", "--fusion", "cross_attention", "--n_mamba_layers", "8"],
        "description": "bert-tiny 2L/128d (~5M) — lightweight baseline",
    },
    # bert-small is already the default in fusion ablation above
    # === Baselines ===
    {
        "name": "BERT-tiny Baseline",
        "script": "train_bert_baseline.py",
        "args": [],
        "description": "Pure BERT-tiny fine-tuning (4.4M params)",
    },
    {
        "name": "Pure Mamba Baseline",
        "script": "train_mamba_baseline.py",
        "args": [],
        "description": "Pure Mamba (PureSSM, 9.7M params)",
    },
]


def run_experiment(config, task, epochs, seed=42):
    """Run a single experiment."""
    cmd = [sys.executable, config["script"], "--task", task, "--epochs", str(epochs), "--seed", str(seed)]
    cmd.extend(config["args"])
    
    print(f"\n{'='*70}")
    print(f"  Experiment: {config['name']}")
    print(f"  Description: {config['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"  ❌ FAILED! (exit code: {result.returncode})")
        return False
    
    print(f"  ✅ Done!")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Ablation Studies v2")
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mnli", "rte"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="all",
                        help="Which configs to run: all, transmamba, baselines, or config index")
    args = parser.parse_args()
    
    # Select configs
    if args.config == "all":
        configs = ABLATION_CONFIGS
    elif args.config == "transmamba":
        configs = [c for c in ABLATION_CONFIGS if "TransMamba" in c["name"]]
    elif args.config == "baselines":
        configs = [c for c in ABLATION_CONFIGS if "Baseline" in c["name"]]
    else:
        idx = int(args.config)
        configs = [ABLATION_CONFIGS[idx]]
    
    print(f"\n{'#'*70}")
    print(f"  ABLATION STUDY v2 — {args.task.upper()}")
    print(f"  Experiments: {len(configs)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")
    print(f"{'#'*70}")
    
    results = {}
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config['name']}")
        success = run_experiment(config, args.task, args.epochs, args.seed)
        results[config["name"]] = "✅" if success else "❌"
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — {args.task.upper()}")
    print(f"{'='*70}")
    for name, status in results.items():
        print(f"  {status} {name}")
    
    print(f"\nRun 'python compare_results.py' to compare results.")


if __name__ == "__main__":
    main()

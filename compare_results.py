# Compare Results — All Models on GLUE (3 Encoder Sizes)
# Usage: python compare_results.py
#
# Supports: TransMamba-tiny/small/base, BERT baselines, Pure Mamba baseline
# Output: Console table + LaTeX table for paper

import json
import os
import glob


def load_results(results_dir="results"):
    """Load all results.json from results subdirectories."""
    all_results = []
    
    for result_file in glob.glob(f"{results_dir}/*/results.json"):
        with open(result_file) as f:
            data = json.load(f)
            data["result_file"] = result_file
            all_results.append(data)
    
    return all_results


def get_model_display_name(r):
    """Get display name including encoder info and fusion type."""
    model = r.get("model", "Unknown")
    
    # Add encoder info for TransMamba models
    config = r.get("config", {})
    encoder = config.get("encoder", "")
    if encoder and "TransMamba" in model:
        if "tiny" in encoder:
            model = model.replace("TransMamba", "TransMamba-tiny")
        elif "small" in encoder:
            model = model.replace("TransMamba", "TransMamba-small")
        elif "base" in encoder:
            model = model.replace("TransMamba", "TransMamba-base")
    
    # Add fusion type
    fusion = r.get("fusion", config.get("fusion", ""))
    if fusion:
        model += f" ({fusion})"
    
    return model


def print_comparison(results):
    """Print comparison table grouped by task."""
    if not results:
        print("No results found! Run training scripts first.")
        print("\nQuick start:")
        print("  python train_transmamba.py --task sst2 --epochs 5")
        print("  python train_bert_baseline.py --task sst2 --epochs 5")
        print("  python train_mamba_baseline.py --task sst2 --epochs 5")
        return
    
    # Group by task
    tasks = sorted(set(r["task"] for r in results))
    
    for task in tasks:
        task_results = [r for r in results if r["task"] == task]
        task_results.sort(key=lambda x: x["best_val_acc"], reverse=True)
        
        print(f"\n{'='*80}")
        print(f"  TASK: {task.upper()}")
        print(f"{'='*80}")
        print(f"{'Model':<40} {'Accuracy':>10} {'Params':>12} {'Time (min)':>12}")
        print(f"{'-'*80}")
        
        for r in task_results:
            model = get_model_display_name(r)
            acc = r["best_val_acc"] * 100
            params = r.get("total_params", 0)
            time_min = r.get("total_time_minutes", 0)
            
            print(f"{model:<40} {acc:>9.2f}% {params:>11,} {time_min:>11.1f}")
        
        # Winner
        best = task_results[0]
        print(f"\n  🏆 Winner: {get_model_display_name(best)} ({best['best_val_acc']*100:.2f}%)")
    
    # Encoder scaling summary (if multiple encoders exist)
    print_encoder_scaling(results)


def print_encoder_scaling(results):
    """Print encoder scaling comparison if multiple encoder sizes exist."""
    configs = [r.get("config", {}) for r in results]
    encoders = set(c.get("encoder", "") for c in configs if c.get("encoder"))
    
    if len(encoders) <= 1:
        return
    
    print(f"\n{'='*80}")
    print(f"  ENCODER SCALING ANALYSIS")
    print(f"{'='*80}")
    
    tasks = sorted(set(r["task"] for r in results))
    
    for task in tasks:
        task_results = [r for r in results if r["task"] == task]
        encoder_results = [r for r in task_results if "TransMamba" in r.get("model", "")]
        
        if len(encoder_results) <= 1:
            continue
        
        print(f"\n  {task.upper()}:")
        encoder_results.sort(key=lambda x: x.get("total_params", 0))
        
        for r in encoder_results:
            encoder = r.get("config", {}).get("encoder", "unknown")
            acc = r["best_val_acc"] * 100
            params = r.get("total_params", 0)
            print(f"    {encoder:<25} → {acc:.2f}% ({params:,} params)")


def generate_latex_table(results):
    """Generate LaTeX table for paper."""
    if not results:
        return
    
    tasks = sorted(set(r["task"] for r in results))
    
    print(f"\n{'='*80}")
    print("  LATEX TABLE (for paper)")
    print(f"{'='*80}")
    
    header = "Model & " + " & ".join(t.upper() for t in tasks) + " & Params \\\\"
    print(f"\\begin{{tabular}}{{l{'c' * len(tasks)}r}}")
    print("\\toprule")
    print(header)
    print("\\midrule")
    
    # Group by model display name
    models = {}
    for r in results:
        model = get_model_display_name(r)
        if model not in models:
            models[model] = {"params": r.get("total_params", 0)}
        models[model][r["task"]] = r["best_val_acc"]
    
    # Sort: baselines first, then by params
    sorted_models = sorted(models.items(), key=lambda x: x[1]["params"])
    
    for model, data in sorted_models:
        cells = [model]
        for task in tasks:
            if task in data:
                cells.append(f"{data[task]*100:.1f}")
            else:
                cells.append("-")
        cells.append(f"{data['params']:,}")
        print(" & ".join(cells) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    results = load_results()
    print_comparison(results)
    generate_latex_table(results)

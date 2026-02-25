# 🧬 TransMamba-Cls — Encoder Scaling for Text Classification

**Hybrid Transformer + Mamba for Text Classification on GLUE Benchmark**

> Based on Zhu et al. "TransMamba" (2025)  
> Fusion:3 encoder sizes cho scaling analysis

---

## Kiến trúc

```
Input → BERT Encoder (tiny/small/base) → E
         ├→ TransformerProj(Linear→SiLU→Linear) → E'    
         └→ MambaDecoder (8L PureSSM, Pre-norm RMSNorm) → H
              └→ MambaProj(Conv1x1→SiLU→Conv1x1) → H'  
                   └→ CrossAttention(Q=H', K=E', V=E')  
                        └→ MeanPool → RMSNorm → Classifier
```

## 3 Encoder Sizes

| Encoder | HuggingFace | Layers | Hidden | Params | Use case |
|:--------|:------------|:-------|:-------|:-------|:---------|
| `bert-tiny` | prajjwal1/bert-tiny | 2 | 128 | ~5M | Ablation nhanh |
| `bert-small` | prajjwal1/bert-small | 4 | 512 | ~30M |  **Main results** |
| `bert-base` | bert-base-uncased | 12 | 768 | ~115M |  Best quality |

## Setup

```bash
cd transmamba_project
pip install -r requirements.txt
# Dependencies: torch, transformers, datasets, tqdm, scikit-learn
```

## Quick Start

```bash
# Main results (bert-small — mặc định)
python train_transmamba.py --task sst2 --epochs 5

# Ablation (bert-tiny — nhanh, CPU ok)
python train_transmamba.py --task sst2 --encoder bert-tiny --epochs 5

# Best quality (bert-base — cần GPU)
python train_transmamba.py --task sst2 --encoder bert-base --epochs 3 --batch_size 16

# Baselines
python train_bert_baseline.py --task sst2 --epochs 5
python train_mamba_baseline.py --task sst2 --epochs 5

# Ablation tất cả (9 experiments: 5 fusion + 2 encoder scaling + 2 baselines)
python run_ablation.py --task sst2 --epochs 5

# So sánh kết quả + tạo LaTeX table + Encoder Scaling Analysis
python compare_results.py
```

## v1 Results (bert-tiny, 2L decoder)

| Model | SST-2 | MNLI | RTE | Params |
|:------|:------|:-----|:----|:-------|
| BERT-tiny | 81.65% | 61.95% | 55.23% | 4.4M |
| Pure Mamba | 83.60% | 61.72% | 53.07% | 9.7M |
| TransMamba v1 | 82.91% | 63.04% | — | 4.7M |

## Project Structure

```
transmamba_project/
├── models/
│   ├── transmamba_cls.py         # TransMamba-Cls (default: bert-small, 8L decoder)
│   └── mamba_baseline.py         # PureSSM baseline
├── data/glue_loader.py           # GLUE datasets (SST-2, MNLI, RTE)
├── train_transmamba.py           # Training: 3 encoder sizes × 4 fusion types
├── train_bert_baseline.py        # BERT baseline training
├── train_mamba_baseline.py       # Pure Mamba baseline training
├── run_ablation.py               # 9 experiments (fusion + encoder scaling + baselines)
├── compare_results.py            # Results comparison + LaTeX table + Encoder Scaling
├── requirements.txt              # Dependencies (torch, transformers, datasets, ...)
├── .gitignore                    # Git ignore file (venv, data, results, cache)
├── huongdan.md                   # Hướng dẫn chi tiết (Vietnamese)
└── results/                      # Output JSON từ training
```

## Documentation

| File | Nội dung |
|:-----|:---------|
| [transmamba.md](../transmamba.md) | Paper comparison + fusion + trade-off analysis |
| [plan.md](../plan.md) | Công việc hàng ngày (2 người) |
| [train_hybrid.md](../train_hybrid.md) | Training commands (3 phases) |
| [ke_hoach.md](../ke_hoach.md) | Kế hoạch 30 ngày |
| [huongdan.md](huongdan.md) | Hướng dẫn chi tiết + troubleshooting |

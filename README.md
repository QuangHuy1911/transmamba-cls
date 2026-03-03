# 🧬 TransMamba-Cls

**Adapting Hybrid Transformer-Mamba Architecture for Text Classification on GLUE Benchmark**

> Based on Zhu et al. "TransMamba" (2025) — adapted for GLUE text classification  
> Paper: APWeb-WAIM 2026

---

## Architecture

```
Input → BERT Encoder (pretrained) → E
         ├→ TransformerProj(Linear→SiLU→Linear) → E'
         └→ MambaDecoder (8L PureSSM, Pre-norm RMSNorm) → H
              └→ MambaProj(Conv1x1→SiLU→Conv1x1) → H'
                   └→ CrossAttention(Q=H', K=E', V=E')
                        └→ MeanPool → RMSNorm → Classifier
```

## Results

### Main Results (GLUE Dev Set)

| Model | SST-2 | RTE | Params |
|:------|:------|:----|:-------|
| BERT-tiny (baseline) | 81.08% | 55.60% | 4.4M |
| Pure Mamba (baseline) | 82.80% | 53.07% | 9.7M |
| **TransMamba-tiny** | 83.49% | 57.04% | 5.5M |
| **TransMamba-small** | **87.73%** | **62.09%** | 43.6M |
| TransMamba-small (frozen) | 84.17% | 56.32% | 17.0M |

### Ablation Study (Fusion Strategies, BERT-small + 8L decoder)

| Fusion | SST-2 | RTE |
|:-------|:------|:----|
| None (Mamba only) | 87.16% | 60.29% |
| Additive | 87.73% | 58.48% |
| Cross-Attn (Simple) | 87.73% | **62.09%** |
| Cross-Attn + Proj | 84.17% | 56.32% |

### Key Findings
- TransMamba > Pure Mamba > BERT on both tasks
- Encoder scaling (tiny→small): +4–5% improvement
- Fusion is task-dependent: simple methods work for short sequences, cross-attention benefits longer sequences

## Encoder Configurations

| Encoder | HuggingFace | Layers | Hidden | Params |
|:--------|:------------|:-------|:-------|:-------|
| BERT-tiny | prajjwal1/bert-tiny | 2 | 128 | 4.4M |
| BERT-small | prajjwal1/bert-small | 4 | 512 | 28.8M |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Main experiment (bert-small, default)
python train_transmamba.py --task sst2 --epochs 5

# Encoder scaling (bert-tiny)
python train_transmamba.py --task sst2 --encoder bert-tiny --epochs 5

# Baselines
python train_bert_baseline.py --task sst2 --epochs 5
python train_mamba_baseline.py --task sst2 --epochs 5

# Full ablation (8 experiments: 4 fusion + encoder scaling + frozen + 2 baselines)
python run_ablation.py --task sst2 --epochs 5
python run_ablation.py --task rte --epochs 15

# Compare results + generate LaTeX table
python compare_results.py
```

## Project Structure

```
transmamba_project/
├── models/
│   ├── transmamba_cls.py         # TransMamba-Cls model
│   └── mamba_baseline.py         # PureSSM baseline
├── data/glue_loader.py           # GLUE datasets (SST-2, RTE)
├── train_transmamba.py           # TransMamba training script
├── train_bert_baseline.py        # BERT baseline training
├── train_mamba_baseline.py       # Pure Mamba baseline training
├── run_ablation.py               # All experiments (fusion + scaling + baselines)
├── compare_results.py            # Results comparison + LaTeX table
├── code.md                       # Full LaTeX paper source
├── references.bib                # BibTeX references (20 entries)
├── requirements.txt              # Dependencies
└── results/                      # Experiment results (JSON)
```

## Citation

```bibtex
@inproceedings{truong2026transmambacls,
  title={TransMamba-Cls: Adapting Hybrid Transformer-Mamba Architecture
         for Text Classification on GLUE Benchmark},
  author={Truong, Long and Nguyen, Quang Huy and Le, Ngoc Thuong and Vo, Tan Dat},
  booktitle={Proceedings of APWeb-WAIM},
  year={2026}
}
```

## License

This project is for academic research purposes.

# 📖 Hướng Dẫn Chi Tiết — TransMamba-Cls

> **3 Encoder Sizes:** bert-tiny (ablation) → bert-small (⭐main) → bert-base (best)  
> **Decoder:** 8L PureSSM | **Fusion:** FeatureProj + CrossAttn (giống paper 100%)

---

## 🔧 SETUP

```bash
cd transmamba_project

# Cài dependencies
pip install -r requirements.txt
# Bao gồm: torch, transformers, datasets, tqdm, scikit-learn
# BERT encoders sẽ tự download từ HuggingFace khi chạy lần đầu
```

---

## ⚡ CHẠY NHANH

### 1. Test model (kiểm tra code ok)

```bash
python models/transmamba_cls.py
# → Test cả 4 fusion types + 2 encoder presets (bert-tiny, bert-small)
# → Hiển thị params, forward pass, param groups
```

### 2. Train (bert-small — mặc định, ⭐ main results)

```bash
python train_transmamba.py --task sst2 --epochs 5
```

### 3. Train với encoder khác

```bash
# bert-tiny (nhanh, CPU ok, cho ablation)
python train_transmamba.py --task sst2 --encoder bert-tiny --epochs 5

# bert-base (chất lượng cao, cần GPU)
python train_transmamba.py --task sst2 --encoder bert-base --epochs 3 --batch_size 16
```

---

## 📋 QUY TRÌNH ĐẦY ĐỦ

### Phase 1: Ablation nhanh (bert-tiny, CPU ok)

```bash
# 4 fusion types — xác định fusion nào tốt nhất
python train_transmamba.py --task sst2 --encoder bert-tiny --fusion cross_attention --epochs 5
python train_transmamba.py --task sst2 --encoder bert-tiny --fusion cross_attention_simple --epochs 5
python train_transmamba.py --task sst2 --encoder bert-tiny --fusion additive --epochs 5
python train_transmamba.py --task sst2 --encoder bert-tiny --fusion none --epochs 5

# Hoặc chạy tất cả ablation một lệnh:
python run_ablation.py --task sst2 --epochs 5
```

### Phase 2: Main results (bert-small, cần GPU)

```bash
python train_transmamba.py --task sst2 --encoder bert-small --epochs 5
python train_transmamba.py --task mnli --encoder bert-small --epochs 3
python train_transmamba.py --task rte --encoder bert-small --epochs 15 --encoder_lr 3e-4 --decoder_lr 5e-4
```

### Phase 3: bert-base (nếu có GPU mạnh)

```bash
python train_transmamba.py --task sst2 --encoder bert-base --epochs 3 --batch_size 16
```

### Phase 4: Baselines + Compare

```bash
# Baselines
python train_bert_baseline.py --task sst2 --epochs 5
python train_mamba_baseline.py --task sst2 --epochs 5

# So sánh kết quả (tự nhận diện encoder size, tạo LaTeX table cho paper)
python compare_results.py
# → Console table: sort by accuracy, hiển thị encoder size
# → Encoder Scaling Analysis: so sánh bert-tiny vs bert-small vs bert-base
# → LaTeX table: copy-paste vào paper
```

---

## 📂 FILE QUAN TRỌNG

| File | Vai trò |
|:-----|:--------|
| `models/transmamba_cls.py` | Model chính (default: bert-small, 8L decoder) |
| `models/mamba_baseline.py` | PureSSM baseline model |
| `train_transmamba.py` | Training script (3 encoder sizes, 4 fusion types) |
| `train_bert_baseline.py` | BERT-only baseline |
| `train_mamba_baseline.py` | Pure Mamba baseline |
| `run_ablation.py` | 9 experiments: 5 fusion + 2 encoder scaling + 2 baselines |
| `compare_results.py` | So sánh kết quả + LaTeX table + Encoder Scaling Analysis |
| `requirements.txt` | Dependencies (torch, transformers, datasets, tqdm, sklearn) |
| `data/glue_loader.py` | GLUE dataset loader (SST-2, MNLI, RTE) |
| `results/` | Output JSON từ training |

---

## ⚠️ Troubleshooting

| Vấn đề | Giải pháp |
|:-------|:----------|
| `pip install` lỗi | `pip install -r requirements.txt` |
| CUDA OOM bert-small | `--batch_size 16` hoặc `--max_length 64` |
| CUDA OOM bert-base | `--batch_size 8` hoặc dùng Google Colab |
| Quá chậm CPU | Dùng `--encoder bert-tiny` hoặc Colab T4 |
| Không converge RTE | `--encoder_lr 3e-4 --decoder_lr 5e-4 --epochs 15` |
| Training instability | `--decoder_lr 5e-4` (giảm từ 1e-3) |

---

## ☁️ ĐƯA PROJECT LÊN GITHUB

Để chia sẻ project hoặc lưu trữ, bạn có thể đưa lên GitHub theo các bước sau:

1. **Khởi tạo:** `git init`
2. **Add file:** `git add .` (File `.gitignore` sẽ tự bỏ qua các file nặng)
3. **Commit:** `git commit -m "Initial commit"`
4. **Push:** Tạo repo trên GitHub và chạy lệnh `remote add` rồi `push`.

> 💡 Xem hướng dẫn chi tiết tại: [github_guide.md](file:///C:/Users/ngoqu/.gemini/antigravity/brain/c33aea78-1a80-47d4-97a6-fcdf66a2cd23/github_guide.md)

---

## 🔗 Tài liệu

- [transmamba.md](../transmamba.md) — Paper comparison + fusion pipeline + trade-off analysis
- [plan.md](../plan.md) — Công việc từng ngày (2 người)
- [train_hybrid.md](../train_hybrid.md) — Training commands 3 phases
- [ke_hoach.md](../ke_hoach.md) — Kế hoạch 30 ngày

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
| `run_ablation.py` | Chạy tự động 9 experiments (xem chi tiết bên dưới ⬇️) |
| `compare_results.py` | So sánh kết quả + LaTeX table + Encoder Scaling Analysis |
| `requirements.txt` | Dependencies (torch, transformers, datasets, tqdm, sklearn) |
| `data/glue_loader.py` | GLUE dataset loader (SST-2, MNLI, RTE) |
| `results/` | Output JSON từ training |

---

## 🔬 ABLATION STUDY LÀ GÌ? (Giải thích `run_ablation.py`)

### Ablation study nghĩa là gì?

**Ablation study** = thí nghiệm "bỏ từng phần" để xem phần nào THỰC SỰ quan trọng.

Ví dụ dễ hiểu: Bạn nấu phở với 5 gia vị. Muốn biết gia vị nào quan trọng nhất → nấu 5 nồi, mỗi nồi bỏ 1 gia vị → nồi nào dở nhất → gia vị đó quan trọng nhất.

Trong TransMamba, ta muốn chứng minh: **Feature Fusion (theo paper) thực sự cải thiện kết quả**, không phải chỉ do encoder mạnh hay decoder nhiều layers.

### `run_ablation.py` chạy 9 experiments — chia 3 nhóm:

#### Nhóm 1: Fusion Ablation (5 experiments) — "Loại fusion nào tốt nhất?"

Giữ nguyên encoder (bert-small) + decoder (8L), chỉ **thay đổi cách kết hợp** encoder + decoder:

| # | Experiment | Fusion | Mục đích |
|:--|:-----------|:-------|:---------|
| 1 | TransMamba-small (cross_attention) | FeatureProj + CrossAttn | ⭐ **Cấu hình chính** — giống paper 100% |
| 2 | TransMamba-small (cross_attention_simple) | CrossAttn (không projection) | So sánh: projection có quan trọng không? |
| 3 | TransMamba-small (additive) | H + E (cộng đơn giản) | So sánh: cần attention không hay cộng là đủ? |
| 4 | TransMamba-small (none) | Chỉ dùng Mamba output | So sánh: encoder có giúp gì không? |
| 5 | TransMamba-small (frozen encoder) | CrossAttn + đóng băng encoder | So sánh: fine-tune encoder có cần không? |

**Kỳ vọng:** Experiment 1 (cross_attention) sẽ cho kết quả **tốt nhất** → chứng minh Feature Projection + CrossAttn là quan trọng.

#### Nhóm 2: Encoder Scaling (2 experiments) — "Encoder size ảnh hưởng thế nào?"

Giữ nguyên fusion (cross_attention) + decoder (8L), chỉ **thay đổi kích thước encoder**:

| # | Experiment | Encoder | Params | Mục đích |
|:--|:-----------|:--------|:-------|:---------|
| 6 | TransMamba-tiny | bert-tiny (2L/128d) | ~5M | Nhẹ nhất — baseline encoder |
| 7 | TransMamba-base | bert-base (12L/768d) | ~115M | Mạnh nhất — nếu có GPU |

(bert-small đã chạy ở nhóm 1, không cần lặp lại)

**Kỳ vọng:** tiny < small < base → encoder mạnh hơn = accuracy cao hơn.

#### Nhóm 3: Baselines (2 experiments) — "TransMamba có tốt hơn từng phần riêng lẻ không?"

| # | Experiment | Mô tả | Mục đích |
|:--|:-----------|:------|:---------|
| 8 | BERT-tiny Baseline | Chỉ BERT, không Mamba, không fusion | Encoder đơn thuần |
| 9 | Pure Mamba Baseline | Chỉ Mamba, không encoder, không fusion | Decoder đơn thuần |

**Kỳ vọng:** TransMamba > BERT-tiny baseline VÀ TransMamba > Pure Mamba → kết hợp 2 cái = tốt hơn từng cái riêng.

### Cách chạy `run_ablation.py`

```bash
# Chạy TẤT CẢ 9 experiments trên SST-2 (mất ~vài giờ trên GPU)
python run_ablation.py --task sst2 --epochs 5

# Chạy chỉ TransMamba (7 experiments, bỏ baselines)
python run_ablation.py --task sst2 --config transmamba

# Chạy chỉ baselines (2 experiments)
python run_ablation.py --task sst2 --config baselines

# Chạy 1 experiment cụ thể (theo số thứ tự 0-8)
python run_ablation.py --task sst2 --config 0   # ← chỉ cross_attention

# Sau khi chạy xong → so sánh kết quả:
python compare_results.py
```

### Kết quả kỳ vọng (bảng cho paper)

```
┌─────────────────────────────────────────────────────┐
│ Model                     │ SST-2 │ Fusion quan trọng│
├───────────────────────────┼───────┼──────────────────┤
│ BERT-tiny Baseline        │ ~81%  │ —                │
│ Pure Mamba Baseline        │ ~83%  │ —                │
│ TransMamba (none)          │ ~84%  │ Không fusion     │
│ TransMamba (additive)      │ ~85%  │ + fusion đơn giản│
│ TransMamba (cross_simple)  │ ~86%  │ + attention      │
│ TransMamba (cross_attn) ⭐ │ ~89%  │ + projection ⭐  │
└─────────────────────────────────────────────────────┘
→ Kết quả tăng dần chứng minh TỪNG COMPONENT đều contribute
```

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

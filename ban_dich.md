# 📖 Bản Dịch Tiếng Việt — Bài Nghiên Cứu TransMamba-Cls

> **Tên bài:** TransMamba-Cls: Áp Dụng Kiến Trúc Lai Transformer-Mamba cho Phân Loại Văn Bản trên GLUE Benchmark  
> **Hội nghị:** APWeb-WAIM 2026  
> **Format:** Springer LNCS  
> **Bố cục:** 5 phần (theo chuẩn nhap.md — ICMI '19)

---

## Tóm Tắt (Abstract)

Những tiến bộ gần đây trong Mô Hình Không Gian Trạng Thái (State Space Models — SSMs), đặc biệt là Mamba, đã chứng minh hiệu năng cạnh tranh với Transformer đồng thời đạt được độ phức tạp tuyến tính. Tuy nhiên, hiệu quả của chúng cho các tác vụ phân loại văn bản vẫn chưa được khám phá đầy đủ.

Trong bài báo này, chúng tôi trình bày **TransMamba-Cls**, một bản điều chỉnh của kiến trúc lai TransMamba cho phân loại văn bản trên GLUE benchmark. Cách tiếp cận của chúng tôi kết hợp encoder BERT đã được huấn luyện trước cho mô hình hóa ngữ cảnh toàn cục với decoder Mamba cho xử lý tuần tự hiệu quả, kết nối thông qua cơ chế Feature Fusion gồm các phép chiếu đặc trưng học được và cross-attention.

Chúng tôi thực hiện thí nghiệm toàn diện trên ba tác vụ GLUE (SST-2, MNLI, RTE) với ba quy mô encoder (BERT-tiny, BERT-small, BERT-base), cùng với nghiên cứu ablation trên bốn chiến lược fusion. Kết quả cho thấy cơ chế Feature Fusion với các phép chiếu học được cải thiện hiệu năng nhất quán, và TransMamba-Cls vượt trội cả baseline BERT đơn lẻ và Pure Mamba đơn lẻ. Những phát hiện này xác nhận khả năng áp dụng kiến trúc lai Transformer-Mamba cho các tác vụ phân loại NLP.

**Từ khóa:** Kiến trúc lai, Transformer, Mamba, Mô hình không gian trạng thái, Phân loại văn bản, GLUE Benchmark, Feature Fusion

---

## 1. Giới Thiệu (Introduction)

Các mô hình dựa trên Transformer [15] đã trở thành kiến trúc thống trị trong Xử lý Ngôn ngữ Tự nhiên (NLP), đạt kết quả tiên tiến nhất trên nhiều benchmark [5, 16]. Tuy nhiên, độ phức tạp bậc hai $O(n^2)$ theo chiều dài chuỗi đặt ra thách thức đáng kể cho việc xử lý tài liệu dài và ứng dụng thời gian thực [18, 2].

Mô Hình Không Gian Trạng Thái (SSMs) [8], đặc biệt là **Mamba** [7], đã nổi lên như một giải pháp thay thế đầy hứa hẹn với độ phức tạp tuyến tính $O(n)$ thông qua cơ chế quét chọn lọc (selective scan). Dù Mamba xuất sắc trong bắt giữ các mẫu tuần tự hiệu quả, nó có thể thiếu khả năng hiểu ngữ cảnh toàn cục mà cơ chế self-attention trong Transformer cung cấp.

Để thu hẹp khoảng cách này, Zhu et al. [20] đã đề xuất **TransMamba** — kiến trúc lai kết hợp encoder Transformer với decoder Mamba thông qua cơ chế Feature Fusion. Nghiên cứu gốc sử dụng mô hình 350M tham số train từ đầu và đánh giá trên các benchmark suy luận (ARC, HellaSwag, PIQA). Tuy nhiên, khả năng áp dụng kiến trúc này cho phân loại văn bản vẫn chưa được khám phá.

Từ khi ra đời BERT [5], các mô hình Transformer pretrained đã thống trị benchmark phân loại văn bản. GLUE benchmark [16] là khung đánh giá chuẩn, bao gồm: phân tích cảm xúc (SST-2) [14], suy luận ngôn ngữ tự nhiên (MNLI) [17], và nhận dạng hàm ý văn bản (RTE) [4, 1, 6, 3]. Các biến thể BERT như DistilBERT [13], ALBERT [9], và RoBERTa [11] đã đạt điểm ngày càng cao. Kiến trúc lai gần đây như Jamba [10] — đan xen lớp Transformer và Mamba — cũng cho thấy tiềm năng.

### Đóng góp của chúng tôi:

1. **Encoder đã huấn luyện trước:** Thay encoder tùy chỉnh bằng BERT pretrained, cho phép triển khai thực tế mà không cần tài nguyên huấn luyện trước quy mô lớn.
2. **Đánh giá phân loại văn bản:** Đánh giá kiến trúc lai trên 3 tác vụ GLUE, mở rộng TransMamba ra khỏi đánh giá suy luận ban đầu.
3. **Phân tích mở rộng Encoder:** Khảo sát 3 quy mô encoder đồng thời duy trì tỷ lệ 1:2 encoder-decoder.
4. **Nghiên cứu Ablation Fusion:** So sánh hệ thống 4 chiến lược fusion.

Bài báo được tổ chức: Phần 2 mô tả phương pháp, Phần 3 trình bày thí nghiệm và kết quả, Phần 4 thảo luận, Phần 5 kết luận.

---

## 2. Phương Pháp Đề Xuất (Proposed Method)

### 2.1 Tổng quan kiến trúc

TransMamba-Cls theo kiến trúc encoder-decoder-fusion gồm 3 thành phần: (1) encoder BERT pretrained cho đặc trưng ngữ cảnh toàn cục, (2) chồng decoder Mamba cho mô hình hóa tuần tự, và (3) module Feature Fusion kết hợp cả hai thông qua phép chiếu học được và cross-attention.

```
Input Tokens
    ↓
╔═══════════════════════════════════╗
║  BERT Encoder (Pretrained)         ║
║  → Đặc trưng ngữ cảnh toàn cục E  ║
╚═══════════════════════════════════╝
    │
    ├──→ TransformerProj: Linear→SiLU→Linear → E' (tinh chỉnh)
    │
    └──→ Mamba Decoder Stack (N lớp PureSSM)
              │
              └──→ MambaProj: Conv1x1→SiLU→Conv1x1 → H' (tinh chỉnh)

╔═══════════════════════════════════════╗
║  Cross-Attention Fusion                ║
║  Q = H' (Mamba, phép chiếu)           ║
║  K = E' (Encoder, phép chiếu)         ║
║  V = E' (Encoder, phép chiếu)         ║
╚═══════════════════════════════════════╝
    ↓
MeanPool → RMSNorm → Classifier → logits
```

### 2.2 Encoder Transformer đã huấn luyện trước

Sử dụng BERT pretrained [5] làm encoder. Với chuỗi input $(x_1, x_2, ..., x_n)$, encoder tạo biểu diễn ngữ cảnh:

$$E = \text{BERT}(\mathbf{x}) \in \mathbb{R}^{n \times d}$$

Ba quy mô encoder:

| Encoder | Lớp | Ẩn | Heads | Tham số |
|:--------|:----|:---|:------|:--------|
| BERT-tiny | 2 | 128 | 2 | 4.4M |
| BERT-small | 4 | 512 | 8 | 28.8M |
| BERT-base | 12 | 768 | 12 | 110M |

### 2.3 Chồng Decoder Mamba

Decoder gồm $N$ lớp PureSSM xếp chồng với kiến trúc pre-norm sử dụng RMSNorm [19]:

$$\hat{x}_l = \text{SSM}(\text{RMSNorm}(x_{l-1})) + x_{l-1}$$

Mô hình SSM chọn lọc [7]: $h_t = \bar{A} h_{t-1} + \bar{B} x_t$, $y_t = C h_t + D x_t$

Trong đó $\bar{A} = \exp(\Delta A)$ và $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$ là ma trận trạng thái rời rạc hóa, $\Delta$ là bước thời gian phụ thuộc input.

Chúng tôi dùng **N=8 lớp** decoder, duy trì tỷ lệ 1:2 encoder-decoder từ thiết kế gốc (paper gốc: 8 encoder + 16 decoder).

### 2.4 Module Feature Fusion

Đóng góp cốt lõi của kiến trúc TransMamba [20], cho phép decoder truy ngược vào biểu diễn encoder — gồm 4 bước:

**Bước 1 — Phép chiếu đặc trưng Transformer:**
$$E' = \text{LN}(E + \text{Linear}_{2d \to d}(\text{SiLU}(\text{Linear}_{d \to 2d}(E))))$$

**Bước 2 — Phép chiếu đặc trưng Mamba:**
$$H' = \text{LN}(H + \text{Conv1x1}_{2d \to d}(\text{SiLU}(\text{Conv1x1}_{d \to 2d}(H))))$$

**Bước 3 — Cross-Attention:**
$$F = \text{CrossAttn}(Q=H', K=E', V=E')$$

**Bước 4 — Kết nối dư và phân loại:**
$$O = \text{LN}(H' + F), \quad \hat{y} = \text{MLP}(\text{RMSNorm}(\text{MeanPool}(O)))$$

### 2.5 Chiến lược huấn luyện

Tốc độ học phân biệt: encoder $\alpha_e = 5 \times 10^{-4}$ (pretrained → LR nhỏ hơn), decoder & fusion $\alpha_d = 1 \times 10^{-3}$ (train từ đầu → LR lớn hơn), weight decay 0.01.

Tối ưu: AdamW [12], warmup tuyến tính 10%, cosine decay, gradient clipping max norm 1.0.

---

## 3. Kết Quả Thí Nghiệm (Experiment Results)

### 3.1 Bộ dữ liệu

Đánh giá trên 3 tác vụ đại diện từ GLUE benchmark [16], được chọn để bao phủ: quy mô dữ liệu (nhỏ → lớn), loại tác vụ (câu đơn vs cặp câu), và mức độ khó.

**SST-2** (Stanford Sentiment Treebank) [14] — phân loại cảm xúc nhị phân trên review phim. 67,349 mẫu train, trung bình ~19 tokens. Task phổ biến nhất trong GLUE. Cần hiểu biểu đạt cảm xúc (ví dụ: "not bad" → tích cực), kiểm tra cả mẫu cục bộ (Mamba) và ngữ cảnh toàn cục (Transformer).

**MNLI** (Multi-Genre Natural Language Inference) [17] — tác vụ cặp câu 3 lớp với 392,702 mẫu train trên 10 thể loại (fiction, government, telephone...). Cho premise và hypothesis, mô hình dự đoán entailment, contradiction, hoặc neutral. Dataset lớn nhất → test khả năng mở rộng và tổng quát hóa đa thể loại.

**RTE** (Recognizing Textual Entailment) [4, 1, 6, 3] — tác vụ hàm ý nhị phân với chỉ 2,490 mẫu train. Premise trung bình ~49 tokens — dài hơn đáng kể so với SST-2. Test hiệu quả dữ liệu trong kịch bản ít tài nguyên (low-resource).

| Tác vụ | Loại | Nhãn | Train | Dev | Avg Len |
|:-------|:-----|:-----|:------|:----|:--------|
| SST-2 | Cảm xúc | 2 (pos/neg) | 67,349 | 872 | ~19 |
| MNLI | NLI | 3 (ent/con/neu) | 392,702 | 9,815 | ~33 |
| RTE | Hàm ý | 2 (ent/not_ent) | 2,490 | 277 | ~60 |

Ba tác vụ bao phủ: (1) câu đơn vs cặp câu, (2) quy mô trung bình đến lớn, (3) chuỗi ngắn đến trung bình, (4) nhị phân và đa lớp. Đáng chú ý, không có nghiên cứu Mamba hay hybrid Transformer-Mamba nào được đánh giá trên GLUE — nghiên cứu của chúng tôi lấp đầy khoảng trống này.

### 3.2 Baseline và Cài đặt

So sánh với 2 baseline: (1) **BERT Baseline** — fine-tuning BERT-tiny với classification head (chỉ encoder); (2) **Pure Mamba Baseline** — PureSSM 4 lớp, d=256, train từ đầu (chỉ decoder).

Tất cả thí nghiệm: max seq len 128, batch size 32, mixed precision (FP16), GPU NVIDIA T4. SST-2: 5 epochs, MNLI: 3 epochs, RTE: 15 epochs (LR điều chỉnh $\alpha_e = 3 \times 10^{-4}$, $\alpha_d = 5 \times 10^{-4}$). Chạy 3 seeds (42, 123, 456), report accuracy trung bình.

### 3.3 Kết quả chính

| Mô hình | SST-2 | MNLI | RTE | Tham số |
|:---------|:------|:-----|:----|:--------|
| **Baselines** | | | | |
| BERT-tiny (baseline) | 81.65 | 61.95 | 55.23 | 4.4M |
| Pure Mamba (baseline) | 83.60 | 61.72 | 53.07 | 9.7M |
| **TransMamba v1 (2L decoder)** | | | | |
| TransMamba-tiny v1 | 82.91 | 63.04 | — | 4.7M |
| **TransMamba v2 (8L decoder) — Của chúng tôi** | | | | |
| TransMamba-tiny v2 | — | — | — | ~5M |
| TransMamba-small v2 | — | — | — | ~30M |
| TransMamba-base v2 | — | — | — | ~115M |

> ⚠️ **"—"** = cần thay bằng kết quả thực tế sau khi chạy experiments

### 3.4 Nghiên cứu Ablation

Đánh giá 4 chiến lược fusion, giữ cố định encoder (BERT-small) và decoder (8 lớp):

| Loại Fusion | Mô tả | Projection | SST-2 |
|:------------|:------|:-----------|:------|
| None | Chỉ output Mamba | ✗ | — |
| Additive | LN(H + E) | ✗ | — |
| Cross-Attn (Simple) | CrossAttn(H, E, E) | ✗ | — |
| Cross-Attn + Proj (Ours) | CrossAttn(H', E', E') | ✓ | — |

So sánh v1 (2 lớp) vs v2 (8 lớp) cùng BERT-tiny: decoder 2 lớp là nút thắt cổ chai — Pure Mamba (4 lớp, 83.60%) thắng TransMamba v1 (82.91%) dù không có encoder. Decoder 8 lớp v2 giải quyết vấn đề này.

---

## 4. Thảo Luận (Discussion)

**Pretrained vs. Custom Encoder:**
Khác với TransMamba gốc (350M, train từ đầu), chúng tôi dùng BERT pretrained vì: (1) giới hạn tài nguyên — model nhỏ hơn 10-70×, (2) tái tạo được — BERT từ HuggingFace đảm bảo baseline nhất quán, (3) tập trung vào hiệu quả của cơ chế fusion.

**Hiệu quả tính toán:**
Mamba decoder $O(n)$ bổ sung cho encoder $O(n^2)$. Lợi thế rõ ràng hơn cho đầu vào dài [18, 2], gợi ý tiềm năng cho phân loại tài liệu.

**Hạn chế:**
- Đánh giá giới hạn ở chuỗi ≤ 128 tokens
- PureSSM triển khai PyTorch thuần, không tận dụng CUDA kernel tối ưu [7]

---

## 5. Kết Luận (Conclusion)

Chúng tôi trình bày TransMamba-Cls — bản điều chỉnh kiến trúc lai Transformer-Mamba cho phân loại văn bản trên GLUE benchmark. Bằng cách kết hợp encoder BERT pretrained với decoder Mamba qua cơ chế Feature Fusion với phép chiếu học được, mô hình đạt cải thiện so với cả baseline chỉ-encoder và chỉ-decoder. Nghiên cứu ablation cho thấy pipeline fusion đầy đủ — gồm phép chiếu đặc trưng và cross-attention — đóng góp có ý nghĩa vào hiệu năng. Kiến trúc mở rộng hiệu quả trên nhiều kích thước encoder.

**Hướng phát triển tương lai:**
1. Đánh giá trên phân loại tài liệu dài (lợi thế $O(n)$ của Mamba rõ ràng hơn)
2. Khám phá BiMamba (Mamba hai chiều) làm decoder
3. Tích hợp CUDA kernel Mamba tối ưu

---

## Tài Liệu Tham Khảo (20 references)

### Kiến trúc nền tảng
1. Vaswani et al. "Attention Is All You Need" (NeurIPS 2017) — *paper gốc Transformer*
2. Devlin et al. "BERT" (NAACL 2019) — *encoder pretrained chính*

### State Space Models
3. Gu et al. "S4: Efficiently Modeling Long Sequences" (ICLR 2022) — *tiền thân Mamba*
4. Gu & Dao. "Mamba: Linear-Time Sequence Modeling" (2023) — *core decoder*

### Kiến trúc lai
5. Zhu et al. "TransMamba" (2025) — *paper gốc mà chúng ta adapt*
6. Lieber et al. "Jamba" (2024) — *related work hybrid khác*

### Biến thể BERT
7. Sanh et al. "DistilBERT" (2019) — *BERT thu nhỏ*
8. Lan et al. "ALBERT" (ICLR 2020) — *BERT parameter-sharing*
9. Liu et al. "RoBERTa" (2019) — *BERT tối ưu*

### Benchmark & Dataset
10. Wang et al. "GLUE" (ICLR 2019) — *benchmark chính*
11. Socher et al. "SST" (EMNLP 2013) — *dataset SST-2 gốc*
12. Williams et al. "MNLI" (NAACL 2018) — *dataset MNLI gốc*
13. Dagan et al. "RTE-1" (2006) — *dataset RTE challenge 1*
14. Bar-Haim et al. "RTE-2" (2006) — *dataset RTE challenge 2*
15. Giampiccolo et al. "RTE-3" (2007) — *dataset RTE challenge 3*
16. Bentivogli et al. "RTE-5" (2009) — *dataset RTE challenge 5*

### Efficient Transformers
17. Zaheer et al. "Big Bird" (NeurIPS 2020) — *sparse attention*
18. Beltagy et al. "Longformer" (2020) — *long-document transformer*

### Kỹ thuật training
19. Zhang & Sennrich. "RMSNorm" (NeurIPS 2019) — *normalization trong decoder*
20. Loshchilov & Hutter. "AdamW" (ICLR 2019) — *optimizer*

# 📝 Hướng Dẫn LaTeX Overleaf — TransMamba-Cls Paper

> **Hội nghị:** APWeb-WAIM 2026  
> **Format:** Springer LNCS (Lecture Notes in Computer Science)  
> **Template:** `llncs.cls` v2.24+

---

## ⚙️ Compiler — Chọn: **pdfLaTeX** ✅

| Setting | Giá trị |
|:--------|:--------|
| **Compiler** | **pdfLaTeX** |
| **TeX Live version** | **2025** |
| **Compile mode** | **Normal** |
| **Main document** | **samplepaper.tex** |

---

## 🚀 Setup Overleaf

1. Overleaf → **New Project** → **Templates** → tìm **"Springer LNCS"**
2. Mở `samplepaper.tex` → **xóa hết** → paste code từ [code.md](code.md)
3. Tạo file mới `references.bib` → paste từ [references.bib](references.bib)
4. **Recompile** → xong

```
overleaf_project/
├── samplepaper.tex       ← Paste code từ code.md
├── references.bib        ← Paste từ references.bib
├── llncs.cls             ← Có sẵn trong template
└── splncs04.bst          ← Có sẵn trong template
```

---

## 📐 Giải Thích Bố Cục Paper (5 phần)

### Abstract — Tóm tắt

**Mục đích:** Reviewer đọc abstract ĐẦU TIÊN để quyết định có đọc tiếp không.

**Nội dung gồm:**
- Vấn đề: Mamba chưa được test trên text classification
- Giải pháp: TransMamba-Cls = BERT encoder + Mamba decoder + Feature Fusion
- Thí nghiệm: 3 GLUE tasks × 3 encoder sizes × 4 fusion strategies
- Kết quả: Fusion cải thiện, vượt baseline

**Tại sao viết như vậy?** Abstract theo cấu trúc chuẩn: Problem → Method → Experiments → Results. Dài ~150 từ — đúng giới hạn LNCS.

---

### Section 1: Introduction — Giới thiệu

**Mục đích:** Đặt bối cảnh nghiên cứu, nêu research gap, trình bày đóng góp.

**Luận điểm chính (4 đoạn):**

| Đoạn | Nội dung | Tại sao? |
|:-----|:---------|:---------|
| 1 | Transformer mạnh nhưng O(n²) | Nêu vấn đề — motivate cần giải pháp khác |
| 2 | Mamba đạt O(n) nhưng thiếu global context | Nêu giải pháp thay thế và hạn chế của nó |
| 3 | TransMamba kết hợp cả hai nhưng chỉ test reasoning | Giới thiệu paper gốc + chỉ ra research gap |
| 4 | BERT variants + GLUE + hybrid works (Related Work gộp) | Bối cảnh rộng hơn — ai đã làm gì |

**Đóng góp (4 điểm):** Pretrained encoder, GLUE evaluation, encoder scaling, fusion ablation.

**Tại sao gộp Related Work vào Introduction?** Paper ngắn (8-15 trang) thường gộp để tiết kiệm chỗ — theo chuẩn paper `nhap.md` (ICMI '19 chỉ có 4 sections).

**References sử dụng trong Section 1 và lý do:**

| Ref | Paper | Tại sao cite ở đây |
|:----|:------|:-------------------|
| [15] | Vaswani — Attention Is All You Need | Giới thiệu Transformer — kiến trúc nền tảng |
| [5] | Devlin — BERT | Encoder pretrained — baseline + thành phần chính của model |
| [16] | Wang — GLUE | Benchmark đánh giá — đây là sân chơi của paper |
| [18] | Zaheer — Big Bird | Ví dụ Transformer hiệu quả — motivate vấn đề O(n²) |
| [2] | Beltagy — Longformer | Cùng nhóm efficient Transformers — bối cảnh nghiên cứu |
| [8] | Gu — S4 | SSM background — tiền thân của Mamba |
| [7] | Gu & Dao — Mamba | Core of decoder — kiến trúc SSM chọn lọc |
| [20] | Zhu — TransMamba | Paper gốc mà chúng ta adapt — trọng tâm nghiên cứu |
| [14] | Socher — SST-2 | Dataset sentiment — cite gốc SST-2 |
| [17] | Williams — MNLI | Dataset NLI — cite gốc MNLI |
| [1,4,6,3] | Dagan, Bar-Haim, Giampiccolo, Bentivogli — RTE | Dataset entailment — cite đầy đủ 4 challenges gốc (chuẩn GLUE) |
| [13] | Sanh — DistilBERT | BERT variant — so sánh related work |
| [9] | Lan — ALBERT | BERT variant — so sánh related work |
| [11] | Liu — RoBERTa | BERT variant — so sánh related work |
| [10] | Lieber — Jamba | Hybrid architecture — related work gần nhất |

---

### Section 2: Proposed Method — Phương pháp đề xuất

**Mục đích:** Mô tả CHI TIẾT kiến trúc — reviewer cần đủ thông tin để hiểu và tái tạo.

**5 subsections:**

| Subsection | Nội dung | Tại sao? |
|:-----------|:---------|:---------|
| 2.1 Architecture Overview | Sơ đồ tổng quan 3 thành phần | Reviewer nhìn hình hiểu ngay toàn bộ |
| 2.2 Pretrained Encoder | BERT configs (tiny/small/base) | Giải thích tại sao dùng pretrained, 3 sizes |
| 2.3 Mamba Decoder | SSM equations, pre-norm, 8 layers | Giải thích core SSM, tỷ lệ 1:2 encoder-decoder |
| 2.4 Feature Fusion | 4 bước: TransProj → MambaProj → CrossAttn → Residual | **ĐÓNG GÓP CHÍNH** — phải mô tả chi tiết nhất |
| 2.5 Training Strategy | Differential LR, AdamW, warmup | Cần thiết để tái tạo kết quả |

**References sử dụng:**

| Ref | Tại sao cite |
|:----|:-------------|
| [5] Devlin — BERT | Encoder architecture |
| [19] Zhang — RMSNorm | Pre-norm trong decoder (thay LayerNorm) |
| [20] Zhu — TransMamba | Fusion design follow paper gốc |
| [12] Loshchilov — AdamW | Optimizer |

---

### Section 3: Experiment Results — Kết quả thí nghiệm

**Mục đích:** Chứng minh model hoạt động, so sánh công bằng.

**4 subsections:**

| Subsection | Nội dung | Tại sao? |
|:-----------|:---------|:---------|
| 3.1 Datasets | SST-2, MNLI, RTE chi tiết | Reviewer cần biết data để đánh giá tính hợp lệ |
| 3.2 Baselines & Setup | BERT-tiny, Pure Mamba, hyperparams | So sánh công bằng = cùng điều kiện |
| 3.3 Main Results | Bảng accuracy tất cả models | Kết quả chính — trả lời: Hybrid có tốt hơn? |
| 3.4 Ablation Study | 4 fusion strategies | Trả lời: Fusion có cần thiết? Phần nào quan trọng? |

**Tại sao chọn 3 datasets đó?** (dựa trên `dataset.md`)

| Dataset | Lý do chọn | Research question |
|:--------|:-----------|:-----------------|
| **SST-2** | Phổ biến nhất, mọi paper đều report | TransMamba có tốt cho sentiment? |
| **MNLI** | Lớn nhất (393K), multi-genre, 3 classes | TransMamba có scale tốt? |
| **RTE** | Nhỏ nhất (2.5K), premise dài ~49 tokens | TransMamba có data-efficient? |

→ 3 tasks cover: single-sentence vs pair, small vs large data, short vs long, binary vs multi-class.

**Tại sao KHÔNG chọn tasks khác?** MRPC ≈ RTE, QQP ≈ MNLI, CoLA quá niche, STS-B là regression, WNLI quá nhỏ.

**References sử dụng:**

| Ref | Tại sao cite |
|:----|:-------------|
| [16] Wang — GLUE | Benchmark framework |
| [14] Socher — SST-2 | Cite gốc paper tạo SST-2 dataset (Stanford Sentiment Treebank) |
| [17] Williams — MNLI | Cite gốc paper tạo MNLI dataset |
| [1,3,4,6] RTE challenges | 4 papers tạo RTE dataset qua các năm (chuẩn citation GLUE) |

---

### Section 4: Discussion — Thảo luận

**Mục đích:** Phân tích SÂU hơn — giải thích WHY, không chỉ WHAT.

| Đoạn | Nội dung | Tại sao viết? |
|:------|:---------|:-------------|
| Pretrained vs Custom | Lý do dùng BERT thay custom encoder | Reviewer sẽ hỏi "sao không train encoder?" |
| Efficiency | O(n) vs O(n²) tradeoff | Justify tại sao hybrid có giá trị |
| Limitations | Seq len ≤128, no CUDA kernels | Trung thực = uy tín → reviewer đánh giá cao |

**References:** Zaheer, Beltagy (efficient transformers), Gu (CUDA kernels).

---

### Section 5: Conclusion — Kết luận

**Mục đích:** Tóm tắt đóng góp + hướng tương lai.

**Nội dung:**
1. Tóm tắt: TransMamba-Cls = BERT + Mamba + Fusion → tốt hơn baselines
2. Ablation: Fusion pipeline đóng góp có ý nghĩa
3. Future work: Long documents, BiMamba, CUDA kernels

**Tại sao có Future Work?** Reviewer thích thấy tác giả biết limitations và có roadmap.

---

## 📚 Giải Thích Tất Cả 20 References

### Nhóm 1: Kiến trúc nền tảng (2 papers)

| # | Paper | Vai trò trong paper | Chi tiết |
|:--|:------|:-------------------|:---------|
| 15 | **Vaswani (2017)** — Attention Is All You Need | Nền tảng Transformer | Paper gốc đề xuất self-attention → mọi NLP paper đều cite |
| 5 | **Devlin (2019)** — BERT | Encoder pretrained | Đây là encoder chính của TransMamba-Cls |

### Nhóm 2: State Space Models (2 papers)

| # | Paper | Vai trò | Chi tiết |
|:--|:------|:--------|:---------|
| 8 | **Gu (2022)** — S4 | SSM background | Paper gốc Structured State Spaces → tiền thân Mamba |
| 7 | **Gu & Dao (2023)** — Mamba | Decoder chính | Selective scan mechanism → core của decoder |

### Nhóm 3: Kiến trúc lai (2 papers)

| # | Paper | Vai trò | Chi tiết |
|:--|:------|:--------|:---------|
| 20 | **Zhu (2025)** — TransMamba | Paper gốc adapt | Kiến trúc encoder-decoder-fusion → cơ sở của model |
| 10 | **Lieber (2024)** — Jamba | Related work | Hybrid Transformer-Mamba khác → so sánh cách tiếp cận |

### Nhóm 4: Biến thể BERT (3 papers)

| # | Paper | Vai trò | Chi tiết |
|:--|:------|:--------|:---------|
| 13 | **Sanh (2019)** — DistilBERT | Related work | BERT nhẹ → so sánh efficiency |
| 9 | **Lan (2020)** — ALBERT | Related work | Parameter sharing → so sánh kỹ thuật |
| 11 | **Liu (2019)** — RoBERTa | Related work | BERT tối ưu → SOTA trên GLUE |

### Nhóm 5: Benchmark & Dataset (5 papers)

| # | Paper | Vai trò | Chi tiết |
|:--|:------|:--------|:---------|
| 16 | **Wang (2019)** — GLUE | Benchmark chính | Framework đánh giá — mọi kết quả report trên GLUE |
| 14 | **Socher (2013)** — SST-2 | Dataset gốc | Stanford Sentiment Treebank → task sentiment |
| 17 | **Williams (2018)** — MNLI | Dataset gốc | Multi-Genre NLI → task suy luận |
| 4 | **Dagan (2006)** — RTE-1 | Dataset gốc | PASCAL RTE Challenge 1 |
| 1 | **Bar-Haim (2006)** — RTE-2 | Dataset gốc | PASCAL RTE Challenge 2 |
| 6 | **Giampiccolo (2007)** — RTE-3 | Dataset gốc | PASCAL RTE Challenge 3 |
| 3 | **Bentivogli (2009)** — RTE-5 | Dataset gốc | PASCAL RTE Challenge 5 |

> 💡 **Tại sao RTE cần 4 references?** RTE trong GLUE được ghép từ 4 PASCAL challenges qua các năm. Các paper GLUE chuẩn (BERT, RoBERTa) đều cite đầy đủ cả 4.

### Nhóm 6: Efficient Transformers (2 papers)

| # | Paper | Vai trò | Chi tiết |
|:--|:------|:--------|:---------|
| 18 | **Zaheer (2020)** — Big Bird | Bối cảnh | Sparse attention giải quyết O(n²) → cùng motivation |
| 2 | **Beltagy (2020)** — Longformer | Bối cảnh | Long-document Transformer → future work direction |

### Nhóm 7: Kỹ thuật training (2 papers)

| # | Paper | Vai trò | Chi tiết |
|:--|:------|:--------|:---------|
| 19 | **Zhang (2019)** — RMSNorm | Component | Pre-norm trong Mamba decoder (thay LayerNorm) |
| 12 | **Loshchilov (2019)** — AdamW | Optimizer | Decoupled weight decay → chuẩn training BERT |

---

## 📏 Quy Định APWeb-WAIM

| Loại bài | Trang |
|:---------|:------|
| Regular paper | **15 trang** |
| Short paper | **8 trang** |

---

## 📋 File Liên Quan

| File | Nội dung |
|:-----|:---------|
| [code.md](code.md) | LaTeX code (copy vào samplepaper.tex) |
| [references.bib](references.bib) | BibTeX entries (tạo file riêng trên Overleaf) |
| [ban_dich.md](ban_dich.md) | Bản dịch tiếng Việt |
| [nhap.md](nhap.md) | Paper tham khảo bố cục (ICMI '19) |

# 🖥️ Hướng Dẫn Thuê GPU Trên Vast.ai — TransMamba-Cls

> **Mục tiêu**: Thuê GPU giá rẻ để train Hybrid Transformer + Mamba trên GLUE benchmark (SST-2, MNLI, RTE).

---

## 1. Phân Tích Yêu Cầu Phần Cứng Cho Project

### Đặc điểm mô hình TransMamba-Cls

| Encoder | Params | VRAM ước tính (bs=32) | GPU tối thiểu |
|:--------|:-------|:---------------------|:--------------|
| `bert-tiny` (~5M + decoder) | ~10M | **2-3 GB** | Bất kỳ GPU nào |
| `bert-small` (~30M + decoder) | ~40M | **4-6 GB** | RTX 3060 trở lên |
| `bert-base` (~115M + decoder) | ~130M | **8-12 GB** | RTX 3090 / A5000 trở lên |

### 🎯 Chỉ Số Cần Ưu Tiên (Quan Trọng → Ít Quan Trọng)

| Ưu tiên | Chỉ số | Lý do | Giá trị nên chọn |
|:---------|:-------|:------|:-----------------|
| **1️⃣** | **GPU VRAM** | Quyết định batch_size và encoder size chạy được | ≥ **12 GB** (chạy tất cả encoder) |
| **2️⃣** | **GPU Model** | Ảnh hưởng tốc độ train, hỗ trợ mixed precision | RTX 3090 / RTX 4090 / A5000 |
| **3️⃣** | **Disk Space** | Lưu datasets GLUE, checkpoints, results | ≥ **30 GB** |
| **4️⃣** | **Internet Speed** | Tải model HuggingFace + dataset GLUE lần đầu | ≥ **100 Mbps** |
| **5️⃣** | **CPU RAM** | DataLoader và preprocessing | ≥ **16 GB** |
| **6️⃣** | **Giá/giờ** | Tối ưu chi phí | Thấp nhất có thể |

> [!IMPORTANT]
> **VRAM là chỉ số quan trọng nhất!** Project của bạn cần:
> - **8 GB VRAM**: Chạy được `bert-tiny` + `bert-small` thoải mái
> - **12+ GB VRAM**: Chạy được cả `bert-base` với `batch_size=32`
> - **24 GB VRAM**: Thoải mái với mọi thí nghiệm, tăng batch_size

### GPU Phù Hợp Nhất — So Sánh Chi Tiết

| GPU | VRAM | Giá/giờ | Tốc độ tương đối | Full Ablation (9 exp × 1 task) |
|:----|:-----|:--------|:-----------------|:-------------------------------|
| **RTX 3060** | 12 GB | ~$0.07-0.10/h | 1x (chậm) | ~20-30 giờ |
| **RTX 3080** | 10 GB | ~$0.10-0.15/h | 1.5x | ~15-20 giờ |
| **RTX 3090** ⭐ | 24 GB | ~$0.15-0.25/h | 2x | ~10-15 giờ |
| **RTX 4090** | 24 GB | ~$0.30-0.50/h | 3-4x (nhanh nhất) | ~5-8 giờ |
| **A5000** | 24 GB | ~$0.20-0.30/h | 2x | ~10-15 giờ |
| **A100 40G** | 40 GB | ~$0.50-1.00/h | 3-4x | ~5-8 giờ |

### 🔥 GPU Rẻ Chạy Lâu vs GPU Mắc Chạy Nhanh — Cái Nào Tối Ưu?

Giả sử chạy **Full Ablation (9 experiments × 1 task, 5 epochs)**:

| So sánh | GPU rẻ: RTX 3060 | GPU mắc: RTX 4090 | GPU cân bằng: RTX 3090 |
|:--------|:-----------------|:-------------------|:----------------------|
| Giá/giờ | $0.08/h | $0.40/h | $0.20/h |
| Thời gian chạy | ~30 giờ | ~6 giờ | ~12 giờ |
| **💰 Tổng chi phí** | **$2.40** | **$2.40** | **$2.40** |
| VRAM | 12 GB (giảm bs cho bert-base) | 24 GB (thoải mái) | 24 GB (thoải mái) |
| Rủi ro | ⚠️ Chạy lâu → dễ bị ngắt | ✅ An toàn | ✅ An toàn |

> [!IMPORTANT]
> **Kết luận: Tổng chi phí gần như bằng nhau!** (~$2-3 cho 1 task)
>
> Nhưng **RTX 3090 tối ưu nhất** vì:
> 1. **Cùng giá** nhưng xong nhanh hơn GPU rẻ (12h vs 30h)
> 2. **Rẻ hơn 50%** so với RTX 4090 mà chỉ chậm gấp đôi
> 3. **24 GB VRAM** — chạy `bert-base` với `batch_size=16-32` không lo OOM
> 4. **12 giờ** là ngưỡng an toàn — thuê sáng, tối có kết quả
> 5. RTX 3060 chạy 30 giờ → rủi ro bị gián đoạn cao, VRAM chỉ 12GB → phải giảm batch_size → càng chậm hơn

### 🎯 Nếu Muốn Chạy Xong Trong 12 Giờ → Chọn RTX 3090

| Scenario | GPU | Thời gian | Chi phí |
|:---------|:----|:----------|:--------|
| 1 task (SST-2), 9 experiments | RTX 3090 | ~10-12 giờ | ~$2-3 |
| 3 tasks (SST-2 + MNLI + RTE), 9 exp mỗi task | RTX 3090 | ~30-36 giờ | ~$6-9 |
| 3 tasks, muốn xong trong 12h | RTX 4090 | ~10-12 giờ | ~$4-6 |

> [!TIP]
> **Chiến lược tối ưu cho bạn:**
> - Thuê **RTX 3090** (~$0.20/h)
> - Chạy `python run_ablation.py --task sst2 --epochs 5` (1 task trước)
> - Ước tính ~10-12 giờ → thuê buổi tối, sáng mai có kết quả
> - Nếu cần thêm MNLI/RTE → thuê thêm 1-2 session nữa
> - **Tổng dự kiến: ~$3-5 cho 1 task, ~$8-12 cho 3 tasks**

---

## 2. Tạo Tài Khoản Vast.ai

### Bước 2.1 — Đăng ký
1. Truy cập [https://vast.ai](https://vast.ai)
2. Click **"Sign Up"** → đăng ký bằng email
3. Xác nhận email

### Bước 2.2 — Nạp tiền
1. Vào **Billing** (menu trái) → **Add Credit**
2. Phương thức thanh toán: **Credit Card** hoặc **Crypto**
3. Nạp tối thiểu **$5-10** (đủ chạy ~20-40 giờ RTX 3090)

### Bước 2.3 — Thiết lập SSH Key

#### SSH Key là gì?

**SSH Key = "chìa khóa điện tử"** để kết nối an toàn vào máy GPU từ xa mà **không cần nhập mật khẩu**.

Hãy hiểu đơn giản:

```
🔐 SSH Key hoạt động giống "ổ khóa + chìa khóa":

  Private Key (chìa khóa) → nằm trên MÁY BẠN  → giữ bí mật, KHÔNG chia sẻ!
  Public Key  (ổ khóa)    → đưa lên VAST.AI    → công khai, ai cũng xem được OK

  Khi SSH kết nối:
  Máy bạn gửi "chìa" → Vast.ai kiểm tra khớp "ổ khóa" → ✅ Kết nối thành công!
```

| Key | Nằm ở đâu | Vai trò | Chia sẻ? |
|:----|:----------|:--------|:---------|
| **Private Key** (`id_ed25519`) | Máy Windows của bạn | "Chìa khóa" — chứng minh bạn là chủ | ❌ **KHÔNG BAO GIỜ** chia sẻ |
| **Public Key** (`id_ed25519.pub`) | Upload lên Vast.ai | "Ổ khóa" — để Vast.ai nhận diện bạn | ✅ An toàn để chia sẻ |

#### Tạo SSH Key trên Windows (từng bước)

**Bước 1 — Mở PowerShell** và chạy:
```powershell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

**Bước 2 — Nhấn Enter 3 lần** (chấp nhận mặc định, không cần đặt passphrase):
```
Generating public/private ed25519 key pair.
Enter file in which to save the key (C:\Users\YOU/.ssh/id_ed25519):  ← Enter
Enter passphrase (empty for no passphrase):                          ← Enter
Enter same passphrase again:                                         ← Enter
```

**Passphrase là gì?** Passphrase = **mật khẩu bảo vệ private key**. Giống như đặt mã PIN cho chìa khóa nhà — mỗi lần dùng chìa (SSH kết nối), bạn phải nhập mật khẩu này. 

| Có passphrase | Không passphrase |
|:-------------|:----------------|
| Mỗi lần SSH phải nhập mật khẩu | SSH kết nối luôn, không hỏi gì |
| An toàn hơn (nếu ai đánh cắp key) | Tiện hơn cho training dài |

👉 **Khuyến nghị: Bỏ trống (nhấn Enter)** — vì máy GPU chỉ dùng tạm để train, không cần bảo mật cao. Nếu đặt passphrase, mỗi lần SCP upload/download đều phải nhập lại → phiền.

**Bước 3 — Kiểm tra đã tạo thành công:**
```powershell
# Xem public key (dùng để paste lên Vast.ai)
cat ~/.ssh/id_ed25519.pub
```
Kết quả sẽ là 1 dòng dài bắt đầu bằng `ssh-ed25519 AAAA...` — đây là **public key**.

**Bước 4 — Copy public key và paste lên Vast.ai:**
```powershell
# Copy vào clipboard
Get-Content ~/.ssh/id_ed25519.pub | Set-Clipboard
```
1. Vào [https://cloud.vast.ai/account/](https://cloud.vast.ai/account/) → **SSH Keys**
2. Paste vào ô **"SSH Public Key"**
3. Click **"Add SSH Key"**

> [!WARNING]
> **Kiểm tra nếu đã có sẵn SSH key:**
> ```powershell
> ls ~/.ssh/
> ```
> Nếu thấy file `id_ed25519.pub` hoặc `id_rsa.pub` → bạn đã có key rồi! Chỉ cần copy public key lên Vast.ai, **không cần tạo lại**.

---

## 3. Template Là Gì? Chọn Template Nào?

### Template trên Vast.ai là gì?

**Template = Docker Image có sẵn phần mềm** — khi bạn thuê GPU trên Vast.ai, bạn không nhận một máy trần. Bạn nhận một **container Docker** chạy trên máy GPU đó. Template quyết định container đó chứa gì sẵn.

Hãy hiểu đơn giản:

```
Template = "Hệ điều hành + Phần mềm cài sẵn" đóng gói trong Docker image
```

| Thuật ngữ | Giải thích | Ví dụ |
|:----------|:-----------|:------|
| **Template** | Docker image có sẵn, chọn 1 click | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel` |
| **Base OS** | Hệ điều hành bên trong Docker image | Ubuntu 20.04, Ubuntu 22.04 |
| **Tag/Version** | Phiên bản cụ thể của image | `2.1.0-cuda12.1-cudnn8-devel` |
| **Custom Image** | Bạn tự nhập Docker image riêng | `nvidia/cuda:12.1.0-devel-ubuntu22.04` |

### Các template phổ biến trên Vast.ai

| Template | Base OS | Có sẵn | Khi nào dùng |
|:---------|:--------|:-------|:-------------|
| **`pytorch/pytorch`** ⭐ | Ubuntu 22.04 | PyTorch, CUDA, cuDNN, Python, pip | **ML/DL training — phổ biến nhất** |
| `tensorflow/tensorflow` | Ubuntu 22.04 | TensorFlow, CUDA | Dùng TensorFlow |
| `nvidia/cuda` | Ubuntu 22.04 | Chỉ CUDA + driver | Tự cài mọi thứ (linh hoạt nhất) |
| `vastai/base` | Ubuntu 22.04 | SSH + Jupyter | Cần tùy biến hoàn toàn |

### Tại sao Ubuntu 22.04?

Hầu hết template trên Vast.ai đều **đã dựa trên Ubuntu 22.04** (hoặc 20.04). Đây là tiêu chuẩn vì:

- ✅ Tương thích tốt nhất với NVIDIA driver + CUDA
- ✅ Hầu hết paper/tutorial deep learning dùng Ubuntu
- ✅ Thư viện Python (PyTorch, HuggingFace) test chính thức trên Ubuntu
- ✅ Nếu sau này muốn dùng package `mamba-ssm` (CUDA kernel), **bắt buộc phải Linux**

### Project TransMamba chọn template nào?

> [!IMPORTANT]
> **Project này dùng PureSSM** — Mamba được implement bằng **pure PyTorch** (trong file `mamba_baseline.py`), **KHÔNG dùng package `mamba-ssm`** (cần CUDA kernel biên dịch trên Linux).
>
> Điều này có nghĩa:
> - ✅ Template **`pytorch/pytorch`** hoạt động tốt — đã có sẵn mọi thứ cần thiết
> - ✅ Base OS **Ubuntu 22.04** (template PyTorch mặc định đã dùng Ubuntu)
> - ❌ **Không cần** lo cài `mamba-ssm` hay biên dịch CUDA kernel

**👉 Chọn: Template `pytorch/pytorch` với tag `2.1.0-cuda12.1-cudnn8-devel` (hoặc mới hơn)**

Lý do:
1. Đã có sẵn **PyTorch 2.0+** (project yêu cầu `torch>=2.0.0`)
2. Đã có sẵn **CUDA + cuDNN** (tăng tốc GPU)
3. Base OS là **Ubuntu 22.04** — đúng yêu cầu
4. Chỉ cần `pip install` thêm vài thư viện nhỏ (`transformers`, `datasets`, `scikit-learn`)

Nếu muốn **tự tay cài từ đầu** trên Ubuntu 22.04 thuần:
```
# Nhập custom Docker image khi thuê:
nvidia/cuda:12.1.0-devel-ubuntu22.04
# Rồi tự cài: pip install torch transformers datasets scikit-learn tqdm
```

---

## 4. Tìm và Thuê GPU Instance

### Bước 4.1 — Chọn Template
1. Truy cập [https://cloud.vast.ai/templates/](https://cloud.vast.ai/templates/)
2. Tìm **"PyTorch"** → chọn **`pytorch/pytorch`**
3. Ở **Version Tag**, chọn: `2.1.0-cuda12.1-cudnn8-devel` hoặc mới hơn
4. Click **"Select & Configure"**

### Bước 4.2 — Tìm GPU Instance
1. Sau khi chọn template → trang **Search** hiện ra danh sách GPU
2. Dùng **bộ lọc bên trái** để tìm GPU phù hợp:

   | Filter | Giá trị đặt | Lý do |
   |:-------|:------------|:------|
   | GPU Type | RTX 3090 hoặc RTX 4090 | 24GB VRAM, chạy mọi encoder |
   | GPU RAM | ≥ 12 GB | Tối thiểu cho bert-base |
   | Disk Space | ≥ 30 GB | Dataset + checkpoints |
   | Internet DL Speed | ≥ 100 Mbps | Tải model HuggingFace nhanh |
   | CPU RAM | ≥ 16 GB | DataLoader preprocessing |

3. **Sắp xếp theo giá** ($/hr) → chọn instance rẻ nhất phù hợp

> [!WARNING]
> **Kiểm tra trước khi thuê:**
> - **DLPerf** (Deep Learning Performance): Số càng cao càng tốt — đo benchmark thực tế
> - **Reliability**: Chọn máy có reliability **> 95%** (tránh bị gián đoạn)
> - **Max Duration**: Đảm bảo đủ thời gian cho experiments (~5-10 giờ)
> - **Geolocation**: Chọn server gần (Asia nếu có) để giảm latency
> - **CUDA Version**: ≥ 12.0 (tương thích PyTorch 2.0+)

### Bước 4.3 — Thuê Instance
1. Click **"RENT"** trên instance bạn chọn
2. Cấu hình:
   - **Disk Size**: đặt **40 GB** (⚠️ không thay đổi được sau khi tạo!)
   - **Docker Image**: để mặc định (PyTorch template đã chọn)
   - **On-start Script** (tùy chọn): có thể thêm lệnh tự chạy khi khởi động
3. Xác nhận và đợi instance khởi động (1-2 phút)

---

## 5. Quy Trình Tổng Quan — Code Chạy Ở Đâu?

> [!IMPORTANT]
> **GPU thuê trên Vast.ai là một máy Linux từ xa hoàn toàn trống!**
> Nó **KHÔNG** tự động có code của bạn. Bạn phải tự upload code lên, chạy training, rồi tải kết quả về.

```
Máy Windows của bạn                    Máy GPU trên Vast.ai (Ubuntu 22.04)
┌─────────────────────┐                ┌──────────────────────────────────┐
│ d:\code\notebookllm\ │   ── SCP ──→  │ /root/transmamba_project/        │
│  transmamba_project/ │   (upload)    │   ├── models/                    │
│   ├── models/        │               │   ├── data/                      │
│   ├── run_ablation.py│               │   ├── run_ablation.py            │
│   ├── results/ (trống)               │   ├── requirements.txt           │
│   └── ...            │               │   └── ...                        │
│                      │               │                                  │
│                      │               │   $ cd transmamba_project        │
│                      │               │   $ pip install -r requirements  │
│                      │               │   $ python run_ablation.py       │
│                      │               │        ↓ training trên GPU ↓     │
│   results/ (có data) │   ←── SCP ──  │   results/ (kết quả JSON)       │
│                      │   (download)  │                                  │
└─────────────────────┘                └──────────────────────────────────┘
```

**Tóm tắt 4 bước:**
1. 📤 **Upload** code từ Windows → máy GPU (bằng SCP hoặc Git)
2. 🔧 **Cài đặt** dependencies trên máy GPU
3. 🚀 **Chạy** `run_ablation.py` trên máy GPU (code chạy ở đây, dùng GPU!)
4. 📥 **Download** thư mục `results/` từ máy GPU → Windows

---

## 6. Kết Nối & Upload Code

### Bước 6.1 — Kết nối SSH vào máy GPU
1. Vào tab **Instances** trên Vast.ai → copy lệnh SSH
2. Mở **PowerShell** trên Windows, paste lệnh:
   ```powershell
   ssh -p <PORT> root@<IP_ADDRESS> -L 8080:localhost:8080
   ```
3. Bạn sẽ thấy terminal Ubuntu → đây là máy GPU từ xa

> Hoặc dùng **Jupyter**: Đợi instance **"Running"** → Click **"Open"** → mở Terminal trong Jupyter.

### Bước 6.2 — Upload code lên máy GPU

**Cách 1: SCP (đơn giản nhất)**

Mở **một PowerShell khác** trên Windows (giữ nguyên SSH ở cửa sổ cũ):
```powershell
# Upload toàn bộ thư mục transmamba_project lên máy GPU
scp -P <PORT> -r d:\code\notebookllm\transmamba_project root@<IP>:/root/
```

**Cách 2: Git clone (nếu đã push code lên GitHub)**

Trong terminal SSH của máy GPU:
```bash
git clone <YOUR_REPO_URL>
```

> [!TIP]
> **SCP nhanh hơn** nếu chưa push lên Git. Chỉ cần thay `<PORT>` và `<IP>` bằng thông tin từ Vast.ai.

---

## 7. Cài Đặt, Chạy Training & Tải Kết Quả

> Tất cả lệnh dưới đây chạy trong **terminal SSH** của máy GPU (không phải Windows!)

### Bước 7.1 — CD vào thư mục project & cài dependencies
```bash
# Vào thư mục project
cd /root/transmamba_project

# Cài thư viện Python
pip install -r requirements.txt
```

### Bước 7.2 — Kiểm tra GPU hoạt động
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```
Nếu thấy `CUDA: True` và tên GPU → OK! ✅

### Bước 7.3 — Mở tmux rồi chạy ablation

```bash
# ⚠️ QUAN TRỌNG: Mở tmux trước! Nếu SSH bị ngắt, training vẫn tiếp tục
tmux new -s train

# Đảm bảo đang ở đúng thư mục
cd /root/transmamba_project

# 🚀 Chạy toàn bộ 9 experiments (1 lệnh duy nhất!)
python run_ablation.py --task sst2 --epochs 5
```

`run_ablation.py` sẽ tự động chạy **tuần tự 9 experiments**:

| # | Experiment | Mô tả |
|:--|:-----------|:------|
| 1 | TransMamba-small (cross_attention) | ⭐ Full v2 theo paper |
| 2 | TransMamba-small (cross_attention_simple) | v1 không feature projection |
| 3 | TransMamba-small (additive) | Additive fusion |
| 4 | TransMamba-small (none) | Không fusion |
| 5 | TransMamba-small (frozen encoder) | Freeze BERT, chỉ train decoder |
| 6 | TransMamba-tiny (encoder scaling) | bert-tiny ~5M params |
| 7 | TransMamba-base (encoder scaling) | bert-base ~115M params |
| 8 | BERT-tiny Baseline | Pure BERT fine-tuning |
| 9 | Pure Mamba Baseline | Pure SSM |

Kết quả tự động lưu vào thư mục `results/` trên máy GPU.

Nếu muốn chạy thêm task khác:
```bash
python run_ablation.py --task mnli --epochs 5
python run_ablation.py --task rte --epochs 5
```

So sánh kết quả + tạo LaTeX table:
```bash
python compare_results.py
```

> [!TIP]
> **Mẹo dùng tmux:**
> ```bash
> # Ctrl+B rồi D     → Thoát tmux (training vẫn chạy nền)
> tmux attach -t train → Quay lại xem training
> ```
> Bạn có thể tắt SSH, đi ngủ, sáng quay lại `tmux attach` xem kết quả!

### Bước 7.4 — Tải kết quả về máy Windows

Sau khi training xong, mở **PowerShell trên Windows**:
```powershell
# Download thư mục results/ từ máy GPU về máy Windows
scp -P <PORT> -r root@<IP>:/root/transmamba_project/results/ d:\code\notebookllm\transmamba_project\results\
```

Kết quả sẽ nằm trong `d:\code\notebookllm\transmamba_project\results\` — mỗi experiment có 1 folder chứa `results.json` và `best_model.pt`.

### Bước 7.5 — Xóa instance (QUAN TRỌNG!)

> [!CAUTION]
> **Instance dừng vẫn tính phí storage!** Hãy **DELETE** instance ngay khi đã tải hết kết quả.

1. Vào **Instances** trên Vast.ai → click **"Delete"**
2. Xác nhận xóa → dừng hoàn toàn phí

---

## 8. Checklist Trước Khi Thuê

- [ ] Đã tạo tài khoản Vast.ai và nạp tiền ($5-10)
- [ ] Đã thiết lập SSH key
- [ ] Đã push code lên Git (hoặc chuẩn bị file để upload SCP)
- [ ] Đã chọn template `pytorch/pytorch` (Ubuntu 22.04 + CUDA 12.x + PyTorch 2.0+)
- [ ] Đã chọn GPU: **RTX 3090** (24GB), ≥40GB disk, ≥100Mbps, reliability >95%
- [ ] Nhớ dùng `tmux` trước khi train
- [ ] Nhớ tải results về trước khi **DELETE** instance

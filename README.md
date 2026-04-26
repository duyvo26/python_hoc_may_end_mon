# 🚀 Hệ thống Phân cụm Internet Chuyên sâu

> Ứng dụng GUI phân tích và phân cụm dữ liệu học thuật, xây dựng bằng **Gradio** — phục vụ mục đích nghiên cứu và báo cáo Thạc sĩ ngành Trí tuệ Nhân tạo / Học máy.

---

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Tính năng](#tính-năng)
3. [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
4. [Cài đặt](#cài-đặt)
5. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
6. [Tối ưu hiệu năng](#tối-ưu-hiệu-năng)
7. [Xuất báo cáo](#xuất-báo-cáo)

---

## Giới thiệu

Đây là một hệ thống phân tích phân cụm khép kín (End-to-End Pipeline) cung cấp đầy đủ các bước từ nạp dữ liệu thô đến xuất kết quả học thuật. Ứng dụng hỗ trợ so sánh song song hai thuật toán phân cụm phổ biến nhất: **K-Means** và **Hierarchical Clustering (Agglomerative)**.

---

## Tính năng

### 1. Phân tích Tương quan Dữ liệu
- Tự động tải và xem trước 5 dòng dữ liệu đầu từ tệp CSV.
- Vẽ **Ma trận tương quan Pearson** (Heatmap) với dải màu `RdBu_r` chuẩn học thuật, độ phân giải 300 DPI, font Times New Roman.

### 2. Tiền xử lý Dữ liệu mạnh mẽ
- Loại bỏ cột định danh không cần thiết (ID, Tên,...).
- Xử lý giá trị khuyết thiếu: `Mean` / `Median` / `Drop`.
- Chuẩn hoá: `StandardScaler` (μ=0, σ=1) hoặc `MinMaxScaler` ([0,1]).
- Tự động phát hiện và loại bỏ nhiễu bằng **Z-Score** (ngưỡng = 3σ).
- Tự động mã hoá các cột phân loại (Label Encoding).

### 3. Tự động tìm K tối ưu (Voting System)
- Vẽ biểu đồ 4 chỉ số đánh giá theo tiêu chuẩn học thuật: `(a) Elbow`, `(b) Silhouette`, `(c) Davies-Bouldin`, `(d) Calinski-Harabasz`.
- Cơ chế **biểu quyết đa số** (Majority Voting): mỗi chỉ số đề xuất K tốt nhất, K có nhiều phiếu nhất được chọn tự động.

### 4. Phân cụm & Trực quan hoá 3D
- Chạy song song **K-Means** và **Hierarchical Clustering**.
- Giảm chiều PCA xuống 3D, vẽ biểu đồ **Interactive 3D Scatter** (Plotly) — có thể xoay, zoom trực tiếp trên trình duyệt.
- Tâm cụm (Centroids) đánh dấu bằng dấu **✕ đỏ đậm**.
- Vẽ biểu đồ cây **Dendrogram** với tô màu từng nhánh cụm và đường ranh giới cắt ngưỡng (nét đứt đỏ).

### 5. So sánh hiệu năng mô hình
- Bảng so sánh 3 chỉ số: **Silhouette Score**, **Davies-Bouldin Index**, **Calinski-Harabasz Index** giữa K-Means và Hierarchical.

### 6. Đặc trưng cụm (Cluster Profiling)
- Bảng giá trị trung bình của từng đặc trưng theo từng cụm — dùng để đặt tên và giải thích ý nghĩa kinh tế/xã hội của từng nhóm.

### 7. Trợ lý AI Viết Báo cáo
- Tự động tổng hợp số liệu vào một **Prompt học thuật** chuẩn Thạc sĩ.
- Nhấn một nút để mở thẳng **ChatGPT** với Prompt đã điền sẵn toàn bộ dữ liệu.

### 8. Xuất Báo cáo ZIP (One-Click)
- Đóng gói tất cả kết quả vào **một file `.zip` duy nhất**:

| Tên tệp | Nội dung |
|---|---|
| `1_data_original.csv` | Dữ liệu gốc |
| `2_data_preprocessed.csv` | Dữ liệu sau tiền xử lý |
| `3_data_clustered.csv` | Dữ liệu đã gán nhãn cụm |
| `4_metrics.csv` | Bảng so sánh chỉ số hiệu năng |
| `5_profiling.csv` | Bảng đặc trưng trung bình từng cụm |
| `chart_1_correlation_heatmap.png` | Heatmap tương quan (300 DPI) |
| `chart_2_elbow_method.png` | Biểu đồ Elbow 4 chỉ số (300 DPI) |
| `chart_3_dendrogram.png` | Dendrogram phân cấp (300 DPI) |
| `chart_4_kmeans_3d.html` | Biểu đồ K-Means 3D (tương tác) |
| `chart_5_hierarchical_3d.html` | Biểu đồ Hierarchical 3D (tương tác) |

---

## Kiến trúc hệ thống

Dự án tuân theo mô hình **MVC (Model-View-Controller)**, tách biệt rõ ràng 3 tầng:

```
python_hoc_may_end_mon/
│
├── main.py              # VIEW — Giao diện Gradio, sự kiện nút bấm
├── app_controller.py    # CONTROLLER — Điều phối luồng dữ liệu, lưu trạng thái
├── model_manager.py     # MODEL — Thuật toán K-Means, Hierarchical, PCA, Elbow
├── data_processor.py    # MODEL — Nạp CSV, làm sạch, chuẩn hoá, Heatmap
└── requirements.txt     # Danh sách thư viện
```

---

## Cài đặt

### Yêu cầu
- Python **3.9+**
- Hệ điều hành: Windows / Linux / macOS

### Các bước

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\activate        # Windows (PowerShell)
# source .venv/bin/activate     # Linux / macOS

# 2. Cài đặt thư viện
pip install -r requirements.txt

# 3. (Tuỳ chọn) Cài psutil để theo dõi CPU/RAM trực tiếp trên giao diện
pip install psutil
```

---

## Hướng dẫn sử dụng

Khởi chạy ứng dụng:

```bash
python main.py
```

Mở trình duyệt tại `http://127.0.0.1:7860` và làm theo **5 bước** trên giao diện:

| Bước | Tab | Hành động |
|---|---|---|
| **Bước 0** | Tab 1 | Tải lên tệp `.csv`, xem Heatmap tương quan |
| **Bước 1** | Tab 2 | Chọn cột xoá, cấu hình tiền xử lý, bấm ⚙️ Chạy |
| **Bước 2** | Tab 3 | Bấm 🔍 Vẽ Elbow để hệ thống tự gợi ý K |
| **Bước 3** | Tab 3 | Điều chỉnh K nếu cần, bấm 🚀 Chạy mô hình so sánh |
| **Bước 4** | Tab 4 | Bấm 🧠 Khởi tạo Prompt → Mở ChatGPT viết báo cáo |
| **Bước 5** | Tab 4 | Bấm 💾 Tải Full Báo Cáo (.ZIP) |

---

## Tối ưu hiệu năng

Hệ thống tự động áp dụng các kỹ thuật tối ưu RAM theo quy mô dữ liệu:

| Ngưỡng dữ liệu | Kỹ thuật áp dụng |
|---|---|
| Mọi kích thước | Sử dụng `float32` thay `float64` — tiết kiệm **50% RAM** |
| > 10,000 dòng | `silhouette_score` dùng **Subsampling** (10,000 mẫu) |
| > 15,000 dòng | Hierarchical dùng **KNN Approximation** — tránh O(N²) |
| > 50,000 dòng | K-Means chuyển sang **MiniBatchKMeans** — nhanh 10x |
| Sau mỗi vòng | `del` biến trung gian + `gc.collect()` — dọn RAM tức thì |

---

## Tiêu chuẩn Biểu đồ Học thuật

Tất cả biểu đồ tĩnh xuất ra tuân theo tiêu chuẩn tạp chí khoa học:

- **Font:** Times New Roman / DejaVu Serif
- **DPI:** 300 (chuẩn IEEE / Springer / Elsevier)
- **Caption:** Đánh số `Hình 1`, `Hình 2`, `Hình 3`; nhãn con `(a)`, `(b)`, `(c)`, `(d)`
- **Style:** `sns.despine()` — chỉ giữ cạnh trái & dưới, lưới nền chấm nhỏ mờ
- **Màu sắc:** Bảng màu `colorblind` — thân thiện với người mù màu

---

*Dự án phục vụ mục đích nghiên cứu học phần **Trí tuệ Nhân tạo / Học máy** — Chương trình Thạc sĩ.*

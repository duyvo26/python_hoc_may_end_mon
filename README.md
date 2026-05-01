# Phân tích Phân cụm Đa thuật toán (K-Means & Hierarchical)

Hệ thống phân tích dữ liệu chuyên sâu sử dụng Machine Learning để tự động tìm kiếm số cụm tối ưu (Optimal K) và trực quan hóa kết quả theo chuẩn báo cáo khoa học.

## 📑 Mục tiêu Nghiên cứu
- Thực hiện tiền xử lý dữ liệu tự động (Xử lý nhiễu, chuẩn hóa).
- Đánh giá số cụm tối ưu thông qua cơ chế **Biểu quyết trọng số (Weighted Voting)** từ nhiều chỉ số đo lường.
- Trực quan hóa không gian đa chiều bằng PCA (2D & 3D).

## 🚀 Tính năng Chính
- **Phân tích K tối ưu:** Kết hợp 4 phương pháp:
  - Elbow Method (Inertia) với thuật toán Kneedle.
  - Silhouette Coefficient (Độ tách biệt cụm).
  - Davies-Bouldin Index (Độ tương tự nội cụm).
  - Calinski-Harabasz Index (Tỷ lệ phương sai).
- **Trực quan hóa chuẩn 300 DPI:** Sử dụng font Times New Roman, hỗ trợ xuất báo cáo.
- **Profiling Cụm:** Tự động tính toán giá trị trung bình các đặc trưng trên từng cụm để giải thích ý nghĩa kinh tế/kỹ thuật.

## 🛠 Cấu trúc Dự án
```text
├── server.py           # Flask Backend xử lý API và Task ngầm
├── model_manager.py    # Lõi xử lý ML & Logic Biểu quyết K
├── data_processor.py   # Tiền xử lý dữ liệu & EDA
├── styles.py           # Cấu hình hiển thị chuẩn khoa học
└── static/templates    # Giao diện người dùng
```

## 📊 Phương pháp Luận
Hệ thống sử dụng cơ chế đồng thuận để chọn $K$ tốt nhất:
1. Chạy lặp $N$ lần (Trials) để loại bỏ tính ngẫu nhiên của K-Means.
2. Mỗi lần lặp tính toán bộ metrics.
3. Áp dụng trọng số biểu quyết (mặc định):
   - **Silhouette:** Trọng số 2 (Ưu tiên tính tách biệt).
   - **Davies-Bouldin:** Trọng số 2 (Ưu tiên tính tinh gọn).
   - **Calinski-Harabasz & Elbow:** Trọng số 1.

## 💻 Cài đặt & Khởi chạy
1. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```
2. Chạy ứng dụng:
   ```bash
   python server.py
   ```
3. Truy cập: `http://localhost:5000`

## 📝 Quy chuẩn Báo cáo
Khi trích dẫn kết quả từ hệ thống này vào báo cáo, lưu ý:
- Sử dụng các tệp `.png` độ phân giải cao trong thư mục `uploads/session_id`.
- Trích dẫn các chỉ số đo lường cụ thể trong bảng `1_K_Metrics_Detail.csv`.

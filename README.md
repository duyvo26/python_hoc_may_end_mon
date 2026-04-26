# Hệ thống Phân cụm Máy học Chuyên sâu (Clustering GUI App)

Đây là một ứng dụng giao diện web (GUI) được xây dựng bằng **Gradio** dành cho mục đích nghiên cứu và báo cáo học thuật (chuyên ngành Thạc sĩ). Ứng dụng cung cấp một quy trình phân tích dữ liệu khép kín (Pipeline) từ việc nạp dữ liệu, tiền xử lý, tự động tìm số cụm tối ưu, cho đến chạy các mô hình phân cụm (K-Means & Hierarchical Clustering) và xuất báo cáo tự động bằng AI (ChatGPT).

## ✨ Tính năng chính

- **1. Nạp và phân tích dữ liệu tương quan:** Tải lên tệp CSV và xem trước cấu trúc dữ liệu. Tự động vẽ ma trận tương quan (Heatmap) với độ phân giải cao.
- **2. Tiền xử lý dữ liệu mạnh mẽ:**
  - Hỗ trợ loại bỏ các cột định danh (ID) không cần thiết.
  - Xử lý giá trị khuyết thiếu (Missing values) bằng Mean, Median hoặc Drop.
  - Chuẩn hoá dữ liệu tự động với `StandardScaler` hoặc `MinMaxScaler`.
  - Tự động phát hiện và loại bỏ nhiễu (Outliers) dựa trên thuật toán Z-Score.
- **3. Phân cụm và So sánh Mô hình:**
  - **Tự động tìm K:** Sử dụng phương pháp Elbow (Khuỷu tay) kết hợp với biểu quyết đa số (Voting System) từ 3 chỉ số đo lường khắt khe (Silhouette, Davies-Bouldin, Calinski-Harabasz) để gợi ý tự động số lượng cụm (K) tối ưu nhất.
  - Trực quan hoá không gian 3D tương tác (Interactive 3D PCA) cực kỳ trực quan với khả năng xoay, zoom, làm nổi bật tâm cụm (bằng thư viện Plotly).
  - Hiển thị song song biểu đồ phân cụm bằng K-Means và Hierarchical.
  - Tự động vẽ biểu đồ dạng cây (Dendrogram) cùng với đường ranh giới cắt ngưỡng.
- **4. Báo cáo Tự động bằng AI:**
  - Tự động thống kê các giá trị đặc trưng trung bình của từng cụm dữ liệu (Profiling).
  - Tích hợp tính năng tạo Prompt AI (ChatGPT): Tự động nạp kết quả đánh giá mô hình và bảng Profile vào Prompt để sinh ngay một đoạn đánh giá học thuật, phân tích chuyên sâu và đặt tên cho từng phân khúc khách hàng.
- **5. One-Click Export (Tải toàn bộ Báo cáo ZIP):**
  - Đóng gói toàn bộ quá trình biến đổi dữ liệu (`data_original.csv`, `data_preprocessed.csv`, `data_clustered.csv`).
  - Xuất các bảng đánh giá chỉ số hiệu năng và bảng cấu hình cụm (`metrics.csv`, `profiling.csv`).
  - Tải xuống tất cả các biểu đồ ở dạng hình ảnh `.png` (đã được cấu hình nền trắng chuẩn báo cáo học thuật) và biểu đồ 3D ở dạng `.html`.
  - Nén tất cả 10+ tệp tài liệu vào một tệp `.zip` duy nhất để thuận tiện cho việc nộp bài và làm minh chứng đính kèm.

## 🚀 Hướng dẫn cài đặt

1. Đảm bảo máy tính đã cài đặt Python 3.9+.
2. Tải/Clone toàn bộ mã nguồn của dự án về máy.
3. *(Khuyến nghị)* Khởi tạo một môi trường ảo (`.venv`):
   ```bash
   python -m venv .venv
   # Active môi trường trên Windows (PowerShell):
   .\.venv\Scripts\activate
   ```
4. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
   *(Các thư viện cốt lõi bao gồm: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `gradio`, `plotly`)*

## 🖥️ Hướng dẫn sử dụng

1. Khởi chạy ứng dụng:
   ```bash
   python main.py
   ```
2. Mở trình duyệt và truy cập vào địa chỉ mạng cục bộ được cung cấp ở Terminal (thường là `http://127.0.0.1:7860`).
3. Đi theo trình tự các bước đã được đánh số trên giao diện:
   - **Tab 1:** Tải lên tệp dữ liệu gốc dạng (`.csv`).
   - **Tab 2:** Lựa chọn các cột không mang ý nghĩa phân cụm và ấn **Bước 1: Chạy Tiền xử lý**.
   - **Tab 3:** Ấn **Bước 2: Vẽ biểu đồ Elbow & Đánh giá tự động K** sau đó ấn **Bước 3: Chạy mô hình so sánh**.
   - **Tab 4:** Ấn **Bước 4: Khởi tạo Prompt cho ChatGPT** để copy mẫu Prompt qua ChatGPT viết báo cáo. Cuối cùng bấm **Bước 5: Tải Full Báo Cáo** để lấy File nén ZIP tổng hợp chứa mọi kết quả gửi nộp bài.

## 📁 Cấu trúc mã nguồn

- `main.py`: Tập lệnh điều khiển chính quản lý Giao diện người dùng (GUI) qua Gradio framework, định tuyến và quản lý sự kiện.
- `model_manager.py`: Khối xử lý chứa thuật toán Machine Learning (K-Means, Agglomerative Clustering), thuật toán giảm chiều dữ liệu (PCA), logic tìm K tối ưu (Voting) và khởi tạo biểu đồ đa chiều.
- `data_processor.py`: Đảm nhận pipeline nạp đọc tệp dữ liệu, khử nhiễu, chuẩn hoá (Standardization/MinMax) và xây dựng cấu trúc tương quan (Heatmap).
- `requirements.txt`: Chứa danh sách các gói thư viện Python cấu hình môi trường.

---
*Dự án phục vụ mục đích nghiên cứu cho học phần Trí tuệ Nhân tạo / Học máy Hệ Thạc sĩ.*

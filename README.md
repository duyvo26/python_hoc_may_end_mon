# Hệ thống Phân tích Phân cụm Đa thuật toán (K-Means & Hierarchical)

Hệ thống phân tích dữ liệu chuyên sâu sử dụng Machine Learning để tự động tìm kiếm số lượng cụm tối ưu (Optimal K) và trực quan hóa kết quả theo chuẩn báo cáo khoa học. Tài liệu này cung cấp cái nhìn toàn diện, phân tích sâu về kiến trúc, luồng đi của dữ liệu (data flow) và nguyên lý hoạt động của các thuật toán được tích hợp trong mã nguồn, phục vụ trực tiếp cho việc tham khảo, bảo trì và viết báo cáo nghiên cứu.

---

## 1. Hệ sinh thái Thư viện (Dependencies)
Hệ thống được xây dựng trên ngôn ngữ Python, kết hợp các thư viện chuyên dụng để tạo thành một pipeline học máy hoàn chỉnh từ Front-end tới Back-end:

### 1.1. Core Framework & Xử lý API
- **Flask**: Framework Web đóng vai trò là xương sống của hệ thống API. Nó quản lý các route (đường dẫn), nhận dữ liệu (upload file dạng chunk), trả về kết quả (JSON/File) và đặc biệt là khởi tạo các `Threading` chạy ngầm để không làm đóng băng giao diện khi mô hình đang huấn luyện.

### 1.2. Xử lý & Phân tích Dữ liệu
- **Pandas & NumPy**: Hai thư viện cốt lõi dùng để lưu trữ và biến đổi dữ liệu dạng ma trận (DataFrame). Chúng xử lý toàn bộ các phép toán vector hóa, lọc nhiễu, điền giá trị khuyết thiếu (missing values) và tổng hợp bảng Profiling thống kê (tính giá trị trung bình (mean) của từng đặc trưng theo từng cụm).

### 1.3. Lõi Machine Learning (Scikit-Learn)
Sử dụng thư viện `scikit-learn` cho hầu hết các thuật toán học máy:
- **Tiền xử lý (Preprocessing)**: Sử dụng `StandardScaler` (Z-score normalization) và `MinMaxScaler` để đưa dữ liệu về cùng một thang đo. `LabelEncoder` được dùng để số hóa các biến phân loại (Categorical).
- **Thuật toán Phân cụm (Clustering)**:
  - `KMeans`: Phân cụm phân hoạch dựa trên khoảng cách, tìm các tâm cụm (centroids). Thuật toán này có tính ngẫu nhiên (chạy nhiều lần để hội tụ).
  - `AgglomerativeClustering`: Thuật toán phân cụm phân cấp (Hierarchical) theo phương pháp gom cụm từ dưới lên (Bottom-up). Mặc định sử dụng liên kết `ward` (tối thiểu hóa phương sai nội cụm).
- **Thuật toán Giảm chiều (Dimensionality Reduction)**:
  - `PCA` (Principal Component Analysis): Phân tích thành phần chính, giúp nén không gian dữ liệu $N$-chiều xuống còn 2 chiều (2D) hoặc 3 chiều (3D) nhằm mục đích trực quan hóa nhưng vẫn giữ được lượng thông tin (phương sai) lớn nhất có thể.
- **Đánh giá & Đo lường (Metrics)**:
  - `silhouette_score`: Đo lường độ tách biệt (cụm này cách cụm kia bao xa) và độ gắn kết nội cụm. Giá trị chạy từ -1 đến 1.
  - `davies_bouldin_score`: Đo lường tỷ lệ giữa sự phân tán nội cụm và khoảng cách giữa các tâm cụm. Giá trị càng nhỏ càng tốt.
  - `calinski_harabasz_score`: Tỷ lệ phương sai giữa các cụm so với phương sai nội cụm. Điểm càng cao cụm càng định hình rõ.

### 1.4. Thuật toán Hỗ trợ & Trực quan hóa
- **Kneed**: Sử dụng thuật toán `Kneedle` để quét đồ thị đường cong WCSS (Inertia của K-Means) và tự động tìm ra điểm "Khuỷu tay" (Elbow Point) một cách chuẩn xác bằng toán học (đạo hàm/độ cong), thay vì phải nhìn bằng mắt.
- **Matplotlib & Seaborn**: Khung vẽ biểu đồ tĩnh chất lượng cao. Hệ thống được cấu hình để xuất ra hình ảnh chuẩn 300 DPI, font Times New Roman, lưới mờ (grid) để đưa thẳng vào báo cáo học thuật (Heatmap tương quan, Scatter plot 2D, Line plot cho Metrics).
- **Plotly**: Khung vẽ biểu đồ tương tác trên nền Web. Được sử dụng riêng cho đồ thị PCA 3D, cho phép người dùng dùng chuột xoay, thu phóng và xem nhãn từng điểm dữ liệu.
- **Python-docx**: Tự động hóa việc chèn các đoạn văn bản (text), số liệu, bảng (dataframes) và hình ảnh PNG đã sinh ra vào một cấu trúc định dạng `.docx` chuẩn chỉnh.

---

## 2. Kiến trúc & Chức năng Từng Module
Hệ thống được thiết kế theo mô hình Modular để dễ dàng quản lý.

```text
├── server.py           # Điều phối API, Threading ngầm và Route
├── data_processor.py   # Chuyên biệt cho Pipeline làm sạch và EDA
├── model_manager.py    # Lõi toán học, thuật toán và xuất hình ảnh đồ thị
├── report_generator.py # Sinh báo cáo tự động Markdown/Word
├── styles.py           # Thiết lập quy chuẩn hiển thị đồ thị (Matplotlib config)
└── static/templates    # Giao diện Frontend (HTML, CSS, JS)
```

### Chi tiết chức năng từng lớp (Class):
1. **Lớp `DataProcessor` (`data_processor.py`)**:
   - `load_data`: Đọc file CSV, chuyển đổi an toàn các giá trị `NaN` thành `Null` để không làm crash JSON khi gửi về frontend.
   - `plot_correlation`: Trích xuất các cột định lượng (numeric), tính ma trận tương quan Pearson và vẽ thành Heatmap bằng Seaborn.
   - `preprocess_data`: Thực hiện chuỗi pipeline xử lý:
     - **Bước 1**: Bỏ các cột rác (người dùng chọn).
     - **Bước 2 (Outliers)**: Quét từng cột số, tính độ lệch chuẩn (Std) và giá trị trung bình. Chỉ giữ lại các bản ghi có Z-score < 3 (Nằm trong khoảng 99.7% phân phối chuẩn). Việc quét từng cột giúp tối ưu RAM thay vì tính ma trận khổng lồ.
     - **Bước 3 (Imputation)**: Điền khuyết (Mean/Median cho dữ liệu số, Mode cho dữ liệu phân loại).
     - **Bước 4 (Encoding & Scaling)**: Số hóa Label và gọi StandardScaler/MinMaxScaler để chuẩn hóa khoảng giá trị.

2. **Lớp `ModelManager` (`model_manager.py`)**:
   - `_detect_elbow_kneedle`: Nhận mảng giá trị WCSS (Inertia) theo $K$, áp dụng thuật toán tối ưu hóa đường cong (convex, decreasing) để trả về đúng vị trí $K$ tối ưu.
   - `analyze_k`: Đây là lõi đánh giá **Biểu quyết (Weighted Voting)**. 
     - Lặp $N$ lần thuật toán KMeans (với random_state khác nhau) từ $K=2$ đến $10$.
     - Tính 4 chỉ số: Silhouette, DB, CH, Elbow.
     - Áp dụng hệ số: Silhouette (x2), Davies-Bouldin (x2), Calinski-Harabasz (x1), Elbow (x1).
     - Thuật toán gom nhóm phiếu bầu (`Counter`) và chọn ra cấu hình có lượng phiếu cao nhất.
     - Sinh đồ thị (Line plot) hiển thị xu hướng của 4 chỉ số biến thiên theo K.
   - `run_clustering`: Quá trình fit model cuối cùng. 
     - Áp dụng KMeans và Agglomerative với số $K$ tốt nhất.
     - Chạy PCA 2 chiều để xuất ảnh tĩnh (Matplotlib).
     - Chạy PCA 3 chiều để xuất cấu trúc đồ thị JSON (Plotly).
     - Áp nhãn cụm (Labels) thu được ngược lại vào bộ dữ liệu thô ban đầu (profile_df) để tính trung bình của từng nhóm (Profiling).

3. **Lớp `ReportGenerator` (`report_generator.py`)**:
   - Thu thập ảnh từ ổ cứng, đọc thông số hình dáng mảng dữ liệu (Shape). Lắp ráp theo cấu trúc dàn bài chuẩn: Giới thiệu, Phân tích K, Trực quan hóa kết quả, Bảng đặc trưng, và Kết luận. Xuất thẳng ra hai định dạng `.md` và `.docx`.

---

## 3. Luồng Thực thi Dữ liệu (Chi tiết Step-by-Step)

Quy trình vòng đời của một file dữ liệu từ khi upload đến khi thành báo cáo:

**1. Tiếp nhận file (Chunking Upload)**
- Người dùng chọn file lớn, Frontend sử dụng Javascript băm file thành các gói nhỏ (chunks).
- API `/api/upload_chunk` nhận từng mảnh và ghi nối tiếp (`ab` mode) vào file `.part` tĩnh trên server.
- Khi nhận đủ mảnh, hệ thống đổi đuôi thành CSV, tạo UUID cho file để không trùng lặp, khởi tạo `DataProcessor` đọc 10 dòng đầu gửi về trình duyệt preview.

**2. Tiền xử lý (Preprocessing)**
- Frontend gửi thông số cấu hình về `/api/preprocess`. 
- `DataProcessor` áp dụng loại bỏ nhiễu Z-score, điền Missing Values, Scaling. Trả về cho frontend xem thông số Shape (số dòng/số cột) sau khi loại bỏ dòng lỗi. Dữ liệu sạch sẽ được lưu tạm trong RAM.

**3. Phân tích K tối ưu (Asynchronous Task)**
- Gọi `/api/analyze-k`. Do bước này tốn nhiều thời gian vòng lặp, `server.py` tạo một Thread (luồng chạy nền) riêng, cấp một `Task ID`. Frontend sẽ ping liên tục `/api/status/<task_id>` để theo dõi.
- Trong Thread, `analyze_k` chạy $N$ lần mô hình, tính toán Metrics và xuất ra các ảnh phân tích biểu quyết. Ảnh vẽ xong được lưu thẳng vào ổ cứng tại mục `uploads/<session_id>/` và encode Base64 gửi ngược lại client để render ra màn hình.
- File CSV chứa chi tiết các chỉ số cũng được xuất ra tại đây.

**4. Huấn luyện (Training & Projection)**
- Sau khi chốt được số lượng cụm (Ví dụ $K=3$), giao diện kích hoạt API `/api/train`.
- Lại một Thread ngầm mới chạy. `run_clustering` fit mô hình bằng K-Means và Hierarchical.
- Dữ liệu bị đưa vào không gian nén PCA. Các tọa độ sau khi nén (Component 1, 2, 3) được vẽ lên Matplotlib (cho ảnh tĩnh) và bọc vào dict Plotly (để hiển thị Web).
- Tính bảng Profiling: Lấy nhãn K-Means ráp vào file CSV gốc $\rightarrow$ Groupby theo Cụm $\rightarrow$ Lấy Mean từng cột.
- Kích hoạt tiến trình `ReportGenerator` sinh ngay file `.docx` báo cáo toàn diện.

**5. Batch Processing (Xử lý hàng loạt quy mô lớn)**
- Nếu gửi file ZIP chứa 10 file CSV qua API `/api/batch-process`.
- Server bung nén vào thư mục tạm. Một Thread khổng lồ sẽ lặp qua (Loop) toàn bộ 10 file CSV này.
- Tại mỗi file, hệ thống tự động chạy vòng đời: Khởi tạo dữ liệu $\rightarrow$ Xử lý $\rightarrow$ Tìm K $\rightarrow$ Fit PCA $\rightarrow$ Sinh `.docx`. Toàn bộ log console được đẩy realtime về Web.
- Sau khi xong 10 file, gom tất cả file báo cáo `.docx` vào một file ZIP mới và trả link tải về cho client.

---

## 4. Hướng dẫn Cài đặt & Khởi chạy

1. Cài đặt các thư viện phụ thuộc từ file cấu hình:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi động server (Ứng dụng Flask sẽ chạy ở chế độ Single Server, `use_reloader=False` nhằm duy trì tính ổn định của Threading):
   ```bash
   python server.py
   ```
3. Mở trình duyệt Web hiện đại (Chrome/Edge/Firefox) và truy cập: 
   `http://localhost:5000`

---

## 5. Quy chuẩn Trích dẫn & Viết Báo cáo Khoa học
Hệ thống này được thiết kế để tự động đáp ứng mọi tiêu chuẩn hình thức:
- Đồ thị `.png` trong thư mục trả về đều được khóa hệ số DPI ở mức 300 (chuẩn nét in ấn) và dùng font Times New Roman. Tuyệt đối dùng file tải về để chèn vào Word, không chụp ảnh màn hình (screenshot).
- Lập luận chọn K: Khi viết báo cáo phân tích, người viết cần trích xuất số liệu từ file `1_K_Metrics_Detail.csv` và ảnh Voting History để minh chứng cho cơ chế đồng thuận của 4 thuật toán, tăng độ thuyết phục (objective) hơn so với việc chỉ nhìn bằng mắt thường đồ thị Elbow.

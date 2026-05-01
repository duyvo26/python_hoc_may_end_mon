# Hệ thống Phân tích Phân cụm Đa thuật toán (K-Means & Hierarchical)

Hệ thống phân tích dữ liệu chuyên sâu sử dụng Machine Learning để tự động tìm kiếm số cụm tối ưu (Optimal K) và trực quan hóa kết quả theo chuẩn báo cáo khoa học. Tài liệu này mô tả chi tiết về mặt kỹ thuật và quy trình hoạt động của mã nguồn để phục vụ việc viết báo cáo và hiểu rõ luồng thực thi.

## 1. Các thư viện sử dụng chính (Dependencies)
Hệ thống được xây dựng trên nền tảng Python, sử dụng các thư viện mạnh mẽ trong phân tích dữ liệu và phát triển web:
- **Flask**: Web framework nhẹ được dùng để xây dựng các API (nhận file, trả kết quả), phục vụ giao diện người dùng và quản lý các tác vụ xử lý chạy nền (background threads).
- **Pandas & NumPy**: Cốt lõi xử lý dữ liệu. Dùng để đọc/ghi file CSV, làm sạch dữ liệu, xử lý ma trận và tính toán các đặc trưng thống kê (Profiling).
- **Scikit-learn (sklearn)**: Thư viện Machine Learning cốt lõi, chịu trách nhiệm:
  - Phân cụm: `KMeans`, `AgglomerativeClustering`.
  - Đánh giá cụm: `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`.
  - Tiền xử lý & Giảm chiều: `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `PCA`.
- **Kneed**: Cung cấp thuật toán Kneedle để tự động dò tìm "điểm khuỷu tay" (elbow point) trên đồ thị WCSS một cách toán học thay vì nhìn bằng mắt thường.
- **Matplotlib & Seaborn**: Xây dựng các đồ thị tĩnh (Heatmap, Scatter 2D, Line plots) chất lượng cao (300 DPI, font Times New Roman) đạt chuẩn xuất bản báo cáo khoa học.
- **Plotly**: Hỗ trợ vẽ và hiển thị đồ thị không gian 3 chiều (PCA 3D) có thể tương tác (xoay, zoom, hover) trực tiếp trên giao diện web.
- **Python-docx**: (Bên trong module ReportGenerator) Hỗ trợ tự động kết xuất toàn bộ dữ liệu, bảng biểu và đồ thị thành một báo cáo định dạng Word `.docx`.

## 2. Các chức năng chính của hệ thống
- **Tải lên và Tiền xử lý dữ liệu tự động**: 
  - Hỗ trợ tải lên file theo từng phần (chunking) giúp tránh lỗi với file kích thước lớn.
  - Cho phép người dùng cấu hình cách điền giá trị khuyết thiếu (Mean, Median, Drop), chuẩn hóa (Standard, MinMax) và tự động loại bỏ nhiễu (Outliers) dựa trên hệ số Z-score (< 3).
- **Đánh giá K tối ưu bằng Biểu quyết Trọng số (Weighted Voting)**:
  - Chạy lặp $N$ lần (Trials) thuật toán K-Means để loại bỏ rủi ro rơi vào cực tiểu địa phương do khởi tạo ngẫu nhiên.
  - Trong mỗi lần lặp, hệ thống tính toán 4 chỉ số (Silhouette, Davies-Bouldin, Calinski-Harabasz, Elbow).
  - Áp dụng hệ số bầu chọn (Silhouette & DB có trọng số cao hơn) để chốt số lượng cụm K ổn định nhất.
- **Huấn luyện mô hình và Trực quan hóa**:
  - Chạy song song K-Means và Hierarchical Clustering để có sự đối chiếu.
  - Sử dụng PCA để nén dữ liệu đa chiều xuống 2 chiều và 3 chiều, sau đó vẽ đồ thị minh họa các cụm.
  - Cung cấp bảng Profiling Cụm: tính toán giá trị trung bình gốc của từng thuộc tính trên từng cụm, giúp phân tích ý nghĩa nghiệp vụ của cụm đó.
- **Xử lý hàng loạt (Batch Processing)**:
  - Tải lên tệp `.zip` chứa hàng loạt file CSV.
  - Tự động lặp qua từng tệp: tiền xử lý $\rightarrow$ phân tích K $\rightarrow$ huấn luyện $\rightarrow$ sinh báo cáo Word.
  - Gói toàn bộ báo cáo vào một file ZIP và trả về cho người dùng.

## 3. Cấu trúc mã nguồn
```text
├── server.py           # Entry point của ứng dụng (Flask API, Threading, Routing)
├── model_manager.py    # Chứa logic học máy (Training, tính Metrics, Voting K)
├── data_processor.py   # Xử lý làm sạch, chuẩn hóa, mã hóa dữ liệu & vẽ Heatmap
├── report_generator.py # Logic kết hợp text, ảnh, bảng để sinh file Markdown/Word
├── styles.py           # Định nghĩa cấu hình style biểu đồ chuẩn khoa học
└── static/templates    # Nơi chứa HTML, JS, CSS cho giao diện Frontend
```

## 4. Quy trình chạy của hệ thống (Luồng thực thi)
1. **Khởi động**: Khi chạy `python server.py`, Flask server được bật ở chế độ không tự khởi động lại (`use_reloader=False`) để tránh xung đột với các luồng nền. Các cấu hình biểu đồ chuẩn khoa học được nạp sẵn.
2. **Nạp & Làm sạch dữ liệu**:
   - Giao diện Frontend gọi API `/api/upload_chunk` đẩy file lên.
   - `DataProcessor.load_data` đọc cấu trúc file. Khi có yêu cầu, `preprocess_data` sẽ loại bỏ nhiễu bằng Z-score tối ưu (chỉ quét từng cột số), điền khuyết, mã hóa biến phân loại và chuẩn hóa (Scale). Khung dữ liệu thô và dữ liệu đã xử lý được lưu trữ vào bộ nhớ phiên.
3. **Phân tích K (Bất đồng bộ)**:
   - Khi nhận lệnh, `server.py` mở một Thread nền để không chặn giao diện. Thread này gọi `ModelManager.analyze_k`.
   - Vòng lặp chạy từ K=2 đến 10 qua $N$ Trials. Mỗi K tính ra các chỉ số mảng (Silhouette, DB, CH, Elbow). Thực hiện logic "bầu cử" K.
   - Vẽ đồ thị xu hướng chỉ số, lưu ảnh PNG 300 DPI vào đĩa cứng và mã hóa Base64 gửi trả lại cho giao diện cập nhật thanh tiến trình.
4. **Huấn luyện và Trực quan (Bất đồng bộ)**:
   - Dựa trên K được chọn, API `/api/train` tạo Thread chạy `ModelManager.run_clustering`.
   - Mô hình tiến hành phân cụm, thực hiện nén PCA.
   - Kết quả xuất ra gồm biểu đồ Matplotlib 2D (lưu cứng), biểu đồ Plotly 3D (trả về dưới dạng JSON structure), bảng thống kê Metrics và Profiling cụm.
5. **Tổng hợp báo cáo**: Ngay sau khi quá trình huấn luyện hoàn tất, server sẽ tự động gọi module `ReportGenerator` để lắp ráp thông tin (số dòng dữ liệu, lịch sử bầu chọn K, các ảnh đồ thị, diễn giải) thành tệp `Full_Report.docx` và `.md`.

## 5. Cài đặt & Khởi chạy
1. Cài đặt các thư viện phụ thuộc:
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi động server ứng dụng:
   ```bash
   python server.py
   ```
3. Mở trình duyệt và truy cập: `http://localhost:5000`

## 6. Quy chuẩn Trích dẫn Báo cáo
Khi đưa kết quả từ hệ thống vào văn bản chính thức:
- Sử dụng trực tiếp hình ảnh trong các tệp `.png` độ phân giải cao tại thư mục tải về (không nên chụp màn hình).
- Trích dẫn các con số cụ thể trong tệp `1_K_Metrics_Detail.csv` và `3_Metrics.csv` để lập luận cho tính khách quan của việc chọn K.

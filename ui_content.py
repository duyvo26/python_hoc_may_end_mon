# ui_content.py - Chứa nội dung văn bản cho giao diện

HEADER_MARKDOWN = "# 🚀 Hệ thống Phân cụm Internet Chuyên sâu (Modular)"

TAB1_GUIDE = """
> 📌 **Hướng dẫn:** Tải lên tệp CSV chứa dữ liệu cần phân cụm. Hệ thống sẽ tự động:
> - Hiển thị 5 dòng dữ liệu đầu tiên để xem trước.
> - Vẽ **Ma trận tương quan** để giúp anh/chị nhận biết mối quan hệ tuyến tính giữa các đặc trưng.
> - Màu **đỏ** = tương quan dương mạnh | Màu **xanh** = tương quan âm | Màu **trắng** = không tương quan.
"""

TAB2_GUIDE = """
> 📌 **Hướng dẫn:** Xem trước dữ liệu gốc, sau đó cấu hình các bước tiền xử lý:
> - **Xoá cột:** Loại bỏ các cột định danh (ID, tên,...) không có ý nghĩa phân cụm.
> - **Xử lý Missing — Mean/Median:** Điền giá trị trung bình/trung vị vào ô trống | **Drop:** Xoá hàng có giá trị trống.
> - **StandardScaler:** Chuẩn hoá về phân phối chuẩn (μ=0, σ=1) — phù hợp K-Means.
> - **MinMaxScaler:** Co giãn về khoảng [0,1] — phù hợp khi dữ liệu không có phân phối chuẩn.
> - **Z-Score Outlier:** Tự động loại bỏ các điểm dữ liệu bất thường (Z > 3 sigma).
"""

TAB3_METRICS_INFO = """
**📌 Giải thích các chỉ số và biểu đồ:**
- **Elbow Method (WCSS):** Thể hiện tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm. Điểm 'khuỷu tay' (nơi độ dốc giảm đột ngột) thường là K tốt.
- **Silhouette Score:** Đo lường độ chặt chẽ bên trong cụm và độ tách biệt giữa các cụm. **(Càng cao càng tốt)**.
- **Davies-Bouldin Index:** Đo lường tỷ lệ giữa độ phân tán trong cụm và khoảng cách giữa các cụm. **(Càng thấp càng tốt)**.
- **Calinski-Harabasz Index:** Tỷ lệ giữa phương sai giữa các cụm và phương sai trong nội bộ cụm. **(Càng cao càng tốt)**.
"""

TAB3_MODEL_CONFIG = """
**⚙️ Cấu hình Mô hình:**
- **K (số cụm):** Hệ thống tự gợi ý K tối ưu ở bước trên. Anh có thể điều chỉnh thủ công nếu muốn.

**🔗 Phương pháp Linkage (Hierarchical):**
| Linkage | Mô tả | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **ward** | Tối thiểu phương sai trong cụm | Cụm đều, không bị mất cân bằng | Chỉ dùng Euclidean |
| **complete** | Khoảng cách lớn nhất giữa 2 cụm | Cụm gọn | Nhạy cảm với nhiễu |
| **average** | Khoảng cách trung bình | Cân bằng giữa ward & complete | Chậm hơn ward |
| **single** | Khoảng cách nhỏ nhất | Phát hiện cụm hình dạng phi tòa | Dễ bị hiệu ứng chuỗi |
| **weighted** | TB giượng các hợp nhất | Phù hợp cụm không đều | Áp dụng hạn chế |
| **centroid** | Khoảng cách tậm cụm | Trực quan | Có thể bj đảo ngược |
"""

TAB3_TRAIN_GUIDE = """
> 📌 **Đọc kết quả:** Biểu đồ PCA chiếu dữ liệu xuống không gian 2D hoặc 3D để trực quan hoá các cụm.
> Bạn có thể xoay, phóng to/thu nhỏ trên biểu đồ để quan sát rõ hơn.
> Tâm cụm được đánh dấu bằng dấu **✕** hoặc **Kim cương đỏ**. 
> **Lưu ý:** Thuật toán **BIRCH** được sử dụng cho lộ trình Hierarchical giúp hệ thống xử lý mượt mà hàng triệu bản ghi mà không gây tràn RAM.
> Đường nét đứt đỏ trên Dendrogram là ngưỡng cắt tương ứng với K đã chọn (vẽ trên mẫu đại diện).
"""

TAB4_GUIDE = """
> 📌 **Hướng dẫn Tab 4:**
> 1. Xem bảng **Đặc trưng cụm** để hiểu rõ giá trị trung bình của từng nhóm (phân tích ý nghĩa kinh doanh).
> 2. Bấm **Khởi tạo Prompt** để copy nội dung gửi cho ChatGPT viết đoạn phân tích học thuật tự động.
> 3. Bấm **Tải Full Báo Cáo** để tải về một file ZIP gồm toàn bộ CSV + hình ảnh biểu đồ chất lượng cao (300 DPI).
"""

TAB5_METHODOLOGY = """
## 🛠️ Quy trình Xử lý & Các Kỹ thuật áp dụng (Methodology)

Hệ thống được thiết kế theo tiêu chuẩn **Pipeline Khoa học Dữ liệu** chuyên nghiệp, tích hợp các kỹ thuật tối ưu hóa cho dữ liệu lớn.

### 1. Luồng xử lý chi tiết (Step-by-Step Workflow)

| Bước | Hoạt động | Dữ liệu sử dụng | Kỹ thuật áp dụng |
| :--- | :--- | :--- | :--- |
| **B1** | Tải & Khám phá | 100% Dữ liệu | `Pandas`, `Correlation Heatmap` |
| **B2** | Tiền xử lý | 100% Dữ liệu | `Imputer`, `Standard/MinMax Scaler`, `Z-Score` |
| **B3** | Tìm K tối ưu | **Lấy mẫu (Sampling)** | `Kneedle`, `Silhouette`, `Davies-Bouldin`, `CH Index` |
| **B4** | **Huấn luyện** | **100% Dữ liệu** | `MiniBatchKMeans`, `BIRCH` (Chuẩn Big Data) |
| **B5** | Trực quan hóa | **Lấy mẫu (Sampling)** | `PCA (Principal Component Analysis)` 2D/3D |
| **B6** | Đánh giá & Xuất | 100% Dữ liệu | `Profiling`, `ZIP Archiving`, `AI Report Assistant` |

---

### 2. 🔍 Giải thích về việc "Cắt dòng" (Sampling) vs "Dữ liệu đầy đủ"
Để đảm bảo hệ thống chạy mượt mà trên cả dữ liệu hàng triệu dòng, chúng tôi áp dụng chiến lược **Hybrid Data Processing**:

*   **Tại sao cần Lấy mẫu (Sampling)?** 
    *   Các thuật toán tìm K tối ưu và các chỉ số học thuật như *Silhouette* có độ phức tạp tính toán rất cao ($O(N^2)$). Nếu chạy trên 1 triệu dòng, máy tính sẽ bị treo.
    *   Theo **Định luật số lớn**, một mẫu ngẫu nhiên đủ lớn (ví dụ 10k-20k dòng) là đủ để phản ánh chính xác cấu trúc của toàn bộ tập dữ liệu.
*   **Cam kết về Độ chính xác:** 
    *   Việc huấn luyện (phân cụm) và gán nhãn cuối cùng được thực hiện trên **TOÀN BỘ (100%)** dữ liệu của bạn.
    *   Kết quả xuất file CSV cuối cùng chứa nhãn cụm cho mọi dòng dữ liệu bạn đã tải lên.

### 3. Các kỹ thuật then chốt cho Luận văn Thạc sĩ
| Kỹ thuật | Mô tả | Ưu điểm học thuật |
| :--- | :--- | :--- |
| **MiniBatchKMeans** | Biến thể của KMeans xử lý theo từng cụm dữ liệu nhỏ. | Tốc độ nhanh, tiết kiệm RAM mà không mất độ chính xác. |
| **BIRCH** | Thuật toán phân cụm phân cấp dùng cấu trúc cây CF-Tree. | Xử lý dữ liệu khổng lồ mà các thuật toán phân cấp khác không làm được. |
| **PCA** | Giảm chiều dữ liệu từ N chiều về 2D hoặc 3D. | Trực quan hóa sự tách biệt giữa các cụm trong không gian. |

> 💡 **Thông tin thêm:** Toàn bộ ngưỡng lấy mẫu (10k, 20k...) và trọng số biểu quyết được cấu hình minh bạch trong file `.env` của hệ thống.
"""

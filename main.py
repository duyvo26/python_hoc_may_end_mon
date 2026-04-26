import gradio as gr
import os
import tempfile
import pandas as pd
import urllib.parse
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from data_processor import DataProcessor
from model_manager import ModelManager

# Cấu hình thẩm mỹ chuẩn Báo cáo Khoa học (Academic Report)
plt.style.use('default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="ticks", rc={"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
sns.set_palette("colorblind")

from app_controller import AppController

def get_sys_info():
    if not PSUTIL_AVAILABLE:
        return "⚠️ Chưa cài psutil"
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    return f"🖥️ CPU: {cpu}% | 🧠 RAM: {ram}%"

controller = AppController()

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    with gr.Row():
        gr.Markdown("# 🚀 Hệ thống Phân cụm Internet Chuyên sâu (Modular)")
    
    with gr.Tab("1. Dữ liệu & Tương quan"):
        gr.Markdown("""
> 📌 **Hướng dẫn:** Tải lên tệp CSV chứa dữ liệu cần phân cụm. Hệ thống sẽ tự động:
> - Hiển thị 5 dòng dữ liệu đầu tiên để xem trước.
> - Vẽ **Ma trận tương quan** để giúp anh/chị nhận biết mối quan hệ tuyến tính giữa các đặc trưng.
> - Màu **đỏ** = tương quan dương mạnh | Màu **xanh** = tương quan âm | Màu **trắng** = không tương quan.
        """)
        with gr.Row():
            file_in = gr.File(label="Tải lên CSV", file_types=[".csv"])
            status_in = gr.Textbox(label="Trạng thái hệ thống")
        with gr.Row():
            preview_in = gr.HTML(label="Dữ liệu gốc")
            heatmap_out = gr.Plot(label="Ma trận tương quan (Heatmap)")
            
    with gr.Tab("2. Tiền xử lý & Outliers"):
        gr.Markdown("""
> 📌 **Hướng dẫn:** Xem trước dữ liệu gốc, sau đó cấu hình các bước tiền xử lý:
> - **Xoá cột:** Loại bỏ các cột định danh (ID, tên,...) không có ý nghĩa phân cụm.
> - **Xử lý Missing — Mean/Median:** Điền giá trị trung bình/trung vị vào ô trống | **Drop:** Xoá hàng có giá trị trống.
> - **StandardScaler:** Chuẩn hoá về phân phối chuẩn (μ=0, σ=1) — phù hợp K-Means.
> - **MinMaxScaler:** Co giãn về khoảng [0,1] — phù hợp khi dữ liệu không có phân phối chuẩn.
> - **Z-Score Outlier:** Tự động loại bỏ các điểm dữ liệu bất thường (Z > 3 sigma).
        """)
        gr.Markdown("### 🔍 Dữ liệu gốc (5 dòng đầu) để tham khảo khi chọn cột cần xoá")
        raw_data_preview_tab2 = gr.DataFrame(label="Dữ liệu gốc (Xem trước 5 dòng)")
        with gr.Row():
            drop_cols = gr.CheckboxGroup(label="Chọn cột cần XOÁ (ID, User_ID, v.v.)")
            with gr.Column():
                imp_method = gr.Radio(["Mean", "Median", "Drop"], label="Xử lý Missing", value="Mean")
                scl_method = gr.Radio(["StandardScaler", "MinMaxScaler"], label="Chuẩn hoá", value="StandardScaler")
                out_check = gr.Checkbox(label="Loại bỏ nhiễu (Z-Score > 3)", value=True)
        btn_pre = gr.Button("Bước 1: ⚙️ Chạy Tiền xử lý", variant="primary")
        status_pre = gr.Textbox(label="Kết quả")
        preview_pre = gr.DataFrame(label="Dữ liệu sau xử lý (Xem trước)")
        
    with gr.Tab("3. Tìm K & Huấn luyện"):
        sys_info = gr.Textbox(value=get_sys_info(), label="Tài nguyên Server (Cập nhật Live)", interactive=False, max_lines=1)
        with gr.Row():
            with gr.Column():
                n_trials_slider = gr.Slider(1, 10, 1, step=1, label="Số lần chạy thử (N trials) - Càng cao càng ổn định nhưng chạy lâu hơn")
                btn_elbow = gr.Button("Bước 2: 🔍 Vẽ biểu đồ Elbow & Đánh giá tự động K", variant="secondary")
                status_k = gr.Textbox(label="Kết quả gợi ý")
                plot_elbow = gr.Plot()
                k_details = gr.DataFrame(label="Bảng chi tiết biểu quyết tìm K tối ưu")
                gr.Markdown("""
**📌 Giải thích các chỉ số và biểu đồ:**
- **Elbow Method (WCSS):** Thể hiện tổng bình phương khoảng cách từ các điểm dữ liệu đến tâm cụm. Điểm 'khuỷu tay' (nơi độ dốc giảm đột ngột) thường là K tốt.
- **Silhouette Score:** Đo lường độ chặt chẽ bên trong cụm và độ tách biệt giữa các cụm. **(Càng cao càng tốt)**.
- **Davies-Bouldin Index:** Đo lường tỷ lệ giữa độ phân tán trong cụm và khoảng cách giữa các cụm. **(Càng thấp càng tốt)**.
- **Calinski-Harabasz Index:** Tỷ lệ giữa phương sai giữa các cụm và phương sai trong nội bộ cụm. **(Càng cao càng tốt)**.
                """)
            with gr.Column():
                gr.Markdown("""
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
                """)
                k_kmeans_slider = gr.Slider(2, 10, 3, step=1, label="🟢 K cho K-Means (tự điều chỉnh sau khi tìm K)")
                k_hier_slider = gr.Slider(2, 10, 3, step=1, label="🔴 K cho Hierarchical (tự điều chỉnh sau khi tìm K)")
                link_type = gr.Dropdown(
                    ["ward", "complete", "average", "single", "weighted", "centroid"],
                    value="ward",
                    label="🔗 Phương pháp Linkage (Hierarchical)"
                )
                btn_train = gr.Button("Bước 3: 🚀 Chạy mô hình so sánh", variant="primary")
        
        with gr.Row():
            plot_cluster_km = gr.Plot(label="K-Means (PCA)")
            plot_cluster_h = gr.Plot(label="Hierarchical (PCA)")
        with gr.Row():
            plot_dendro = gr.Plot(label="Biểu đồ Dendrogram")
        gr.Markdown("""
> 📌 **Đọc kết quả:** Biểu đồ 3D PCA chiếu dữ liệu xuống không gian 3 chiều để trực quan hoá các cụm.
> Tâm cụm được đánh dấu bằng dấu **✕ đỏ đậm**. Đường nét đứt đỏ trên Dendrogram là ngưỡng cắt tương ứng với K đã chọn.
        """)
        res_metrics = gr.DataFrame(label="Bảng so sánh hiệu năng")

    with gr.Tab("4. Đặc trưng & Xuất file"):
        gr.Markdown("""
> 📌 **Hướng dẫn Tab 4:**
> 1. Xem bảng **Đặc trưng cụm** để hiểu rõ giá trị trung bình của từng nhóm (phân tích ý nghĩa kinh doanh).
> 2. Bấm **Khởi tạo Prompt** để copy nội dung gửi cho ChatGPT viết đoạn phân tích học thuật tự động.
> 3. Bấm **Tải Full Báo Cáo** để tải về một file ZIP gồm toàn bộ CSV + hình ảnh biểu đồ chất lượng cao (300 DPI).
        """)
        gr.Markdown("### 📊 Đặc trưng trung bình của từng cụm (Profiling)")
        res_profile = gr.DataFrame(label="Đặc trưng cụm")
        
        gr.Markdown("---")
        gr.Markdown("### 🤖 Trợ lý AI Viết Báo Cáo")
        with gr.Row():
            btn_chatgpt = gr.Button("Bước 4: 🧠 Khởi tạo Prompt cho ChatGPT", variant="secondary")
        with gr.Row():
            chatgpt_prompt = gr.Textbox(label="Nội dung Prompt (Có thể copy tay)", lines=5)
            chatgpt_link = gr.HTML(label="Link mở nhanh")
            
        gr.Markdown("---")
        gr.Markdown("### 📥 Tải Xuống Toàn Bộ Dữ Liệu Báo Cáo")
        gr.Markdown("Bấm nút dưới đây để tải về một file nén (.zip) chứa tất cả: Dữ liệu gốc, Dữ liệu sau xử lý, Kết quả gán nhãn cụm, Bảng đánh giá, cùng với **Tất cả các hình ảnh biểu đồ**.")
        btn_exp = gr.Button("Bước 5: 💾 Tải Full Báo Cáo (.ZIP)", variant="primary")
        file_out = gr.File(label="File Tổng hợp Báo cáo")
            
    # Sự kiện
    file_in.change(controller.handle_load, inputs=[file_in], outputs=[preview_in, raw_data_preview_tab2, drop_cols, status_in, heatmap_out])
    btn_pre.click(controller.handle_preprocess, inputs=[drop_cols, imp_method, scl_method, out_check], outputs=[status_pre, preview_pre])
    btn_elbow.click(controller.handle_elbow, inputs=[n_trials_slider], outputs=[plot_elbow, k_details, status_k, k_kmeans_slider, k_hier_slider])
    btn_train.click(controller.handle_train, inputs=[k_kmeans_slider, k_hier_slider, link_type], outputs=[plot_cluster_km, plot_cluster_h, plot_dendro, res_metrics, res_profile])
    btn_chatgpt.click(controller.handle_chatgpt, inputs=[res_metrics, res_profile], outputs=[chatgpt_prompt, chatgpt_link])
    btn_exp.click(controller.handle_export_all, outputs=[file_out])
    
    if PSUTIL_AVAILABLE:
        timer = gr.Timer(2)
        timer.tick(get_sys_info, outputs=[sys_info])

if __name__ == "__main__":
    # demo.launch()
    demo.launch(share=True)
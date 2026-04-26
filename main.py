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

# Cấu hình thẩm mỹ chung
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("viridis")

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
        sys_info = gr.Textbox(value=get_sys_info(), label="Theo dõi tài nguyên", interactive=False, max_lines=1)
    
    with gr.Tab("1. Dữ liệu & Tương quan"):
        with gr.Row():
            file_in = gr.File(label="Tải lên CSV", file_types=[".csv"])
            status_in = gr.Textbox(label="Trạng thái hệ thống")
        with gr.Row():
            preview_in = gr.HTML(label="Dữ liệu gốc")
            heatmap_out = gr.Plot(label="Ma trận tương quan (Heatmap)")
            
    with gr.Tab("2. Tiền xử lý & Outliers"):
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
        with gr.Row():
            with gr.Column():
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
                k_num = gr.Slider(2, 10, 3, step=1, label="Chọn số cụm (K)")
                link_type = gr.Dropdown(["ward", "complete", "average", "single"], value="ward", label="Phương pháp Linkage")
                btn_train = gr.Button("Bước 3: 🚀 Chạy mô hình so sánh", variant="primary")
        
        with gr.Row():
            plot_cluster_km = gr.Plot(label="K-Means (PCA)")
            plot_cluster_h = gr.Plot(label="Hierarchical (PCA)")
        with gr.Row():
            plot_dendro = gr.Plot(label="Biểu đồ Dendrogram")
        res_metrics = gr.DataFrame(label="Bảng so sánh hiệu năng")

    with gr.Tab("4. Đặc trưng & Xuất file"):
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
    btn_elbow.click(controller.handle_elbow, outputs=[plot_elbow, k_details, status_k, k_num])
    btn_train.click(controller.handle_train, inputs=[k_num, link_type], outputs=[plot_cluster_km, plot_cluster_h, plot_dendro, res_metrics, res_profile])
    btn_chatgpt.click(controller.handle_chatgpt, inputs=[res_metrics, res_profile], outputs=[chatgpt_prompt, chatgpt_link])
    btn_exp.click(controller.handle_export_all, outputs=[file_out])
    
    if PSUTIL_AVAILABLE:
        timer = gr.Timer(2)
        timer.tick(get_sys_info, outputs=[sys_info])

if __name__ == "__main__":
    # demo.launch()
    demo.launch(share=True)
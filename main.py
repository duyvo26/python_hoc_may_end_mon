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

js_copy_rich = """
(text, html) => {
    if (!text || text.trim().length === 0) return;
    try {
        const typeHtml = "text/html";
        const typeText = "text/plain";
        const blobHtml = new Blob([html], { type: typeHtml });
        const blobText = new Blob([text], { type: typeText });
        const data = [new ClipboardItem({ [typeHtml]: blobHtml, [typeText]: blobText })];
        navigator.clipboard.write(data).then(() => {
            alert("📋 Đã copy bảng! Bạn có thể dán vào Word/Excel dưới định dạng bảng chuẩn.");
        });
    } catch (err) {
        navigator.clipboard.writeText(text).then(() => {
            alert("📋 Đã sao chép (Dạng văn bản).");
        });
    }
}
"""

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
            btn_copy_raw = gr.Button("📋 Copy bảng trên", variant="secondary", size="sm")
            copy_buffer_raw_text = gr.Textbox(visible=False)
            copy_buffer_raw_html = gr.Textbox(visible=False)
        with gr.Row():
            drop_cols = gr.CheckboxGroup(label="Chọn cột cần XOÁ (ID, User_ID, v.v.)")
            with gr.Column():
                imp_method = gr.Radio(["Mean", "Median", "Drop"], label="Xử lý Missing", value="Mean")
                scl_method = gr.Radio(["StandardScaler", "MinMaxScaler"], label="Chuẩn hoá", value="StandardScaler")
                out_check = gr.Checkbox(label="Loại bỏ nhiễu (Z-Score > 3)", value=True)
        btn_pre = gr.Button("Bước 1: ⚙️ Chạy Tiền xử lý", variant="primary")
        status_pre = gr.Textbox(label="Kết quả")
        preview_pre = gr.DataFrame(label="Dữ liệu sau xử lý (Xem trước)", elem_id="preprocess_results")
        with gr.Row():
            btn_copy_pre = gr.Button("📋 Copy bảng trên", variant="secondary", size="sm")
            copy_buffer_pre_text = gr.Textbox(visible=False)
            copy_buffer_pre_html = gr.Textbox(visible=False)
        
    with gr.Tab("3. Tìm K & Huấn luyện"):
        sys_info = gr.Textbox(value=get_sys_info(), label="Tài nguyên Server (Cập nhật Live)", interactive=False, max_lines=1)
        with gr.Row():
            with gr.Column():
                n_trials_slider = gr.Slider(1, 10, 1, step=1, label="Số lần chạy thử (N trials) - Càng cao càng ổn định nhưng chạy lâu hơn")
                btn_elbow = gr.Button("Bước 2: 🔍 Vẽ biểu đồ Elbow & Đánh giá tự động K", variant="secondary")
                status_k = gr.Textbox(label="Kết quả gợi ý")
                plot_elbow_km = gr.Plot(label="Phân tích K-Means (Elbow, Sil, DB, CH)", elem_id="elbow_results")
                plot_elbow_h = gr.Plot(label="Phân tích Hierarchical - BIRCH (Sil, DB, CH)")
                k_details = gr.DataFrame(label="Bảng chi tiết biểu quyết tìm K tối ưu")
                with gr.Row():
                    btn_copy_k = gr.Button("📋 Copy bảng trên", variant="secondary", size="sm")
                    copy_buffer_k_text = gr.Textbox(visible=False)
                    copy_buffer_k_html = gr.Textbox(visible=False)
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
                k_hier_slider = gr.Slider(2, 10, 3, step=1, label="🔴 K cho Hierarchical - BIRCH")
                link_type = gr.Dropdown(
                    ["ward", "complete", "average", "single", "weighted", "centroid"],
                    value="ward",
                    label="🔗 Phương pháp Linkage (Hierarchical)"
                )
                pca_dim_radio = gr.Radio(["2D", "3D"], value="3D", label="📐 Chiều không gian PCA (Visual)")
                btn_train = gr.Button("Bước 3: 🚀 Chạy mô hình so sánh", variant="primary")
        
        with gr.Row():
            plot_cluster_km = gr.Plot(label="K-Means (PCA)", elem_id="train_results")
            plot_cluster_h = gr.Plot(label="Hierarchical (PCA)")
        with gr.Row():
            plot_dendro = gr.Plot(label="Biểu đồ Dendrogram")
        gr.Markdown("""
> 📌 **Đọc kết quả:** Biểu đồ PCA chiếu dữ liệu xuống không gian 2D hoặc 3D để trực quan hoá các cụm.
> Bạn có thể xoay, phóng to/thu nhỏ trên biểu đồ để quan sát rõ hơn.
> Tâm cụm được đánh dấu bằng dấu **✕** hoặc **Kim cương đỏ**. 
> **Lưu ý:** Thuật toán **BIRCH** được sử dụng cho lộ trình Hierarchical giúp hệ thống xử lý mượt mà hàng triệu bản ghi mà không gây tràn RAM.
> Đường nét đứt đỏ trên Dendrogram là ngưỡng cắt tương ứng với K đã chọn (vẽ trên mẫu đại diện).
        """)
        res_metrics = gr.DataFrame(label="Bảng so sánh hiệu năng")
        with gr.Row():
            btn_copy_metrics = gr.Button("📋 Copy bảng trên", variant="secondary", size="sm")
            copy_buffer_metrics_text = gr.Textbox(visible=False)
            copy_buffer_metrics_html = gr.Textbox(visible=False)

    with gr.Tab("4. Đặc trưng & Xuất file"):
        gr.Markdown("""
> 📌 **Hướng dẫn Tab 4:**
> 1. Xem bảng **Đặc trưng cụm** để hiểu rõ giá trị trung bình của từng nhóm (phân tích ý nghĩa kinh doanh).
> 2. Bấm **Khởi tạo Prompt** để copy nội dung gửi cho ChatGPT viết đoạn phân tích học thuật tự động.
> 3. Bấm **Tải Full Báo Cáo** để tải về một file ZIP gồm toàn bộ CSV + hình ảnh biểu đồ chất lượng cao (300 DPI).
        """)
        gr.Markdown("### 📊 Đặc trưng trung bình của từng cụm (Profiling)")
        with gr.Row():
            with gr.Column():
                res_profile_km = gr.DataFrame(label="Đặc trưng cụm (K-Means)")
                with gr.Row():
                    btn_copy_profile_km = gr.Button("📋 Copy bảng K-Means", variant="secondary", size="sm")
                    copy_buffer_km_text = gr.Textbox(visible=False)
                    copy_buffer_km_html = gr.Textbox(visible=False)
            with gr.Column():
                res_profile_h = gr.DataFrame(label="Đặc trưng cụm (Hierarchical - BIRCH)")
                with gr.Row():
                    btn_copy_profile_h = gr.Button("📋 Copy bảng BIRCH", variant="secondary", size="sm")
                    copy_buffer_h_text = gr.Textbox(visible=False)
                    copy_buffer_h_html = gr.Textbox(visible=False)
        
        gr.Markdown("---")
        gr.Markdown("### 🤖 Trợ lý AI Viết Báo Cáo")
        with gr.Row():
            btn_chatgpt = gr.Button("Bước 4: 🧠 Khởi tạo Prompt cho ChatGPT", variant="secondary")
        with gr.Row():
            chatgpt_prompt = gr.Textbox(label="Nội dung Prompt (Có thể copy tay)", lines=5, elem_id="prompt_results")
            with gr.Column():
                btn_copy_prompt = gr.Button("📋 Copy Prompt", variant="secondary", size="sm")
                copy_buffer_prompt = gr.Textbox(visible=False)
            chatgpt_link = gr.HTML(label="Link mở nhanh")
            
        gr.Markdown("---")
        gr.Markdown("### 📥 Tải Xuống Toàn Bộ Dữ Liệu Báo Cáo")
        gr.Markdown("Bấm nút dưới đây để tải về một file nén (.zip) chứa tất cả: Dữ liệu gốc, Dữ liệu sau xử lý, Kết quả gán nhãn cụm, Bảng đánh giá, cùng với **Tất cả các hình ảnh biểu đồ**.")
        btn_exp = gr.Button("Bước 5: 💾 Tải Full Báo Cáo (.ZIP)", variant="primary")
        file_out = gr.File(label="File Tổng hợp Báo cáo")

    with gr.Tab("5. 📚 Quy trình & Kỹ thuật"):
        gr.Markdown("""
## 🛠️ Quy trình Xử lý & Các Kỹ thuật áp dụng (Methodology)

Hệ thống được thiết kế theo tiêu chuẩn **Pipeline Khoa học Dữ liệu** chuyên nghiệp, tích hợp các kỹ thuật tối ưu hóa cho dữ liệu lớn.

### 1. Luồng xử lý (Data Workflow)
```mermaid
graph LR
    A[Dữ liệu gốc] --> B[Tiền xử lý & Chuẩn hoá]
    B --> C[Phân tích K tối ưu]
    C --> D[Huấn luyện So sánh]
    D --> E[PCA Giảm chiều]
    E --> F[Profiling & Xuất báo cáo]
```

### 2. Các kỹ thuật & Thuật toán then chốt
| Giai đoạn | Kỹ thuật áp dụng | Mục đích |
| :--- | :--- | :--- |
| **Tiền xử lý** | `StandardScaler` / `MinMaxScaler` | Đưa dữ liệu về cùng thang đo, tránh thiên kiến cho các biến có giá trị lớn. |
| **Xử lý Nhiễu** | `Z-Score Outlier Detection` | Loại bỏ các điểm dữ liệu bất thường (Z > 3) để tăng độ ổn định của tâm cụm. |
| **Tìm K tối ưu** | `Kneedle Algorithm` | Tự động xác định điểm "Khuỷu tay" (Elbow) bằng toán học thay vì nhìn bằng mắt. |
| **Biểu quyết K** | `Weighted Voting System` | Kết hợp 4 chỉ số (Sil, DB, CH, Elbow) với trọng số tùy chỉnh để tìm K khách quan nhất. |
| **K-Means** | `MiniBatchKMeans` | Tự động kích hoạt khi dữ liệu > 25,000 dòng để tối ưu tốc độ và RAM. |
| **Hierarchical** | `BIRCH Algorithm` | Giải pháp thay thế cho Agglomerative truyền thống, giúp phân cụm phân cấp trên hàng triệu dòng dữ liệu. |
| **Giảm chiều** | `PCA (Principal Component Analysis)` | Chiếu dữ liệu đa chiều xuống không gian 2D/3D mà vẫn giữ được đặc trưng chính (variance). |

### 3. Cơ sở khoa học của các chỉ số Đánh giá
Hệ thống sử dụng các minh chứng khoa học từ các công trình nghiên cứu kinh điển:
*   **Silhouette Score:** (Rousseeuw, 1987) - Đo lường độ chặt chẽ và tách biệt.
*   **Davies-Bouldin Index:** (Davies & Bouldin, 1979) - Tỷ lệ phân tán và khoảng cách.
*   **Calinski-Harabasz:** (Caliński & Harabasz, 1974) - Tỷ lệ phương sai nội cụm và ngoại cụm.

> 💡 **Lưu ý:** Tất cả các ngưỡng lấy mẫu và trọng số biểu quyết đều có thể được tinh chỉnh trong file cấu hình hệ thống (`.env`).
        """)
            
    # JS Scroll function
    js_scroll = "(id) => { setTimeout(() => { const el = document.getElementById(id); if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'}); }, 100); }"

    # Sự kiện
    file_in.change(controller.handle_load, inputs=[file_in], outputs=[preview_in, raw_data_preview_tab2, drop_cols, status_in, heatmap_out])
    
    btn_pre.click(controller.handle_preprocess, inputs=[drop_cols, imp_method, scl_method, out_check], outputs=[status_pre, preview_pre, btn_pre]).then(fn=None, inputs=None, outputs=None, js=f"() => {{ {js_scroll.replace('(id) =>', '').strip('{} ')}('preprocess_results'); }}")
    
    btn_elbow.click(controller.handle_elbow, inputs=[n_trials_slider], outputs=[plot_elbow_km, plot_elbow_h, k_details, status_k, k_kmeans_slider, k_hier_slider, btn_elbow]).then(fn=None, inputs=None, outputs=None, js=f"() => {{ {js_scroll.replace('(id) =>', '').strip('{} ')}('elbow_results'); }}")
    
    btn_train.click(controller.handle_train, inputs=[k_kmeans_slider, k_hier_slider, link_type, pca_dim_radio], outputs=[plot_cluster_km, plot_cluster_h, plot_dendro, res_metrics, res_profile_km, res_profile_h, btn_train]).then(fn=None, inputs=None, outputs=None, js=f"() => {{ {js_scroll.replace('(id) =>', '').strip('{} ')}('train_results'); }}")
    
    btn_chatgpt.click(controller.handle_chatgpt, inputs=[res_metrics, res_profile_km, res_profile_h], outputs=[chatgpt_prompt, chatgpt_link, btn_chatgpt]).then(fn=None, inputs=None, outputs=None, js=f"() => {{ {js_scroll.replace('(id) =>', '').strip('{} ')}('prompt_results'); }}")
    
    btn_exp.click(controller.handle_export_all, outputs=[file_out, btn_exp])

    # Sự kiện Copy
    btn_copy_raw.click(controller.handle_copy_table, inputs=[raw_data_preview_tab2], outputs=[copy_buffer_raw_text, copy_buffer_raw_html, btn_copy_raw]).then(fn=None, inputs=[copy_buffer_raw_text, copy_buffer_raw_html], outputs=None, js=js_copy_rich)
    btn_copy_pre.click(controller.handle_copy_table, inputs=[preview_pre], outputs=[copy_buffer_pre_text, copy_buffer_pre_html, btn_copy_pre]).then(fn=None, inputs=[copy_buffer_pre_text, copy_buffer_pre_html], outputs=None, js=js_copy_rich)
    btn_copy_k.click(controller.handle_copy_table, inputs=[k_details], outputs=[copy_buffer_k_text, copy_buffer_k_html, btn_copy_k]).then(fn=None, inputs=[copy_buffer_k_text, copy_buffer_k_html], outputs=None, js=js_copy_rich)
    btn_copy_metrics.click(controller.handle_copy_table, inputs=[res_metrics], outputs=[copy_buffer_metrics_text, copy_buffer_metrics_html, btn_copy_metrics]).then(fn=None, inputs=[copy_buffer_metrics_text, copy_buffer_metrics_html], outputs=None, js=js_copy_rich)
    btn_copy_profile_km.click(controller.handle_copy_table, inputs=[res_profile_km], outputs=[copy_buffer_km_text, copy_buffer_km_html, btn_copy_profile_km]).then(fn=None, inputs=[copy_buffer_km_text, copy_buffer_km_html], outputs=None, js=js_copy_rich)
    btn_copy_profile_h.click(controller.handle_copy_table, inputs=[res_profile_h], outputs=[copy_buffer_h_text, copy_buffer_h_html, btn_copy_profile_h]).then(fn=None, inputs=[copy_buffer_h_text, copy_buffer_h_html], outputs=None, js=js_copy_rich)
    
    btn_copy_prompt.click(lambda x: (x, gr.update(value="✅ Đã Copy", interactive=True)), inputs=[chatgpt_prompt], outputs=[copy_buffer_prompt, btn_copy_prompt]).then(fn=None, inputs=[copy_buffer_prompt], outputs=None, js="(x) => { navigator.clipboard.writeText(x); alert('📋 Đã copy Prompt!'); }")
    
    if PSUTIL_AVAILABLE:
        timer = gr.Timer(2)
        timer.tick(get_sys_info, outputs=[sys_info])

if __name__ == "__main__":
    # demo.launch()
    demo.launch(share=True)
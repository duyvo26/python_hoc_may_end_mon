import gradio as gr
import pandas as pd

from app_controller import AppController
from styles import setup_styles, get_sys_info, JS_COPY_RICH, JS_SCROLL
import ui_content as content

setup_styles()
controller = AppController()

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(content.HEADER_MARKDOWN)
    
    with gr.Tab("1. Dữ liệu & Tương quan"):
        gr.Markdown(content.TAB1_GUIDE)
        with gr.Row():
            file_in = gr.File(label="Tải lên CSV", file_types=[".csv"])
            status_in = gr.Textbox(label="Trạng thái hệ thống")
        with gr.Row():
            preview_in = gr.HTML(label="Dữ liệu gốc")
            heatmap_out = gr.Plot(label="Ma trận tương quan (Heatmap)")
            
    with gr.Tab("2. Tiền xử lý & Outliers"):
        gr.Markdown(content.TAB2_GUIDE)
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
                gr.Markdown(content.TAB3_METRICS_INFO)
            with gr.Column():
                gr.Markdown(content.TAB3_MODEL_CONFIG)
                k_kmeans_slider = gr.Slider(2, 10, 3, step=1, label="🟢 K cho K-Means (tự điều chỉnh sau khi tìm K)")
                k_hier_slider = gr.Slider(2, 10, 3, step=1, label="🔴 K cho Hierarchical - BIRCH")
                link_type = gr.Dropdown(
                    ["ward", "complete", "average", "single", "weighted", "centroid"],
                    value="ward",
                    label="🔗 Phương pháp Linkage (Hierarchical)"
                )
                pca_dim_radio = gr.Radio(["2D", "3D"], value="3D", label="📐 Chiều không gian PCA (Visual)")
                btn_train = gr.Button("Bước 3: 🚀 Bắt đầu Huấn luyện (Chạy ngầm)", variant="primary")
                task_id_state = gr.State("")
                status_task = gr.Textbox(label="Trạng thái Huấn luyện", interactive=False)
        
        with gr.Row():
            plot_cluster_km = gr.Plot(label="K-Means (PCA)", elem_id="train_results")
            plot_cluster_h = gr.Plot(label="Hierarchical (PCA)")
        with gr.Row():
            plot_dendro = gr.Plot(label="Biểu đồ Dendrogram")
        gr.Markdown(content.TAB3_TRAIN_GUIDE)
        res_metrics = gr.DataFrame(label="Bảng so sánh hiệu năng")
        with gr.Row():
            btn_copy_metrics = gr.Button("📋 Copy bảng trên", variant="secondary", size="sm")
            copy_buffer_metrics_text = gr.Textbox(visible=False)
            copy_buffer_metrics_html = gr.Textbox(visible=False)
        
        # Timer 5 giây để check status (Sẽ định nghĩa sự kiện ở cuối file)
        timer_task = gr.Timer(5, active=False)

    with gr.Tab("4. Đặc trưng & Xuất file"):
        gr.Markdown(content.TAB4_GUIDE)
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
        gr.Markdown(content.TAB5_METHODOLOGY)
            
    # --- PHẦN ĐỊNH NGHĨA SỰ KIỆN (EVENTS) ---
    def get_scroll_js(element_id):
        return f"() => {{ const el = document.getElementById('{element_id}'); if (el) el.scrollIntoView({{behavior: 'smooth', block: 'center'}}); }}"

    # Timer check status tác vụ huấn luyện
    timer_task.tick(
        controller.check_task_status, 
        inputs=[task_id_state], 
        outputs=[timer_task, status_task, plot_cluster_km, plot_cluster_h, plot_dendro, res_metrics, res_profile_km, res_profile_h]
    )

    file_in.change(controller.handle_load, inputs=[file_in], outputs=[preview_in, raw_data_preview_tab2, drop_cols, status_in, heatmap_out])
    
    btn_pre.click(controller.handle_preprocess, inputs=[drop_cols, imp_method, scl_method, out_check], outputs=[status_pre, preview_pre, btn_pre]).then(fn=None, inputs=None, outputs=None, js=get_scroll_js('preprocess_results'))
    
    btn_elbow.click(controller.handle_elbow, inputs=[n_trials_slider], outputs=[plot_elbow_km, plot_elbow_h, k_details, status_k, k_kmeans_slider, k_hier_slider, btn_elbow]).then(fn=None, inputs=None, outputs=None, js=get_scroll_js('elbow_results'))
    
    btn_train.click(
        controller.handle_train_async, 
        inputs=[k_kmeans_slider, k_hier_slider, link_type, pca_dim_radio], 
        outputs=[status_task, task_id_state]
    ).then(
        lambda: gr.update(active=True), 
        outputs=[timer_task]
    )
    
    btn_chatgpt.click(controller.handle_chatgpt, inputs=[res_metrics, res_profile_km, res_profile_h], outputs=[chatgpt_prompt, chatgpt_link, btn_chatgpt]).then(fn=None, inputs=None, outputs=None, js=get_scroll_js('prompt_results'))
    
    btn_exp.click(controller.handle_export_all, outputs=[file_out, btn_exp])

    # Sự kiện Copy
    btn_copy_raw.click(controller.handle_copy_table, inputs=[raw_data_preview_tab2], outputs=[copy_buffer_raw_text, copy_buffer_raw_html, btn_copy_raw]).then(fn=None, inputs=[copy_buffer_raw_text, copy_buffer_raw_html], outputs=None, js=JS_COPY_RICH)
    btn_copy_pre.click(controller.handle_copy_table, inputs=[preview_pre], outputs=[copy_buffer_pre_text, copy_buffer_pre_html, btn_copy_pre]).then(fn=None, inputs=[copy_buffer_pre_text, copy_buffer_pre_html], outputs=None, js=JS_COPY_RICH)
    btn_copy_k.click(controller.handle_copy_table, inputs=[k_details], outputs=[copy_buffer_k_text, copy_buffer_k_html, btn_copy_k]).then(fn=None, inputs=[copy_buffer_k_text, copy_buffer_k_html], outputs=None, js=JS_COPY_RICH)
    btn_copy_metrics.click(controller.handle_copy_table, inputs=[res_metrics], outputs=[copy_buffer_metrics_text, copy_buffer_metrics_html, btn_copy_metrics]).then(fn=None, inputs=[copy_buffer_metrics_text, copy_buffer_metrics_html], outputs=None, js=JS_COPY_RICH)
    btn_copy_profile_km.click(controller.handle_copy_table, inputs=[res_profile_km], outputs=[copy_buffer_km_text, copy_buffer_km_html, btn_copy_profile_km]).then(fn=None, inputs=[copy_buffer_km_text, copy_buffer_km_html], outputs=None, js=JS_COPY_RICH)
    btn_copy_profile_h.click(controller.handle_copy_table, inputs=[res_profile_h], outputs=[copy_buffer_h_text, copy_buffer_h_html, btn_copy_profile_h]).then(fn=None, inputs=[copy_buffer_h_text, copy_buffer_h_html], outputs=None, js=JS_COPY_RICH)
    
    btn_copy_prompt.click(lambda x: (x, gr.update(value="✅ Đã Copy", interactive=True)), inputs=[chatgpt_prompt], outputs=[copy_buffer_prompt, btn_copy_prompt]).then(fn=None, inputs=[copy_buffer_prompt], outputs=None, js="(x) => { navigator.clipboard.writeText(x); alert('📋 Đã copy Prompt!'); }")
    
    timer_sys = gr.Timer(2)
    timer_sys.tick(get_sys_info, outputs=[sys_info])

if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft(primary_hue="blue"))
html], outputs=None, js=JS_COPY_RICH)
    
    btn_copy_prompt.click(lambda x: (x, gr.update(value="✅ Đã Copy", interactive=True)), inputs=[chatgpt_prompt], outputs=[copy_buffer_prompt, btn_copy_prompt]).then(fn=None, inputs=[copy_buffer_prompt], outputs=None, js="(x) => { navigator.clipboard.writeText(x); alert('📋 Đã copy Prompt!'); }")
    
    timer = gr.Timer(2)
    timer.tick(get_sys_info, outputs=[sys_info])

if __name__ == "__main__":
    demo.launch(share=True)
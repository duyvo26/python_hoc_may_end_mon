import gradio as gr
import os
import tempfile
import pandas as pd
import urllib.parse
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor
from model_manager import ModelManager

# Cấu hình thẩm mỹ chung
plt.style.use('default')
sns.set_theme(style="whitegrid")
sns.set_palette("viridis")

class AppController:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        self.fig_corr = None
        self.fig_elbow = None
        self.fig_km = None
        self.fig_h = None
        self.fig_dendro = None
        self.metrics = None
        self.profile = None

    def handle_load(self, file):
        if file is None:
            return None, pd.DataFrame(), gr.update(choices=[]), "⚠️ Vui lòng tải lên file CSV.", None
        try:
            preview, cols, fig_corr = self.data_processor.load_data(file.name)
            self.fig_corr = fig_corr
            head5 = self.data_processor.df.head(5)
            return preview, head5, gr.update(choices=cols, value=[]), f"✅ Đã tải: {len(self.data_processor.df)} dòng, {len(cols)} cột.", fig_corr
        except Exception as e:
            return None, pd.DataFrame(), gr.update(choices=[]), f"❌ Lỗi: {str(e)}", None

    def handle_preprocess(self, cols_to_drop, imputer_method, scaler_method, remove_outliers):
        try:
            processed_df, _ = self.data_processor.preprocess_data(cols_to_drop, imputer_method, scaler_method, remove_outliers)
            return f"✅ Tiền xử lý xong! Còn lại {len(processed_df)} dòng.", processed_df.head(10)
        except Exception as e:
            return f"❌ Lỗi tiền xử lý: {str(e)}", None

    def handle_elbow(self):
        if self.data_processor.processed_df is None:
            return None, pd.DataFrame(), "⚠️ Hãy thực hiện Tiền xử lý trước!", gr.update()
        fig = self.model_manager.get_elbow_plot(self.data_processor.processed_df)
        self.fig_elbow = fig
        final_k, detail_df = self.model_manager.find_optimal_k(self.data_processor.processed_df)
        msg = f"📊 Đã tính toán. Hệ thống tự động chọn K = {final_k} (Theo biểu quyết đa số)."
        return fig, detail_df, msg, gr.update(value=final_k)

    def handle_train(self, n_clusters, linkage_type):
        if self.data_processor.processed_df is None:
            err_df = pd.DataFrame({"Lỗi": ["⚠️ Hãy thực hiện Tiền xử lý trước."]})
            return None, None, None, err_df, err_df
        
        try:
            fig_km, fig_h, fig_dendro, metrics, profile_data, _ = self.model_manager.run_clustering(
                self.data_processor.processed_df, 
                self.data_processor.profile_base_df, 
                n_clusters, 
                linkage_type
            )
            self.fig_km = fig_km
            self.fig_h = fig_h
            self.fig_dendro = fig_dendro
            self.metrics = metrics
            self.profile = profile_data
            return fig_km, fig_h, fig_dendro, metrics, profile_data
        except Exception as e:
            err_df = pd.DataFrame({"Lỗi": [f"❌ Lỗi: {str(e)}"]})
            return None, None, None, err_df, err_df

    def handle_export_all(self):
        export_dir = os.path.join(tempfile.gettempdir(), "clustering_export")
        os.makedirs(export_dir, exist_ok=True)
        
        # Save CSVs
        if self.data_processor.df is not None:
            self.data_processor.df.to_csv(os.path.join(export_dir, "1_data_original.csv"), index=False, encoding='utf-8-sig')
        if self.data_processor.processed_df is not None:
            self.data_processor.processed_df.to_csv(os.path.join(export_dir, "2_data_preprocessed.csv"), index=False, encoding='utf-8-sig')
        if self.model_manager.final_labeled_df is not None:
            self.model_manager.final_labeled_df.to_csv(os.path.join(export_dir, "3_data_clustered.csv"), index=False, encoding='utf-8-sig')
        if self.metrics is not None:
            self.metrics.to_csv(os.path.join(export_dir, "4_metrics.csv"), index=False, encoding='utf-8-sig')
        if self.profile is not None:
            self.profile.to_csv(os.path.join(export_dir, "5_profiling.csv"), index=False, encoding='utf-8-sig')
            
        # Save Charts
        if self.fig_corr is not None:
            self.fig_corr.savefig(os.path.join(export_dir, "chart_1_correlation_heatmap.png"), bbox_inches='tight')
        if self.fig_elbow is not None:
            self.fig_elbow.savefig(os.path.join(export_dir, "chart_2_elbow_method.png"), bbox_inches='tight')
        if self.fig_dendro is not None:
            self.fig_dendro.savefig(os.path.join(export_dir, "chart_3_dendrogram.png"), bbox_inches='tight')
            
        if self.fig_km is not None:
            if hasattr(self.fig_km, 'write_html'):
                self.fig_km.write_html(os.path.join(export_dir, "chart_4_kmeans_3d.html"))
            else:
                self.fig_km.savefig(os.path.join(export_dir, "chart_4_kmeans.png"), bbox_inches='tight')
                
        if self.fig_h is not None:
            if hasattr(self.fig_h, 'write_html'):
                self.fig_h.write_html(os.path.join(export_dir, "chart_5_hierarchical_3d.html"))
            else:
                self.fig_h.savefig(os.path.join(export_dir, "chart_5_hierarchical.png"), bbox_inches='tight')
                
        zip_path = os.path.join(tempfile.gettempdir(), "Bao_Cao_Phan_Cum")
        shutil.make_archive(zip_path, 'zip', export_dir)
        return f"{zip_path}.zip"

    def handle_chatgpt(self, metrics, profile):
        if metrics is None or profile is None or metrics.empty or profile.empty:
            return "⚠️ Cần chạy mô hình so sánh trước để có số liệu.", ""
        try:
            metrics_str = metrics.to_string(index=False)
            profile_str = profile.to_string(index=False)
            prompt = (
                f"Đóng vai là một chuyên gia Data Science, hãy nhận xét các chỉ số phân cụm sau:\n\n"
                f"1. Hiệu năng mô hình:\n{metrics_str}\n\n"
                f"2. Đặc trưng của các cụm (K-Means):\n{profile_str}\n\n"
                f"Dựa vào đó, hãy viết một đoạn báo cáo học thuật khoảng 300-500 từ đánh giá hiệu năng của K-Means và Hierarchical, "
                f"đồng thời phân tích ý nghĩa và đặt tên cho từng cụm (dựa vào đặc trưng trung bình). Giọng văn học thuật, dùng cho báo cáo Thạc sĩ."
            )
            encoded_prompt = urllib.parse.quote(prompt)
            link = f'<a href="https://chatgpt.com/?prompt={encoded_prompt}" target="_blank" style="display:inline-block; padding:10px 15px; background-color:#10a37f; color:white; border-radius:5px; text-decoration:none; font-weight:bold; font-size:16px;">🚀 Mở ChatGPT và tự động dán Prompt này</a>'
            return prompt, link
        except Exception as e:
            return f"Lỗi: {e}", ""

controller = AppController()

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🚀 Hệ thống Phân cụm Internet Chuyên sâu (Modular)")
    
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

if __name__ == "__main__":
    # demo.launch()
    demo.launch(share=True)
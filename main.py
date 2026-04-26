import gradio as gr
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor
from model_manager import ModelManager

# Cấu hình thẩm mỹ chung
plt.style.use('dark_background')
sns.set_palette("viridis")

class AppController:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()

    def handle_load(self, file):
        if file is None:
            return None, pd.DataFrame(), gr.update(choices=[]), "⚠️ Vui lòng tải lên file CSV.", None
        try:
            preview, cols, fig_corr = self.data_processor.load_data(file.name)
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
        final_k, detail_df = self.model_manager.find_optimal_k(self.data_processor.processed_df)
        msg = f"📊 Đã tính toán. Hệ thống tự động chọn K = {final_k} (Theo biểu quyết đa số)."
        return fig, detail_df, msg, gr.update(value=final_k)

    def handle_train(self, n_clusters, linkage_type):
        if self.data_processor.processed_df is None:
            err_df = pd.DataFrame({"Lỗi": ["⚠️ Hãy thực hiện Tiền xử lý trước."]})
            return None, None, err_df, err_df
        
        try:
            fig, fig_dendro, metrics, profile_data, _ = self.model_manager.run_clustering(
                self.data_processor.processed_df, 
                self.data_processor.profile_base_df, 
                n_clusters, 
                linkage_type
            )
            return fig, fig_dendro, metrics, profile_data
        except Exception as e:
            err_df = pd.DataFrame({"Lỗi": [f"❌ Lỗi: {str(e)}"]})
            return None, None, err_df, err_df

    def handle_export(self):
        if self.model_manager.final_labeled_df is None:
            return None
        temp_file = os.path.join(tempfile.gettempdir(), "ket_qua_phan_cum.csv")
        self.model_manager.final_labeled_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
        return temp_file

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
        btn_pre = gr.Button("⚙️ Chạy Tiền xử lý", variant="primary")
        status_pre = gr.Textbox(label="Kết quả")
        preview_pre = gr.DataFrame(label="Dữ liệu sau xử lý (Xem trước)")
        
    with gr.Tab("3. Tìm K & Huấn luyện"):
        with gr.Row():
            with gr.Column():
                btn_elbow = gr.Button("🔍 Vẽ biểu đồ Elbow & Đánh giá tự động K")
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
                btn_train = gr.Button("🚀 Chạy mô hình so sánh", variant="primary")
        
        with gr.Row():
            plot_cluster = gr.Plot(label="Biểu đồ phân cụm (PCA)")
            plot_dendro = gr.Plot(label="Biểu đồ Dendrogram")
        res_metrics = gr.DataFrame(label="Bảng so sánh hiệu năng")

    with gr.Tab("4. Đặc trưng & Xuất file"):
        gr.Markdown("### 📊 Đặc trưng trung bình của từng cụm (Profiling)")
        res_profile = gr.DataFrame(label="Đặc trưng cụm")
        gr.Markdown("---")
        gr.Markdown("### 📥 Tải xuống dữ liệu đã gán nhãn")
        btn_exp = gr.Button("💾 Xuất kết quả CSV", variant="primary")
        file_out = gr.File(label="File kết quả")

    # Sự kiện
    file_in.change(controller.handle_load, inputs=[file_in], outputs=[preview_in, raw_data_preview_tab2, drop_cols, status_in, heatmap_out])
    btn_pre.click(controller.handle_preprocess, inputs=[drop_cols, imp_method, scl_method, out_check], outputs=[status_pre, preview_pre])
    btn_elbow.click(controller.handle_elbow, outputs=[plot_elbow, k_details, status_k, k_num])
    btn_train.click(controller.handle_train, inputs=[k_num, link_type], outputs=[plot_cluster, plot_dendro, res_metrics, res_profile])
    btn_exp.click(controller.handle_export, outputs=[file_out])

if __name__ == "__main__":
    demo.launch()
import os
import tempfile
import pandas as pd
import urllib.parse
import shutil
import gradio as gr
from data_processor import DataProcessor
from model_manager import ModelManager

class AppController:
    """
    Lớp điều khiển (Controller) dùng để kết nối giữa giao diện Gradio và phần xử lý logic (ModelManager, DataProcessor).
    Lưu trữ trạng thái các biểu đồ và dữ liệu để phục vụ chức năng xuất báo cáo.
    """
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model_manager = ModelManager()
        self.fig_corr = None
        self.fig_elbow_km = None
        self.fig_elbow_h = None
        self.fig_km = None
        self.fig_h = None
        self.fig_dendro = None
        self.metrics = None
        self.profile_km = None
        self.profile_h = None
        self.original_filename = "Bao_Cao_Phan_Cum"

    def handle_load(self, file):
        """Xử lý sự kiện khi người dùng tải tệp CSV lên."""
        if file is None:
            return None, pd.DataFrame(), gr.update(choices=[]), "⚠️ Vui lòng tải lên file CSV.", None
        try:
            self.original_filename = os.path.splitext(os.path.basename(file.name))[0]
            preview, cols, fig_corr = self.data_processor.load_data(file.name)
            self.fig_corr = fig_corr
            head5 = self.data_processor.df.head(5)
            return preview, head5, gr.update(choices=cols, value=[]), f"✅ Đã tải: {len(self.data_processor.df)} dòng, {len(cols)} cột.", fig_corr
        except Exception as e:
            return None, pd.DataFrame(), gr.update(choices=[]), f"❌ Lỗi: {str(e)}", None

    def handle_preprocess(self, cols_to_drop, imputer_method, scaler_method, remove_outliers):
        """Xử lý sự kiện tiền xử lý dữ liệu từ Tab 2."""
        try:
            processed_df, _ = self.data_processor.preprocess_data(cols_to_drop, imputer_method, scaler_method, remove_outliers)
            return f"✅ Tiền xử lý xong! Còn lại {len(processed_df)} dòng.", processed_df.head(10), gr.update(interactive=True)
        except Exception as e:
            return f"❌ Lỗi tiền xử lý: {str(e)}", None, gr.update()

    def handle_elbow(self, n_trials):
        """Vẽ biểu đồ và trả về 2 K tối ưu riêng biệt cho K-Means và Hierarchical."""
        if self.data_processor.processed_df is None:
            return None, None, pd.DataFrame(), "⚠️ Hãy thực hiện Tiền xử lý trước!", gr.update(), gr.update(), gr.update()
        fig_km, fig_h, detail_df, k_kmeans, k_hierarchical = self.model_manager.analyze_k(self.data_processor.processed_df, n_trials=int(n_trials))
        self.fig_elbow_km = fig_km
        self.fig_elbow_h = fig_h
        msg = (f"📊 Đã tính toán (TB qua {n_trials} lần). "
               f"K-Means → {k_kmeans} | Hierarchical → {k_hierarchical}")
        return fig_km, fig_h, detail_df, msg, gr.update(value=k_kmeans), gr.update(value=k_hierarchical), gr.update(interactive=True)

    def handle_train(self, k_kmeans, k_hierarchical, linkage_type, pca_dim):
        """Chạy K-Means với k_kmeans và Hierarchical với k_hierarchical riêng biệt."""
        if self.data_processor.processed_df is None:
            err_df = pd.DataFrame({"Lỗi": ["⚠️ Hãy thực hiện Tiền xử lý trước."]})
            return None, None, None, err_df, err_df, err_df, gr.update()
        
        try:
            fig_km, fig_h, fig_dendro, metrics, profile_km, profile_h, _ = self.model_manager.run_clustering(
                self.data_processor.processed_df, 
                self.data_processor.profile_base_df, 
                k_kmeans,
                k_hierarchical,
                linkage_type,
                pca_dim
            )
            self.fig_km = fig_km
            self.fig_h = fig_h
            self.fig_dendro = fig_dendro
            self.metrics = metrics
            self.profile_km = profile_km
            self.profile_h = profile_h
            return fig_km, fig_h, fig_dendro, metrics, profile_km, profile_h, gr.update(interactive=True)
        except Exception as e:
            err_df = pd.DataFrame({"Lỗi": [f"❌ Lỗi: {str(e)}"]})
            return None, None, None, err_df, err_df, err_df, gr.update()

    def handle_export_all(self):
        """Đóng gói toàn bộ file dữ liệu và hình ảnh biểu đồ vào 1 file ZIP duy nhất để tải về."""
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
        if self.profile_km is not None:
            self.profile_km.to_csv(os.path.join(export_dir, "5_profiling_kmeans.csv"), index=False, encoding='utf-8-sig')
        if self.profile_h is not None:
            self.profile_h.to_csv(os.path.join(export_dir, "6_profiling_hierarchical.csv"), index=False, encoding='utf-8-sig')
            
        # Save Charts
        if self.fig_corr is not None:
            self.fig_corr.savefig(os.path.join(export_dir, "chart_1_correlation_heatmap.png"), bbox_inches='tight')
        if self.fig_elbow_km is not None:
            self.fig_elbow_km.savefig(os.path.join(export_dir, "chart_2a_elbow_kmeans.png"), bbox_inches='tight')
        if self.fig_elbow_h is not None:
            self.fig_elbow_h.savefig(os.path.join(export_dir, "chart_2b_elbow_hierarchical.png"), bbox_inches='tight')
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
                
        zip_name = f"Bao_Cao_{self.original_filename}"
        zip_path = os.path.join(tempfile.gettempdir(), zip_name)
        shutil.make_archive(zip_path, 'zip', export_dir)
        return f"{zip_path}.zip", gr.update(interactive=True)

    def handle_chatgpt(self, metrics, profile_km, profile_h):
        """Tạo URL Prompt tự động điền sẵn dữ liệu để gửi cho ChatGPT viết báo cáo."""
        if metrics is None or profile_km is None or profile_h is None or metrics.empty:
            return "⚠️ Cần chạy mô hình so sánh trước để có số liệu.", "", gr.update()
        try:
            metrics_str = metrics.to_string(index=False)
            profile_km_str = profile_km.to_string(index=False)
            profile_h_str = profile_h.to_string(index=False)
            prompt = (
                f"Đóng vai là một chuyên gia Data Science, hãy nhận xét các chỉ số phân cụm sau:\n\n"
                f"1. Hiệu năng so sánh mô hình:\n{metrics_str}\n\n"
                f"2. Đặc trưng của các cụm (K-Means):\n{profile_km_str}\n\n"
                f"3. Đặc trưng của các cụm (Hierarchical):\n{profile_h_str}\n\n"
                f"Dựa vào đó, hãy viết một đoạn báo cáo học thuật khoảng 400-600 từ đánh giá hiệu năng so sánh giữa K-Means và Hierarchical, "
                f"đồng thời phân tích ý nghĩa và gợi ý đặt tên cho từng cụm của cả 2 mô hình. Giọng văn học thuật, dùng cho báo cáo nghiên cứu."
            )
            encoded_prompt = urllib.parse.quote(prompt)
            link = f'<a href="https://chatgpt.com/?prompt={encoded_prompt}" target="_blank" style="display:inline-block; padding:10px 15px; background-color:#10a37f; color:white; border-radius:5px; text-decoration:none; font-weight:bold; font-size:16px;">🚀 Mở ChatGPT và tự động dán Prompt này</a>'
            return prompt, link, gr.update(interactive=True)
        except Exception as e:
            return f"Lỗi: {e}", "", gr.update()

    def handle_copy_table(self, df):
        """Chuyển đổi DataFrame thành chuỗi văn bản và HTML để copy đa định dạng (Rich Copy)."""
        if df is None:
            return "", "", gr.update()
            
        # Xử lý trường hợp Gradio truyền vào dict (thường gặp ở bản Gradio mới)
        if isinstance(df, dict):
            try:
                df = pd.DataFrame(data=df.get('data', []), columns=df.get('headers', []))
            except:
                return str(df), str(df), gr.update(value="✅ Đã Copy", interactive=True)

        if not isinstance(df, pd.DataFrame) or df.empty:
            return "", "", gr.update()

        try:
            # 1. Tạo bản Plain Text (để dán vào Notepad)
            headers = [str(c) for c in df.columns]
            rows = [[str(v) for v in r] for r in df.values]
            col_widths = [len(h) for h in headers]
            for row in rows:
                for i, v in enumerate(row):
                    col_widths[i] = max(col_widths[i], len(v))
            
            lines = []
            lines.append(" | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)))
            lines.append("-+-".join("-" * col_widths[i] for i in range(len(headers))))
            for row in rows:
                lines.append(" | ".join(v.ljust(col_widths[i]) for i, v in enumerate(row)))
            text_table = "\n".join(lines)
            
            # 2. Tạo bản HTML (để dán vào Word/Excel ra định dạng bảng chuẩn)
            html_table = df.to_html(index=False, border=1)
            
            return text_table, html_table, gr.update(value="✅ Đã Copy", interactive=True)
        except Exception:
            return df.to_string(index=False), df.to_html(index=False, border=1), gr.update(value="✅ Đã Copy", interactive=True)

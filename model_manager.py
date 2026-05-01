import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from kneed import KneeLocator
from collections import Counter
import plotly.graph_objects as go
import gc

class ModelManager:
    """Quản lý toàn bộ quá trình phân cụm: phân tích K tối ưu và huấn luyện mô hình."""

    def __init__(self):
        self.results = {}

    def _detect_elbow_kneedle(self, wcss):
        """Phát hiện điểm khuỷu tay (Elbow) tối ưu từ danh sách WCSS bằng thuật toán Kneedle."""
        try:
            kn = KneeLocator(range(2, 2 + len(wcss)), wcss, curve='convex', direction='decreasing')
            return kn.elbow - 2 if kn.elbow else 0 # Mặc định chọn K=2 nếu không tìm thấy elbow
        except:
            return 0

    def analyze_k(self, processed_df, n_trials=1):
        """Phân tích và biểu quyết K tối ưu cho K-Means và Hierarchical Clustering."""
        X = processed_df.values.astype(np.float32)
        K_range = range(2, 11)
        km_final_votes = Counter()
        h_final_votes = Counter()
        voting_history = []
        
        # Trọng số biểu quyết chuẩn khoa học
        w_sil = int(os.getenv("WEIGHT_SIL", 2))
        w_ch = int(os.getenv("WEIGHT_CH", 1))
        w_db = int(os.getenv("WEIGHT_DB", 2))
        w_elbow = int(os.getenv("WEIGHT_ELBOW", 1))

        last_data = {}
        for t in range(n_trials):
            wcss, sil_km, db_km, ch_km = [], [], [], []
            sil_h, db_h, ch_h = [], [], []
            for k in K_range:
                # --- PHÂN CỤM BẰNG K-MEANS ---
                # Cách tiếp cận: Phân hoạch không gian thành k vùng bao quanh tâm cụm.
                km = KMeans(n_clusters=k, n_init='auto', random_state=42+t).fit(X)
                wcss.append(km.inertia_)
                # Tính chỉ số đánh giá dựa trên bộ nhãn (labels) của K-Means
                sil_km.append(silhouette_score(X, km.labels_))
                db_km.append(davies_bouldin_score(X, km.labels_))
                ch_km.append(calinski_harabasz_score(X, km.labels_))
                
                # --- PHÂN CỤM BẰNG HIERARCHICAL (AGGLOMERATIVE) ---
                # Cách tiếp cận: Gộp dần các điểm/cụm gần nhau nhất theo cấu trúc cây (Dendrogram).
                hcl = AgglomerativeClustering(n_clusters=k).fit(X)
                # Tính chỉ số đánh giá dựa trên bộ nhãn (labels) của Hierarchical
                # LƯU Ý: Dù cùng K, nhưng nhãn của KM và H khác nhau -> Điểm Silhouette sẽ khác nhau.
                sil_h.append(silhouette_score(X, hcl.labels_))
                db_h.append(davies_bouldin_score(X, hcl.labels_))
                ch_h.append(calinski_harabasz_score(X, hcl.labels_))

            idx_elbow = self._detect_elbow_kneedle(wcss)
            b_elbow = K_range[idx_elbow]
            
            b_sil_km, b_db_km, b_ch_km = K_range[np.argmax(sil_km)], K_range[np.argmin(db_km)], K_range[np.argmax(ch_km)]
            b_sil_h, b_db_h, b_ch_h = K_range[np.argmax(sil_h)], K_range[np.argmin(db_h)], K_range[np.argmax(ch_h)]

            # Biểu quyết có trọng số
            km_votes = [b_sil_km]*w_sil + [b_db_km]*w_db + [b_ch_km]*w_ch + [b_elbow]*w_elbow
            h_votes = [b_sil_h]*w_sil + [b_db_h]*w_db + [b_ch_h]*w_ch
            
            km_trial_k = Counter(km_votes).most_common(1)[0][0]
            h_trial_k = Counter(h_votes).most_common(1)[0][0]
            
            km_final_votes[km_trial_k] += 1
            h_final_votes[h_trial_k] += 1
            voting_history.append({"Lần": t+1, "K-Means": km_trial_k, "Hierarchical": h_trial_k})
            last_data = {"wcss": wcss, "sil_km": sil_km, "db_km": db_km, "ch_km": ch_km, "sil_h": sil_h, "db_h": db_h, "ch_h": ch_h}

        final_k_km = km_final_votes.most_common(1)[0][0]
        final_k_h = h_final_votes.most_common(1)[0][0]

        fig_km, axes_km = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
        axes_km = axes_km.flatten()
        metrics_plot = [
            (last_data['wcss'], 'Elbow Method (Inertia)'),
            (last_data['sil_km'], 'Silhouette Score'),
            (last_data['db_km'], 'Davies-Bouldin Index'),
            (last_data['ch_km'], 'Calinski-Harabasz Index')
        ]
        for i, (data, title) in enumerate(metrics_plot):
            axes_km[i].plot(K_range, data, 'o-', linewidth=2)
            axes_km[i].set_title(title)
            axes_km[i].axvline(final_k_km, color='r', linestyle='--', label=f'Best K={final_k_km}')
            axes_km[i].grid(True, alpha=0.3)
        plt.tight_layout()

        fig_h, axes_h = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)
        h_metrics_plot = [
            (last_data['sil_h'], 'Silhouette Score'),
            (last_data['db_h'], 'Davies-Bouldin Index'),
            (last_data['ch_h'], 'Calinski-Harabasz Index')
        ]
        for i, (data, title) in enumerate(h_metrics_plot):
            axes_h[i].plot(K_range, data, 's-', color='green', linewidth=2)
            axes_h[i].set_title(title)
            axes_h[i].axvline(final_k_h, color='r', linestyle='--', label=f'Best K={final_k_h}')
            axes_h[i].grid(True, alpha=0.3)
        plt.tight_layout()

        detail_df = pd.DataFrame({
            "Chỉ số (Lần cuối)": ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Elbow"],
            "K tốt nhất KM": [b_sil_km, b_db_km, b_ch_km, b_elbow],
            "K tốt nhất H": [b_sil_h, b_db_h, b_ch_h, "—"]
        })
        msg = f"Kết quả đồng thuận sau {n_trials} lần thử: KM={final_k_km}, H={final_k_h}"
        return fig_km, fig_h, detail_df, msg, final_k_km, final_k_h, voting_history

    def run_clustering(self, df, profile_df, k_km, k_h, linkage='ward'):
        """Huấn luyện K-Means và Hierarchical Clustering, tạo trực quan PCA 2D/3D và Dendrogram.

        Args:
            df (pd.DataFrame): Dữ liệu đã chuẩn hóa để phân cụm.
            profile_df (pd.DataFrame): Dữ liệu gốc (chưa chuẩn hóa) để tạo bảng profiling.
            k_km (int): Số cụm cho K-Means.
            k_h (int): Số cụm cho Hierarchical Clustering.
            linkage (str): Phương pháp liên kết cho Hierarchical ('ward', 'complete', ...).
        Returns:
            dict: Chứa các hình ảnh PCA (2D/3D), Dendrogram, Metrics và Profile DataFrames.
        """
        X = df.values.astype(np.float32)
        km = KMeans(n_clusters=k_km, n_init='auto', random_state=42).fit(X)
        hcl = AgglomerativeClustering(n_clusters=k_h, linkage=linkage).fit(X)
        
        # PCA 2D (Static Matplotlib) - Tách riêng 2 hình
        pca2 = PCA(n_components=2)
        X_pca2 = pca2.fit_transform(X)
        
        def create_pca2d_fig(labels, title):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            sc = ax.scatter(X_pca2[:,0], X_pca2[:,1], c=labels, cmap='viridis', s=30, alpha=0.7)
            ax.set_title(title)
            ax.grid(True, alpha=0.2)
            plt.colorbar(sc, ax=ax)
            fig.tight_layout()
            return fig

        fig_pca2_km = create_pca2d_fig(km.labels_, f"K-Means 2D (K={k_km})")
        fig_pca2_h = create_pca2d_fig(hcl.labels_, f"Hierarchical 2D (K={k_h})")

        # PCA 3D (Interactive Plotly)
        pca3 = PCA(n_components=3)
        X_pca3 = pca3.fit_transform(X)
        def create_3d(labels, title):
            fig = go.Figure(data=[go.Scatter3d(
                x=X_pca3[:,0].tolist(), y=X_pca3[:,1].tolist(), z=X_pca3[:,2].tolist(),
                mode='markers', marker=dict(size=4, color=labels.tolist(), colorscale='Viridis', opacity=0.8)
            )])
            fig.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=30))
            return fig
        
        fig_pca3_km = create_3d(km.labels_, f"K-Means 3D (K={k_km})")
        fig_pca3_h = create_3d(hcl.labels_, f"Hierarchical 3D (K={k_h})")

        # Metrics & Profiling
        metrics = pd.DataFrame({
            "Mô hình": ["K-Means", "Hierarchical"],
            "Silhouette": [silhouette_score(X, km.labels_), silhouette_score(X, hcl.labels_)],
            "Davies-Bouldin": [davies_bouldin_score(X, km.labels_), davies_bouldin_score(X, hcl.labels_)],
            "Calinski-Harabasz": [calinski_harabasz_score(X, km.labels_), calinski_harabasz_score(X, hcl.labels_)]
        })

        def get_profile(labels):
            temp_df = profile_df.copy()
            if 'Cluster' in temp_df.columns: temp_df = temp_df.drop(columns=['Cluster'])
            temp_df['Cluster'] = labels
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Cluster' in numeric_cols: numeric_cols.remove('Cluster')
            return temp_df.groupby('Cluster')[numeric_cols].mean().round(2).reset_index()

        return {
            "pca2d_km": fig_pca2_km,
            "pca2d_h": fig_pca2_h,
            "pca3d_km": fig_pca3_km,
            "pca3d_h": fig_pca3_h,
            "metrics": metrics,
            "profile_km": get_profile(km.labels_),
            "profile_h": get_profile(hcl.labels_)
        }

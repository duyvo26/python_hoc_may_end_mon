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
        """Phát hiện điểm khuỷu tay (Elbow) tối ưu từ danh sách WCSS bằng thuật toán Kneedle.
        
        Args:
            wcss (list): Danh sách giá trị WCSS tương ứng với mỗi K.
        Returns:
            int: Chỉ số (index) của K tối ưu trong danh sách.
        """
        try:
            kn = KneeLocator(range(2, 2 + len(wcss)), wcss, curve='convex', direction='decreasing')
            return kn.elbow - 2 if kn.elbow else np.argmin(wcss)
        except:
            return np.argmin(wcss)

    def analyze_k(self, processed_df, n_trials=1):
        """Phân tích và biểu quyết K tối ưu cho K-Means và Hierarchical Clustering.

        Chạy n_trials lần, mỗi lần tính Elbow/Silhouette/DB/CH rồi biểu quyết
        có trọng số để chọn K tốt nhất cho từng thuật toán.

        Args:
            processed_df (pd.DataFrame): Dữ liệu đã được chuẩn hóa.
            n_trials (int): Số lần thử nghiệm để lấy kết quả biểu quyết ổn định hơn.
        Returns:
            tuple: (fig_km, fig_h, detail_df, msg, final_k_km, final_k_h, voting_history)
        """
        X = processed_df.values.astype(np.float32)
        K_range = range(2, 11)
        km_final_votes = Counter()
        h_final_votes = Counter()
        voting_history = []
        
        w_sil_km = int(os.getenv("WEIGHT_SIL_KM", 2))
        w_ch_km = int(os.getenv("WEIGHT_CH_KM", 2))
        w_elbow_km = int(os.getenv("WEIGHT_ELBOW_KM", 1))
        w_db_h = int(os.getenv("WEIGHT_DB_H", 2))
        w_sil_h = int(os.getenv("WEIGHT_SIL_H", 1))
        w_ch_h = int(os.getenv("WEIGHT_CH_H", 1))

        last_data = {}
        for t in range(n_trials):
            wcss, sil_km, db_km, ch_km = [], [], [], []
            sil_h, db_h, ch_h = [], [], []
            for k in K_range:
                km = KMeans(n_clusters=k, n_init='auto', random_state=42+t).fit(X)
                wcss.append(km.inertia_); sil_km.append(silhouette_score(X, km.labels_))
                db_km.append(davies_bouldin_score(X, km.labels_)); ch_km.append(calinski_harabasz_score(X, km.labels_))
                hcl = AgglomerativeClustering(n_clusters=k).fit(X)
                sil_h.append(silhouette_score(X, hcl.labels_)); db_h.append(davies_bouldin_score(X, hcl.labels_))
                ch_h.append(calinski_harabasz_score(X, hcl.labels_))

            b_elbow = K_range[self._detect_elbow_kneedle(wcss)]
            b_sil_km, b_db_km, b_ch_km = K_range[np.argmax(sil_km)], K_range[np.argmin(db_km)], K_range[np.argmax(ch_km)]
            b_sil_h, b_db_h, b_ch_h = K_range[np.argmax(sil_h)], K_range[np.argmin(db_h)], K_range[np.argmax(ch_h)]

            km_trial_k = Counter([b_sil_km]*w_sil_km + [b_ch_km]*w_ch_km + [b_elbow]*w_elbow_km).most_common(1)[0][0]
            h_trial_k = Counter([b_sil_h]*w_sil_h + [b_db_h]*w_db_h + [b_ch_h]*w_ch_h).most_common(1)[0][0]
            
            km_final_votes[km_trial_k] += 1
            h_final_votes[h_trial_k] += 1
            voting_history.append({"Lần": t+1, "K-Means": km_trial_k, "Hierarchical": h_trial_k})
            last_data = {"wcss": wcss, "sil_km": sil_km, "db_km": db_km, "ch_km": ch_km, "sil_h": sil_h, "db_h": db_h, "ch_h": ch_h}

        final_k_km = km_final_votes.most_common(1)[0][0]
        final_k_h = h_final_votes.most_common(1)[0][0]

        fig_km, axes_km = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
        axes_km = axes_km.flatten()
        axes_km[0].plot(K_range, last_data['wcss'], 'o-'); axes_km[0].set_title('Elbow')
        axes_km[1].plot(K_range, last_data['sil_km'], 's-'); axes_km[1].set_title('Silhouette')
        axes_km[2].plot(K_range, last_data['db_km'], '^-'); axes_km[2].set_title('Davies-Bouldin')
        axes_km[3].plot(K_range, last_data['ch_km'], 'D-'); axes_km[3].set_title('Calinski-Harabasz')
        for ax in axes_km: ax.axvline(final_k_km, color='r', linestyle='--'); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_h, axes_h = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)
        axes_h[0].plot(K_range, last_data['sil_h'], 's-'); axes_h[0].set_title('Silhouette')
        axes_h[1].plot(K_range, last_data['db_h'], '^-'); axes_h[1].set_title('Davies-Bouldin')
        axes_h[2].plot(K_range, last_data['ch_h'], 'D-'); axes_h[2].set_title('Calinski-Harabasz')
        for ax in axes_h: ax.axvline(final_k_h, color='r', linestyle='--'); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        detail_df = pd.DataFrame({
            "Chỉ số (Lần cuối)": ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Elbow"],
            "K tốt nhất KM": [K_range[np.argmax(last_data['sil_km'])], K_range[np.argmin(last_data['db_km'])], K_range[np.argmax(last_data['ch_km'])], K_range[self._detect_elbow_kneedle(last_data['wcss'])]],
            "K tốt nhất H": [K_range[np.argmax(last_data['sil_h'])], K_range[np.argmin(last_data['db_h'])], K_range[np.argmax(last_data['ch_h'])], "—"]
        })
        msg = f"Kết quả sau {n_trials} lần biểu quyết: KM={final_k_km}, H={final_k_h}"
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

        # Dendrogram
        from scipy.cluster.hierarchy import dendrogram, linkage as sch_linkage
        fig_dendro, ax_d = plt.subplots(figsize=(10, 5), dpi=100)
        Z = sch_linkage(X, method=linkage)
        
        # Tính toán độ cao nhát cắt cho K cụm
        # Z[:, 2] chứa khoảng cách tại mỗi bước hợp nhất. 
        # Nhát cắt cho K cụm nằm giữa bước hợp nhất thứ (N-K) và (N-K+1)
        n_samples = len(X)
        if n_samples > k_h:
            cut_height = (Z[n_samples - k_h, 2] + Z[n_samples - k_h - 1, 2]) / 2
            ax_d.axhline(y=cut_height, color='r', linestyle='--', label=f'Cut for K={k_h}')
        
        dendrogram(Z, ax=ax_d, truncate_mode='lastp', p=30)
        ax_d.set_title(f"Dendrogram (Nhát cắt cho K={k_h})")
        ax_d.legend()
        fig_dendro.tight_layout()

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
            "dendrogram": fig_dendro,
            "metrics": metrics,
            "profile_km": get_profile(km.labels_),
            "profile_h": get_profile(hcl.labels_)
        }

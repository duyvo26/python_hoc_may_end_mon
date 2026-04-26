import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import gc
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class ModelManager:
    """
    Lớp quản lý các mô hình Học máy phân cụm.
    Chịu trách nhiệm thực thi các thuật toán tìm K tối ưu (Elbow, Silhouette, ...),
    huấn luyện mô hình K-Means & Hierarchical, và trực quan hoá biểu đồ (Matplotlib/Plotly).
    """
    def __init__(self):
        self.final_labeled_df = None

    def _detect_elbow_kneedle(self, wcss_all):
        """
        Thuật toán Kneedle đơn giản: phát hiện điểm khuỷu tay bằng đạo hàm bậc 2 của WCSS.
        Điểm có đạo hàm bậc 2 lớn nhất = điểm thay đổi độ dốc mạnh nhất = điểm khuỷu tay.
        
        Returns:
            int: Chỉ số trong wcss_all tương ứng với điểm khuỷu tay (tính từ k=2).
        """
        if len(wcss_all) < 4:
            return 0
        wcss = np.array(wcss_all, dtype=np.float64)
        # Chuẩn hoá về [0,1] để so sánh công bằng
        wcss_norm = (wcss - wcss.min()) / (wcss.max() - wcss.min() + 1e-9)
        # Đạo hàm bậc 2: điểm có giá trị cao nhất là điểm khuỷu tay
        d2 = np.diff(wcss_norm, n=2)
        return int(np.argmax(d2))  # index trong K_range (bắt đầu từ k=2)

    def analyze_k(self, processed_df):
        """
        Hàm gộp: Tính toán một lần, trả về 2 K tối ưu riêng biệt cho K-Means và Hierarchical.

        Lý do 2 K có thể khác nhau:
        - K-Means: ưu tiên Silhouette ×2, CH ×2, Kneedle ×1 (cụm cầu, compact)
        - Hierarchical: ưu tiên Davies-Bouldin ×2, Silhouette ×1, Kneedle ×1 (linh hoạt hình dạng)

        Returns:
            tuple: (Figure, DataFrame chi tiết, int k_kmeans, int k_hierarchical)
        """
        X = processed_df.values.astype(np.float32)
        if len(X) < 3:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Dữ liệu quá ít", ha='center')
            return fig, pd.DataFrame([{"Lỗi": "Dữ liệu quá ít."}]), 2, 2

        limit = min(11, len(X))
        K_range = list(range(2, limit))
        use_mini_batch = len(X) > 50000
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            use_mini_batch = False

        km1 = KMeans(n_clusters=1, init='k-means++', random_state=42, n_init=10)
        km1.fit(X)
        wcss_all = [km1.inertia_]
        del km1

        sil, db, ch, results = [], [], [], []
        sil_sample = 10000 if len(X) > 10000 else None

        for k in K_range:
            km = (MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=2048)
                  if use_mini_batch else
                  KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10))
            labels = km.fit_predict(X)
            wcss_all.append(km.inertia_)
            del km
            s = silhouette_score(X, labels, sample_size=sil_sample, random_state=42)
            d = davies_bouldin_score(X, labels)
            c = calinski_harabasz_score(X, labels)
            sil.append(s); db.append(d); ch.append(c)
            results.append({'K': k, 'Silhouette (↑)': round(s, 4),
                            'Davies-Bouldin (↓)': round(d, 4),
                            'Calinski-Harabasz (↑)': round(c, 4)})
            del labels
            gc.collect()

        res_df = pd.DataFrame(results)
        best_sil   = int(res_df.loc[res_df['Silhouette (↑)'].idxmax()]['K'])
        best_db    = int(res_df.loc[res_df['Davies-Bouldin (↓)'].idxmin()]['K'])
        best_ch    = int(res_df.loc[res_df['Calinski-Harabasz (↑)'].idxmax()]['K'])
        elbow_idx  = self._detect_elbow_kneedle(wcss_all[1:])
        best_elbow = K_range[min(elbow_idx, len(K_range) - 1)]

        # ── Voting K-Means: Silhouette×2, CH×2, Kneedle×1 ───────────────────
        km_votes = Counter([best_sil, best_sil, best_ch, best_ch, best_elbow])
        k_kmeans = km_votes.most_common(1)[0][0]

        # ── Voting Hierarchical: DB×2, Silhouette×1, Kneedle×1 ───────────────
        h_votes = Counter([best_db, best_db, best_sil, best_elbow])
        k_hierarchical = h_votes.most_common(1)[0][0]

        # ── Bảng chi tiết ────────────────────────────────────────────────────
        detail_df = pd.DataFrame({
            "Phương pháp": ["Silhouette Score", "Davies-Bouldin Index",
                            "Calinski-Harabasz Index", "Elbow (Kneedle)"],
            "Tiêu chí":   ["Càng cao càng tốt", "Càng thấp càng tốt",
                            "Càng cao càng tốt", "Điểm khuỷu tay WCSS"],
            "K tốt nhất": [best_sil, best_db, best_ch, best_elbow],
            f"K-Means ({k_kmeans})":        ["×2", "—",  "×2", "×1"],
            f"Hierarchical ({k_hierarchical})": ["×1", "×2", "—",  "×1"],
        })

        # ── Biểu đồ: xanh lá = K-Means, đỏ = Hierarchical ──────────────────
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
        axes = axes.flatten()
        fig.suptitle(
            f"Hình 2: Phân tích K tối ưu — K-Means={k_kmeans}  |  Hierarchical={k_hierarchical}",
            fontsize=13, y=1.02)

        def _vlines(ax):
            ax.axvline(k_kmeans, color='#1a9641', linestyle='--', linewidth=1.8,
                       alpha=0.9, label=f'K-Means = {k_kmeans}')
            if k_hierarchical != k_kmeans:
                ax.axvline(k_hierarchical, color='#d7191c', linestyle=':', linewidth=1.8,
                           alpha=0.9, label=f'Hierarchical = {k_hierarchical}')
            ax.legend(fontsize=8)

        axes[0].plot(range(1, limit), wcss_all, marker='o', color='#2c7bb6',
                     linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[0].set_title('(a) Elbow Method (WCSS)', fontsize=12)
        axes[0].set_xlabel('Số cụm K', fontsize=11); axes[0].set_ylabel('WCSS', fontsize=11)
        axes[0].grid(True, linestyle=':', alpha=0.6); _vlines(axes[0]); sns.despine(ax=axes[0])

        axes[1].plot(K_range, sil, marker='s', color='#1a9641',
                     linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[1].set_title('(b) Silhouette Score', fontsize=12)
        axes[1].set_xlabel('Số cụm K', fontsize=11); axes[1].set_ylabel('Score', fontsize=11)
        axes[1].grid(True, linestyle=':', alpha=0.6); _vlines(axes[1]); sns.despine(ax=axes[1])

        axes[2].plot(K_range, db, marker='^', color='#d7191c',
                     linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[2].set_title('(c) Davies-Bouldin Index', fontsize=12)
        axes[2].set_xlabel('Số cụm K', fontsize=11); axes[2].set_ylabel('Index', fontsize=11)
        axes[2].grid(True, linestyle=':', alpha=0.6); _vlines(axes[2]); sns.despine(ax=axes[2])

        axes[3].plot(K_range, ch, marker='D', color='#756bb1',
                     linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[3].set_title('(d) Calinski-Harabasz Index', fontsize=12)
        axes[3].set_xlabel('Số cụm K', fontsize=11); axes[3].set_ylabel('Score', fontsize=11)
        axes[3].grid(True, linestyle=':', alpha=0.6); _vlines(axes[3]); sns.despine(ax=axes[3])

        plt.tight_layout(pad=2.0)
        plt.close(fig)
        gc.collect()
        return fig, detail_df, k_kmeans, k_hierarchical

    def run_clustering(self, processed_df, profile_base_df, k_kmeans, k_hierarchical, linkage_type):
        """
        Thực thi K-Means (dùng k_kmeans) và Hierarchical (dùng k_hierarchical) riêng biệt.
        Sau đó thực hiện giảm chiều PCA và dựng biểu đồ 3D Interactive.

        Args:
            processed_df (DataFrame): Dữ liệu đã tiền xử lý.
            profile_base_df (DataFrame): Dữ liệu gốc để profiling.
            k_kmeans (int): Số cụm cho K-Means.
            k_hierarchical (int): Số cụm cho Hierarchical.
            linkage_type (str): Phương pháp linkage.

        Returns:
            tuple: (fig_km, fig_h, fig_dendro, metrics, profile_data, final_labeled_df)
        """
        X = processed_df.values.astype(np.float32)

        # ── K-Means ──────────────────────────────────────────────────────────
        use_mini_batch = len(X) > 50000
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            use_mini_batch = False

        if use_mini_batch:
            kmeans = MiniBatchKMeans(n_clusters=k_kmeans, random_state=42, n_init=5, batch_size=2048)
        else:
            kmeans = KMeans(n_clusters=k_kmeans, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(X)

        # ── Hierarchical + KNN Approximation nếu N > 15,000 ──────────────────
        if len(X) > 15000:
            np.random.seed(42)
            indices = np.random.choice(len(X), 15000, replace=False)
            X_sample_h = X[indices]
            hierarchical = AgglomerativeClustering(n_clusters=k_hierarchical, linkage=linkage_type)
            h_labels_sample = hierarchical.fit_predict(X_sample_h)
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_sample_h, h_labels_sample)
            h_labels = knn.predict(X)
            self.X_for_dendro = X_sample_h
        else:
            hierarchical = AgglomerativeClustering(n_clusters=k_hierarchical, linkage=linkage_type)
            h_labels = hierarchical.fit_predict(X)
            self.X_for_dendro = X

        # ── PCA giảm chiều (dùng max của 2 K để đủ trục) ─────────────────────
        n_components = min(3, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        km_centroids = pca.transform(kmeans.cluster_centers_)
        h_centroids = np.array([X_pca[h_labels == i].mean(axis=0) for i in range(k_hierarchical)])

        # ── Biểu đồ 3D ───────────────────────────────────────────────────────
        if n_components == 3 and PLOTLY_AVAILABLE:
            fig_km = go.Figure()
            fig_km.add_trace(go.Scatter3d(
                x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], mode='markers',
                marker=dict(color=km_labels, colorscale='Turbo', size=6, opacity=1.0,
                            line=dict(color='black', width=1)), name='Dữ liệu'))
            fig_km.add_trace(go.Scatter3d(
                x=km_centroids[:, 0], y=km_centroids[:, 1], z=km_centroids[:, 2], mode='markers',
                marker=dict(color='darkred', symbol='x', size=4, line=dict(width=3, color='darkred')),
                name='Tâm cụm'))
            fig_km.update_layout(
                title_text=f'K-Means (K={k_kmeans}) - 3D Interactive',
                height=600, showlegend=True, margin=dict(l=0, r=0, b=0, t=40), template='plotly_white')

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter3d(
                x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2], mode='markers',
                marker=dict(color=h_labels, colorscale='Turbo', size=6, opacity=1.0,
                            line=dict(color='black', width=1)), name='Dữ liệu'))
            fig_h.add_trace(go.Scatter3d(
                x=h_centroids[:, 0], y=h_centroids[:, 1], z=h_centroids[:, 2], mode='markers',
                marker=dict(color='darkred', symbol='x', size=4, line=dict(width=3, color='darkred')),
                name='Tâm cụm'))
            fig_h.update_layout(
                title_text=f'Hierarchical (K={k_hierarchical}, {linkage_type}) - 3D Interactive',
                height=600, showlegend=True, margin=dict(l=0, r=0, b=0, t=40), template='plotly_white')

        else:
            fig_km, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
            ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='viridis', edgecolor='k', alpha=0.7, s=50)
            ax1.scatter(km_centroids[:, 0], km_centroids[:, 1], c='darkred', marker='x', s=20, linewidths=2, label='Tâm cụm')
            ax1.set_title(f'K-Means (K={k_kmeans}) - PCA')
            ax1.legend()

            fig_h, ax2 = plt.subplots(figsize=(8, 6), dpi=300)
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=h_labels, cmap='plasma', edgecolor='k', alpha=0.7, s=50)
            ax2.scatter(h_centroids[:, 0], h_centroids[:, 1], c='darkred', marker='x', s=20, linewidths=2, label='Tâm cụm')
            ax2.set_title(f'Hierarchical (K={k_hierarchical}, {linkage_type}) - PCA')
            ax2.legend()

        # ── Dendrogram (dùng k_hierarchical làm ngưỡng cắt) ─────────────────
        fig_dendro, ax_dendro = plt.subplots(figsize=(12, 5), dpi=300)
        fig_dendro.suptitle(
            f"Hình 3: Dendrogram — K-Means={k_kmeans} | Hierarchical={k_hierarchical}",
            fontsize=13, y=1.02)
        Z = linkage(self.X_for_dendro, method=linkage_type)

        if 1 < k_hierarchical <= len(Z):
            cut_distance = (Z[-k_hierarchical, 2] + Z[-k_hierarchical+1, 2]) / 2.0
            dendrogram(Z, ax=ax_dendro, truncate_mode='lastp', p=30,
                       color_threshold=cut_distance, above_threshold_color='grey')
            ax_dendro.axhline(y=cut_distance, color='red', linestyle='--', linewidth=2.5,
                              label=f'Ngưỡng cắt Hierarchical (K={k_hierarchical})')
            ax_dendro.legend(fontsize=11)
        else:
            dendrogram(Z, ax=ax_dendro, truncate_mode='lastp', p=30)

        ax_dendro.set_title("(a) Cấu trúc phân cấp", fontsize=12)
        ax_dendro.set_xlabel("Mẫu / Cụm", fontsize=11)
        ax_dendro.set_ylabel("Khoảng cách (Distance)", fontsize=11)
        sns.despine(ax=ax_dendro)
        fig_dendro.tight_layout(pad=2.0)

        # ── Bảng so sánh Metrics ─────────────────────────────────────────────
        sil_sample = 10000 if len(X) > 10000 else None
        metrics = pd.DataFrame({
            "Chỉ số": ["Silhouette Score (↑)", "Davies-Bouldin Index (↓)", "Calinski-Harabasz Index (↑)"],
            f"K-Means (K={k_kmeans})": [
                f"{silhouette_score(X, km_labels, sample_size=sil_sample, random_state=42):.4f}",
                f"{davies_bouldin_score(X, km_labels):.4f}",
                f"{calinski_harabasz_score(X, km_labels):.4f}"
            ],
            f"Hierarchical (K={k_hierarchical})": [
                f"{silhouette_score(X, h_labels, sample_size=sil_sample, random_state=42):.4f}",
                f"{davies_bouldin_score(X, h_labels):.4f}",
                f"{calinski_harabasz_score(X, h_labels):.4f}"
            ]
        })

        # ── Profiling (dựa trên nhãn K-Means) ────────────────────────────────
        self.final_labeled_df = profile_base_df.copy()
        self.final_labeled_df['Cluster_KMeans'] = km_labels
        self.final_labeled_df['Cluster_Hierarchical'] = h_labels
        profile_data = self.final_labeled_df.groupby('Cluster_KMeans').mean(numeric_only=True).reset_index()

        return fig_km, fig_h, fig_dendro, metrics, profile_data, self.final_labeled_df


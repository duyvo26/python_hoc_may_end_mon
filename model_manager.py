import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import gc
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MiniBatchKMeans
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

    def analyze_k(self, processed_df, n_trials=1):
        """
        Hàm phân tích: Tính toán các chỉ số Silhouette, DB, CH cho cả K-Means và Hierarchical.
        Trả về 2 biểu đồ riêng biệt để anh/chị dễ dàng quan sát điểm tối ưu của từng giải thuật.
        """
        X_full = processed_df.values.astype(np.float32)
        if len(X_full) < 3:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Dữ liệu quá ít", ha='center')
            return fig, fig, pd.DataFrame([{"Lỗi": "Dữ liệu quá ít."}]), 2, 2

        # --- Tối ưu RAM: Lấy mẫu nếu dữ liệu quá lớn (N > 10,000) ---
        # Việc tìm K tối ưu không nhất thiết cần toàn bộ dữ liệu nếu dữ liệu đủ lớn.
        MAX_SAMPLES = 10000 
        if len(X_full) > MAX_SAMPLES:
            np.random.seed(42)
            idx = np.random.choice(len(X_full), MAX_SAMPLES, replace=False)
            X = X_full[idx]
        else:
            X = X_full

        limit = min(11, len(X))
        K_range = list(range(2, limit))
        
        # Biến lưu trữ cho K-Means
        sum_wcss_km = np.zeros(len(range(1, limit)))
        sum_sil_km = np.zeros(len(K_range))
        sum_db_km = np.zeros(len(K_range))
        sum_ch_km = np.zeros(len(K_range))
        
        # Biến lưu trữ cho Hierarchical
        sum_sil_h = np.zeros(len(K_range))
        sum_db_h = np.zeros(len(K_range))
        sum_ch_h = np.zeros(len(K_range))

        # Silhouette score sample
        sil_sample = 5000 if len(X) > 5000 else None

        for trial in range(n_trials):
            seed = 42 + trial
            
            # WCSS cho K=1 (chỉ K-Means)
            km1 = KMeans(n_clusters=1, init='k-means++', random_state=seed, n_init=10)
            km1.fit(X)
            sum_wcss_km[0] += km1.inertia_

            for i, k in enumerate(K_range):
                # 1. K-Means Analysis
                km = KMeans(n_clusters=k, init='k-means++', random_state=seed, n_init=10)
                labels_km = km.fit_predict(X)
                sum_wcss_km[i+1] += km.inertia_
                sum_sil_km[i] += silhouette_score(X, labels_km, sample_size=sil_sample, random_state=seed)
                sum_db_km[i] += davies_bouldin_score(X, labels_km)
                sum_ch_km[i] += calinski_harabasz_score(X, labels_km)
                
                # 2. Hierarchical Analysis (Sử dụng BIRCH thay cho Agglomerative để tránh O(N^2) RAM)
                # BIRCH xây dựng cây CF-Tree, cực kỳ hiệu quả cho dữ liệu lớn.
                brc = Birch(n_clusters=k)
                labels_h = brc.fit_predict(X)
                sum_sil_h[i] += silhouette_score(X, labels_h, sample_size=sil_sample, random_state=seed)
                sum_db_h[i] += davies_bouldin_score(X, labels_h)
                sum_ch_h[i] += calinski_harabasz_score(X, labels_h)

        # Tính trung bình
        avg_wcss_km = sum_wcss_km / n_trials
        avg_sil_km = sum_sil_km / n_trials
        avg_db_km = sum_db_km / n_trials
        avg_ch_km = sum_ch_km / n_trials
        
        avg_sil_h = sum_sil_h / n_trials
        avg_db_h = sum_db_h / n_trials
        avg_ch_h = sum_ch_h / n_trials

        # Tìm K tốt nhất cho K-Means
        best_sil_km = K_range[np.argmax(avg_sil_km)]
        best_db_km = K_range[np.argmin(avg_db_km)]
        best_ch_km = K_range[np.argmax(avg_ch_km)]
        elbow_idx = self._detect_elbow_kneedle(avg_wcss_km)
        best_elbow_km = K_range[min(elbow_idx, len(K_range) - 1)]
        
        km_votes = Counter([best_sil_km, best_sil_km, best_ch_km, best_ch_km, best_elbow_km])
        k_kmeans = km_votes.most_common(1)[0][0]

        # Tìm K tốt nhất cho Hierarchical (Sử dụng BIRCH để tối ưu RAM và tốc độ)
        best_sil_h = K_range[np.argmax(avg_sil_h)]
        best_db_h = K_range[np.argmin(avg_db_h)]
        best_ch_h = K_range[np.argmax(avg_ch_h)]
        
        h_votes = Counter([best_db_h, best_db_h, best_sil_h, best_ch_h])
        k_hierarchical = h_votes.most_common(1)[0][0]

        # Bảng chi tiết
        detail_df = pd.DataFrame({
            "Chỉ số": ["Silhouette (↑)", "Davies-Bouldin (↓)", "Calinski-Harabasz (↑)", "Elbow Method"],
            "K tốt nhất (K-Means)": [best_sil_km, best_db_km, best_ch_km, best_elbow_km],
            "K tốt nhất (Hierarchical)": [best_sil_h, best_db_h, best_ch_h, "—"]
        })

        # --- Biểu đồ 1: K-Means Analysis ---
        fig_km, axes_km = plt.subplots(2, 2, figsize=(12, 8), dpi=100) # Giảm DPI để tiết kiệm memory khi render
        fig_km.suptitle(f"Hình 2a: Phân tích K tối ưu cho K-Means (Gợi ý K={k_kmeans})", fontsize=14, y=1.02)
        axes_km = axes_km.flatten()
        
        axes_km[0].plot(range(1, limit), avg_wcss_km, 'o-', color='#2c7bb6'); axes_km[0].set_title('Elbow Method (WCSS)')
        axes_km[1].plot(K_range, avg_sil_km, 's-', color='#1a9641'); axes_km[1].set_title('Silhouette Score')
        axes_km[2].plot(K_range, avg_db_km, '^-', color='#d7191c'); axes_km[2].set_title('Davies-Bouldin Index')
        axes_km[3].plot(K_range, avg_ch_km, 'D-', color='#756bb1'); axes_km[3].set_title('Calinski-Harabasz Index')
        for ax in axes_km: ax.axvline(k_kmeans, color='black', linestyle='--'); ax.grid(True, alpha=0.3)
        fig_km.tight_layout()

        # --- Biểu đồ 2: Hierarchical Analysis ---
        fig_h, axes_h = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)
        fig_h.suptitle(f"Hình 2b: Phân tích K tối ưu cho Hierarchical (Gợi ý K={k_hierarchical})", fontsize=14, y=1.05)
        
        axes_h[0].plot(K_range, avg_sil_h, 's-', color='#1a9641'); axes_h[0].set_title('Silhouette Score')
        axes_h[1].plot(K_range, avg_db_h, '^-', color='#d7191c'); axes_h[1].set_title('Davies-Bouldin Index')
        axes_h[2].plot(K_range, avg_ch_h, 'D-', color='#756bb1'); axes_h[2].set_title('Calinski-Harabasz Index')
        for ax in axes_h: ax.axvline(k_hierarchical, color='black', linestyle='--'); ax.grid(True, alpha=0.3)
        fig_h.tight_layout()

        plt.close(fig_km); plt.close(fig_h)
        gc.collect()
        return fig_km, fig_h, detail_df, k_kmeans, k_hierarchical

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
        use_mini_batch = len(X) > 25000
        gc.collect()
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            use_mini_batch = False

        if use_mini_batch:
            kmeans = MiniBatchKMeans(n_clusters=k_kmeans, random_state=42, n_init=5, batch_size=2048)
        else:
            kmeans = KMeans(n_clusters=k_kmeans, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(X)

        # ── Hierarchical sử dụng BIRCH (Chuẩn cho dữ liệu lớn) ───────────────
        # BIRCH thực hiện phân cụm phân cấp thông qua cấu trúc cây CF-Tree,
        # cho phép xử lý hàng triệu dòng dữ liệu mà vẫn giữ tính chất Hierarchical.
        birch_model = Birch(n_clusters=k_hierarchical)
        h_labels = birch_model.fit_predict(X)
        
        # Để vẽ Dendrogram, chúng ta vẫn cần một mẫu dữ liệu (vì Dendrogram không thể vẽ cho 100k điểm)
        if len(X) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(X), 5000, replace=False)
            self.X_for_dendro = X[indices]
        else:
            self.X_for_dendro = X

        # ── PCA giảm chiều (dùng max của 2 K để đủ trục) ─────────────────────
        n_components = min(3, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        gc.collect()

        # Tìm tâm cụm
        km_centroids = pca.transform(kmeans.cluster_centers_)
        h_centroids = []
        for i in range(k_hierarchical):
            mask = (h_labels == i)
            if np.any(mask):
                h_centroids.append(X_pca[mask].mean(axis=0))
            else:
                h_centroids.append(np.zeros(n_components))
        h_centroids = np.array(h_centroids)

        # ── Giới hạn số điểm vẽ để tránh treo trình duyệt (Max 10,000 điểm) ───
        MAX_PLOT_POINTS = 10000
        if len(X_pca) > MAX_PLOT_POINTS:
            np.random.seed(42)
            plot_idx = np.random.choice(len(X_pca), MAX_PLOT_POINTS, replace=False)
            X_plot = X_pca[plot_idx]
            km_labels_plot = km_labels[plot_idx]
            h_labels_plot = h_labels[plot_idx]
        else:
            X_plot = X_pca
            km_labels_plot = km_labels
            h_labels_plot = h_labels

        # ── Biểu đồ 3D ───────────────────────────────────────────────────────
        if n_components == 3 and PLOTLY_AVAILABLE:
            fig_km = go.Figure()
            fig_km.add_trace(go.Scatter3d(
                x=X_plot[:, 0], y=X_plot[:, 1], z=X_plot[:, 2], mode='markers',
                marker=dict(color=km_labels_plot, colorscale='Turbo', size=6, opacity=1.0,
                            line=dict(color='black', width=1)), name='Dữ liệu (Mẫu)'))
            fig_km.add_trace(go.Scatter3d(
                x=km_centroids[:, 0], y=km_centroids[:, 1], z=km_centroids[:, 2], mode='markers',
                marker=dict(color='darkred', symbol='x', size=4, line=dict(width=3, color='darkred')),
                name='Tâm cụm'))
            fig_km.update_layout(
                title_text=f'K-Means (K={k_kmeans}) - 3D Interactive (Hiển thị mẫu {len(X_plot)} điểm)',
                height=600, showlegend=True, margin=dict(l=0, r=0, b=0, t=40), template='plotly_white')

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter3d(
                x=X_plot[:, 0], y=X_plot[:, 1], z=X_plot[:, 2], mode='markers',
                marker=dict(color=h_labels_plot, colorscale='Turbo', size=6, opacity=1.0,
                            line=dict(color='black', width=1)), name='Dữ liệu (Mẫu)'))
            fig_h.add_trace(go.Scatter3d(
                x=h_centroids[:, 0], y=h_centroids[:, 1], z=h_centroids[:, 2], mode='markers',
                marker=dict(color='darkred', symbol='x', size=4, line=dict(width=3, color='darkred')),
                name='Tâm cụm'))
            fig_h.update_layout(
                title_text=f'Hierarchical (K={k_hierarchical}, {linkage_type}) - 3D (Hiển thị mẫu {len(X_plot)} điểm)',
                height=600, showlegend=True, margin=dict(l=0, r=0, b=0, t=40), template='plotly_white')

        else:
            fig_km, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
            ax1.scatter(X_plot[:, 0], X_plot[:, 1], c=km_labels_plot, cmap='viridis', edgecolor='k', alpha=0.7, s=50)
            ax1.scatter(km_centroids[:, 0], km_centroids[:, 1], c='darkred', marker='x', s=20, linewidths=2, label='Tâm cụm')
            ax1.set_title(f'K-Means (K={k_kmeans}) - PCA (Mẫu {len(X_plot)} điểm)')
            ax1.legend()

            fig_h, ax2 = plt.subplots(figsize=(8, 6), dpi=300)
            ax2.scatter(X_plot[:, 0], X_plot[:, 1], c=h_labels_plot, cmap='plasma', edgecolor='k', alpha=0.7, s=50)
            ax2.scatter(h_centroids[:, 0], h_centroids[:, 1], c='darkred', marker='x', s=20, linewidths=2, label='Tâm cụm')
            ax2.set_title(f'Hierarchical (K={k_hierarchical}, {linkage_type}) - PCA (Mẫu {len(X_plot)} điểm)')
            ax2.legend()

        # ── Dendrogram ───────────────────────────────────────────────────────
        Z = linkage(self.X_for_dendro, method=linkage_type)
        
        if k_kmeans == k_hierarchical:
            fig_dendro, ax_dendro = plt.subplots(figsize=(12, 5), dpi=300)
            fig_dendro.suptitle(f"Hình 3: Dendrogram — Ngưỡng cắt tối ưu (K={k_hierarchical})", fontsize=13, y=1.02)
            
            cut_distance = (Z[-k_hierarchical, 2] + Z[-k_hierarchical+1, 2]) / 2.0
            dendrogram(Z, ax=ax_dendro, truncate_mode='lastp', p=30,
                       color_threshold=cut_distance, above_threshold_color='grey')
            ax_dendro.axhline(y=cut_distance, color='red', linestyle='--', linewidth=2,
                              label=f'Ngưỡng cắt (K={k_hierarchical})')
            ax_dendro.legend(fontsize=10)
            ax_dendro.set_title("Cấu trúc phân cấp dữ liệu", fontsize=11)
            sns.despine(ax=ax_dendro)
        else:
            # Nếu 2 K khác nhau, vẽ 2 biểu đồ để so sánh
            fig_dendro, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
            fig_dendro.suptitle(f"Hình 3: So sánh Dendrogram với K khác nhau", fontsize=14, y=1.05)
            
            # Subplot 1: K-Means K
            cut_km = (Z[-k_kmeans, 2] + Z[-k_kmeans+1, 2]) / 2.0
            dendrogram(Z, ax=ax1, truncate_mode='lastp', p=30, color_threshold=cut_km, above_threshold_color='grey')
            ax1.axhline(y=cut_km, color='green', linestyle='--', label=f'Ngưỡng cắt K-Means (K={k_kmeans})')
            ax1.set_title(f"(a) Theo K-Means (K={k_kmeans})")
            ax1.legend(fontsize=9); sns.despine(ax=ax1)
            
            # Subplot 2: Hierarchical K
            cut_h = (Z[-k_hierarchical, 2] + Z[-k_hierarchical+1, 2]) / 2.0
            dendrogram(Z, ax=ax2, truncate_mode='lastp', p=30, color_threshold=cut_h, above_threshold_color='grey')
            ax2.axhline(y=cut_h, color='red', linestyle='--', label=f'Ngưỡng cắt Hierarchical (K={k_hierarchical})')
            ax2.set_title(f"(b) Theo Hierarchical (K={k_hierarchical})")
            ax2.legend(fontsize=9); sns.despine(ax=ax2)

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

        # ── Profiling (dựa trên nhãn của cả 2 mô hình) ───────────────────────
        self.final_labeled_df = profile_base_df.copy()
        self.final_labeled_df['Cluster_KMeans'] = km_labels
        self.final_labeled_df['Cluster_Hierarchical'] = h_labels
        
        profile_km = self.final_labeled_df.groupby('Cluster_KMeans').mean(numeric_only=True).reset_index()
        profile_h = self.final_labeled_df.groupby('Cluster_Hierarchical').mean(numeric_only=True).reset_index()

        return fig_km, fig_h, fig_dendro, metrics, profile_km, profile_h, self.final_labeled_df


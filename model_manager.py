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

    def get_elbow_plot(self, processed_df):
        """
        Vẽ biểu đồ Elbow và các chỉ số (Silhouette, Davies-Bouldin, Calinski-Harabasz) 
        để trực quan hoá quá trình tìm số lượng cụm (K) tối ưu.
        
        Args:
            processed_df (DataFrame): Dữ liệu đã qua tiền xử lý.
            
        Returns:
            Figure: Đối tượng Figure của matplotlib chứa 4 biểu đồ con.
        """
        X = processed_df.values.astype(np.float32)  # float32 tiết kiệm 50% RAM
        if len(X) < 3:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Dữ liệu quá ít", ha='center')
            return fig
            
        limit = min(11, len(X))
        K_range = range(2, limit)
        
        # Dùng MiniBatchKMeans cho dataset lớn (> 50,000 dòng) để giảm tải RAM đáng kể
        use_mini_batch = len(X) > 50000
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            use_mini_batch = False
        
        km1 = KMeans(n_clusters=1, init='k-means++', random_state=42, n_init=10)
        km1.fit(X)
        wcss_all = [km1.inertia_]
        del km1  # Giải phóng bộ nhớ người dùng
        
        sil = []
        db = []
        ch = []
        
        for k in K_range:
            if use_mini_batch:
                km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=2048)
            else:
                km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = km.fit_predict(X)
            wcss_all.append(km.inertia_)
            
            sil_sample = 10000 if len(X) > 10000 else None
            sil.append(silhouette_score(X, labels, sample_size=sil_sample, random_state=42))
            db.append(davies_bouldin_score(X, labels))
            ch.append(calinski_harabasz_score(X, labels))
            del km, labels  # Dọn dắp từng vòng
            gc.collect()
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
        axes = axes.flatten()
        fig.suptitle("Hình 2: Phân tích số lượng cụm tối ưu", fontsize=15, y=1.01)
        
        axes[0].plot(range(1, limit), wcss_all, marker='o', color='#2c7bb6', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[0].set_title('(a) Elbow Method (WCSS)', fontsize=12)
        axes[0].set_xlabel('Số cụm K', fontsize=11)
        axes[0].set_ylabel('WCSS', fontsize=11)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        sns.despine(ax=axes[0])
        
        axes[1].plot(K_range, sil, marker='s', color='#1a9641', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[1].set_title('(b) Silhouette Score', fontsize=12)
        axes[1].set_xlabel('Số cụm K', fontsize=11)
        axes[1].set_ylabel('Score', fontsize=11)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        sns.despine(ax=axes[1])
        
        axes[2].plot(K_range, db, marker='^', color='#d7191c', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[2].set_title('(c) Davies-Bouldin Index', fontsize=12)
        axes[2].set_xlabel('Số cụm K', fontsize=11)
        axes[2].set_ylabel('Index', fontsize=11)
        axes[2].grid(True, linestyle=':', alpha=0.6)
        sns.despine(ax=axes[2])
        
        axes[3].plot(K_range, ch, marker='D', color='#756bb1', linewidth=2, markersize=6, markerfacecolor='white', markeredgewidth=1.5)
        axes[3].set_title('(d) Calinski-Harabasz Index', fontsize=12)
        axes[3].set_xlabel('Số cụm K', fontsize=11)
        axes[3].set_ylabel('Score', fontsize=11)
        axes[3].grid(True, linestyle=':', alpha=0.6)
        sns.despine(ax=axes[3])
        
        plt.tight_layout(pad=2.0)
        plt.close(fig)  # Đóng figure khỏi RAM Matplotlib (không ảnh hưởng Gradio)
        gc.collect()
        return fig

    def find_optimal_k(self, processed_df):
        """
        Thuật toán tự động tìm số lượng cụm (K) tối ưu thông qua cơ chế biểu quyết đa số (Voting System)
        dựa trên 3 chỉ số đo lường hiệu năng phân cụm.
        
        Args:
            processed_df (DataFrame): Dữ liệu đã qua tiền xử lý.
            
        Returns:
            tuple: K tối ưu (int) và Bảng chi tiết kết quả biểu quyết (DataFrame).
        """
        X = processed_df.values.astype(np.float32)  # float32 tiết kiệm 50% RAM
        if len(X) < 3:
            return 2, pd.DataFrame([{"Lỗi": "Dữ liệu quá ít để tìm K."}])
        
        limit = min(11, len(X))
        use_mini_batch = len(X) > 50000
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            use_mini_batch = False
        results = []
        
        for k in range(2, limit):
            if use_mini_batch:
                km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=2048)
            else:
                km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = km.fit_predict(X)
            del km  # Giải phóng ngay sau khi dùng xong
            
            sil_sample = 10000 if len(X) > 10000 else None
            sil_score = silhouette_score(X, labels, sample_size=sil_sample, random_state=42)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            del labels
            gc.collect()
            
            results.append({
                'K': k,
                'Silhouette (↑)': sil_score,
                'Davies-Bouldin (↓)': db_score,
                'Calinski-Harabasz (↑)': ch_score
            })
            
        res_df = pd.DataFrame(results)
        
        best_sil = res_df.loc[res_df['Silhouette (↑)'].idxmax()]['K']
        best_db = res_df.loc[res_df['Davies-Bouldin (↓)'].idxmin()]['K']
        best_ch = res_df.loc[res_df['Calinski-Harabasz (↑)'].idxmax()]['K']
        
        votes = [int(best_sil), int(best_db), int(best_ch)]
        vote_counts = Counter(votes)
        final_k = vote_counts.most_common(1)[0][0]
        
        detail_df = pd.DataFrame({
            "Phương pháp": ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"],
            "Tiêu chí": ["Càng cao càng tốt", "Càng thấp càng tốt", "Càng cao càng tốt"],
            "K Đề Xuất": [int(best_sil), int(best_db), int(best_ch)]
        })
                
        return final_k, detail_df

    def run_clustering(self, processed_df, profile_base_df, n_clusters, linkage_type):
        """
        Thực thi thuật toán K-Means và Hierarchical Clustering trên dữ liệu đầu vào.
        Sau đó thực hiện giảm chiều dữ liệu bằng PCA (nếu cần) và dựng biểu đồ 3D Interactive.
        
        Args:
            processed_df (DataFrame): Dữ liệu đã tiền xử lý dùng để huấn luyện.
            profile_base_df (DataFrame): Dữ liệu gốc để lấy đặc trưng (profiling) sau khi gán nhãn.
            n_clusters (int): Số lượng cụm (K) cần phân chia.
            linkage_type (str): Phương pháp linkage cho Hierarchical Clustering ('ward', 'complete', 'average', 'single').
            
        Returns:
            tuple: Trả về 6 giá trị gồm biểu đồ 3D K-Means, biểu đồ 3D Hierarchical, biểu đồ Dendrogram,
                   bảng so sánh hiệu năng (metrics), bảng đặc trưng cụm (profiling), và tập dữ liệu cuối cùng đã gắn nhãn.
        """
        X = processed_df.values.astype(np.float32)
        
        # K-Means (dùng MiniBatchKMeans nếu data quá lớn)
        use_mini_batch = len(X) > 50000
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            use_mini_batch = False
            
        if use_mini_batch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=5, batch_size=2048)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(X)
        
        # Tối ưu RAM cho Hierarchical Clustering (Độ phức tạp O(N^2))
        # Nếu data > 15,000 dòng, dùng Subsampling + KNN để gán nhãn tránh OOM (Out of memory)
        if len(X) > 15000:
            np.random.seed(42)
            indices = np.random.choice(len(X), 15000, replace=False)
            X_sample_h = X[indices]
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
            h_labels_sample = hierarchical.fit_predict(X_sample_h)
            
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_sample_h, h_labels_sample)
            h_labels = knn.predict(X)
            self.X_for_dendro = X_sample_h # Lưu riêng mẫu nhỏ để vẽ Dendrogram
        else:
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
            h_labels = hierarchical.fit_predict(X)
            self.X_for_dendro = X
        
        n_components = min(3, X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        km_centroids = pca.transform(kmeans.cluster_centers_)
        h_centroids = np.array([X_pca[h_labels == i].mean(axis=0) for i in range(n_clusters)])
        
        if n_components == 3 and PLOTLY_AVAILABLE:
            # K-Means Figure
            fig_km = go.Figure()
            fig_km.add_trace(go.Scatter3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                       mode='markers', marker=dict(color=km_labels, colorscale='Turbo', size=6, opacity=1.0, line=dict(color='black', width=1)),
                                       name='Dữ liệu'))
            fig_km.add_trace(go.Scatter3d(x=km_centroids[:, 0], y=km_centroids[:, 1], z=km_centroids[:, 2],
                                       mode='markers', marker=dict(color='darkred', symbol='x', size=4, line=dict(width=3, color='darkred')),
                                       name='Tâm cụm'))
            fig_km.update_layout(title_text=f'K-Means (k={n_clusters}) - 3D Interactive', height=600, showlegend=True, margin=dict(l=0, r=0, b=0, t=40), template='plotly_white')
            
            # Hierarchical Figure
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter3d(x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
                                       mode='markers', marker=dict(color=h_labels, colorscale='Turbo', size=6, opacity=1.0, line=dict(color='black', width=1)),
                                       name='Dữ liệu'))
            fig_h.add_trace(go.Scatter3d(x=h_centroids[:, 0], y=h_centroids[:, 1], z=h_centroids[:, 2],
                                       mode='markers', marker=dict(color='darkred', symbol='x', size=4, line=dict(width=3, color='darkred')),
                                       name='Tâm cụm'))
            fig_h.update_layout(title_text=f'Hierarchical ({linkage_type}) - 3D Interactive', height=600, showlegend=True, margin=dict(l=0, r=0, b=0, t=40), template='plotly_white')
            
        elif n_components == 3:
            fig_km = plt.figure(figsize=(18, 6), dpi=200)
            fig_h = plt.figure(figsize=(18, 6), dpi=200)
            angles = [(20, 30), (20, 120), (90, 0)]
            titles = ["Góc nhìn xiên 1", "Góc nhìn xiên 2", "Từ trên xuống"]
            
            for i, (elev, azim) in enumerate(angles):
                # K-Means
                ax1 = fig_km.add_subplot(1, 3, i + 1, projection='3d')
                ax1.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=km_labels, cmap='viridis', edgecolor='k', alpha=0.7, s=30)
                ax1.scatter(km_centroids[:, 0], km_centroids[:, 1], km_centroids[:, 2], c='darkred', marker='x', s=15, linewidths=2)
                ax1.view_init(elev=elev, azim=azim)
                ax1.set_title(f'{titles[i]}')
                
                # Hierarchical
                ax2 = fig_h.add_subplot(1, 3, i + 1, projection='3d')
                ax2.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=h_labels, cmap='plasma', edgecolor='k', alpha=0.7, s=30)
                ax2.scatter(h_centroids[:, 0], h_centroids[:, 1], h_centroids[:, 2], c='darkred', marker='x', s=15, linewidths=2)
                ax2.view_init(elev=elev, azim=azim)
                ax2.set_title(f'{titles[i]}')
            
            fig_km.suptitle(f'K-Means (k={n_clusters}) - 3D PCA', fontsize=14)
            fig_h.suptitle(f'Hierarchical ({linkage_type}) - 3D PCA', fontsize=14)
            fig_km.tight_layout()
            fig_h.tight_layout()
        else:
            fig_km, ax1 = plt.subplots(figsize=(8, 6), dpi=200)
            ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='viridis', edgecolor='k', alpha=0.7, s=50)
            ax1.scatter(km_centroids[:, 0], km_centroids[:, 1], c='darkred', marker='x', s=20, linewidths=2, label='Tâm cụm')
            ax1.set_title(f'K-Means (k={n_clusters}) - 2D PCA')
            ax1.legend()
            
            fig_h, ax2 = plt.subplots(figsize=(8, 6), dpi=200)
            ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=h_labels, cmap='plasma', edgecolor='k', alpha=0.7, s=50)
            ax2.scatter(h_centroids[:, 0], h_centroids[:, 1], c='darkred', marker='x', s=20, linewidths=2, label='Tâm cụm')
            ax2.set_title(f'Hierarchical ({linkage_type}) - 2D PCA')
            ax2.legend()
        
        fig_dendro, ax_dendro = plt.subplots(figsize=(12, 5), dpi=300)
        fig_dendro.suptitle(f"Hình 3: Biểu đồ phân cấp (Dendrogram) - K={n_clusters}", fontsize=14, y=1.02)
        Z = linkage(self.X_for_dendro, method=linkage_type)
        
        # Vẽ đường cắt ngang (cut line) & màu sắc nhánh
        if 1 < n_clusters <= len(Z):
            cut_distance = (Z[-n_clusters, 2] + Z[-n_clusters+1, 2]) / 2.0
            dendrogram(Z, ax=ax_dendro, truncate_mode='lastp', p=30, color_threshold=cut_distance, above_threshold_color='grey')
            ax_dendro.axhline(y=cut_distance, color='red', linestyle='--', linewidth=2.5, label=f'Ngưỡng cắt (K={n_clusters})')
            ax_dendro.legend(fontsize=12)
        else:
            dendrogram(Z, ax=ax_dendro, truncate_mode='lastp', p=30)
            
        ax_dendro.set_title("(a) Cấu trúc phân cấp", fontsize=12)
        ax_dendro.set_xlabel("Mẫu / Cụm", fontsize=11)
        ax_dendro.set_ylabel("Khoảng cách (Distance)", fontsize=11)
        sns.despine(ax=ax_dendro)
        fig_dendro.tight_layout(pad=2.0)
        
        sil_sample = 10000 if len(X) > 10000 else None
        metrics = pd.DataFrame({
            "Chỉ số": ["Silhouette Score (Càng cao càng tốt)", "Davies-Bouldin Index (Càng thấp càng tốt)", "Calinski-Harabasz Index (Càng cao càng tốt)"],
            "K-Means": [f"{silhouette_score(X, km_labels, sample_size=sil_sample, random_state=42):.4f}", f"{davies_bouldin_score(X, km_labels):.4f}", f"{calinski_harabasz_score(X, km_labels):.4f}"],
            "Hierarchical": [f"{silhouette_score(X, h_labels, sample_size=sil_sample, random_state=42):.4f}", f"{davies_bouldin_score(X, h_labels):.4f}", f"{calinski_harabasz_score(X, h_labels):.4f}"]
        })
        
        self.final_labeled_df = profile_base_df.copy()
        self.final_labeled_df['Cluster_Label'] = km_labels 
        profile_data = self.final_labeled_df.groupby('Cluster_Label').mean(numeric_only=True).reset_index()
        
        self.final_labeled_df['Cluster_Hierarchical'] = h_labels
        
        return fig_km, fig_h, fig_dendro, metrics, profile_data, self.final_labeled_df

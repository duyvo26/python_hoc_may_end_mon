import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from collections import Counter

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class ModelManager:
    def __init__(self):
        self.final_labeled_df = None

    def get_elbow_plot(self, processed_df):
        X = processed_df.values
        if len(X) < 3:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Dữ liệu quá ít", ha='center')
            return fig
            
        limit = min(11, len(X))
        K_range = range(2, limit)
        
        km1 = KMeans(n_clusters=1, init='k-means++', random_state=42, n_init=10)
        km1.fit(X)
        wcss_all = [km1.inertia_]
        
        sil = []
        db = []
        ch = []
        
        for k in K_range:
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = km.fit_predict(X)
            wcss_all.append(km.inertia_)
            sil.append(silhouette_score(X, labels))
            db.append(davies_bouldin_score(X, labels))
            ch.append(calinski_harabasz_score(X, labels))
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
        axes = axes.flatten()
        
        axes[0].plot(range(1, limit), wcss_all, marker='o', color='#3498db')
        axes[0].set_title('Elbow Method (WCSS)')
        axes[0].set_xlabel('Số lượng cụm (k)')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(K_range, sil, marker='s', color='#2ecc71')
        axes[1].set_title('Silhouette Score (↑ Càng cao càng tốt)')
        axes[1].set_xlabel('Số lượng cụm (k)')
        axes[1].grid(alpha=0.3)
        
        axes[2].plot(K_range, db, marker='^', color='#e74c3c')
        axes[2].set_title('Davies-Bouldin Index (↓ Càng thấp càng tốt)')
        axes[2].set_xlabel('Số lượng cụm (k)')
        axes[2].grid(alpha=0.3)
        
        axes[3].plot(K_range, ch, marker='D', color='#f1c40f')
        axes[3].set_title('Calinski-Harabasz Index (↑ Càng cao càng tốt)')
        axes[3].set_xlabel('Số lượng cụm (k)')
        axes[3].grid(alpha=0.3)
        
        plt.tight_layout()
        return fig

    def find_optimal_k(self, processed_df):
        X = processed_df.values
        if len(X) < 3:
            return 2, pd.DataFrame([{"Lỗi": "Dữ liệu quá ít để tìm K."}])
        
        limit = min(11, len(X))
        results = []
        
        for k in range(2, limit):
            km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
            labels = km.fit_predict(X)
            
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            ch_score = calinski_harabasz_score(X, labels)
            
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
        X = processed_df.values
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(X)
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
        h_labels = hierarchical.fit_predict(X)
        
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
        
        fig_dendro, ax_dendro = plt.subplots(figsize=(12, 6), dpi=200)
        Z = linkage(X, method=linkage_type)
        dendrogram(Z, ax=ax_dendro, truncate_mode='lastp', p=30)
        ax_dendro.set_title("Cấu trúc phân cấp (Dendrogram)")
        
        # Vẽ đường cắt ngang (cut line)
        if 1 < n_clusters <= len(Z):
            cut_distance = (Z[-n_clusters, 2] + Z[-n_clusters+1, 2]) / 2.0
            ax_dendro.axhline(y=cut_distance, color='red', linestyle='--', linewidth=2, label=f'Ngưỡng cắt (K={n_clusters})')
            ax_dendro.legend()
        
        metrics = pd.DataFrame({
            "Chỉ số": ["Silhouette Score (Càng cao càng tốt)", "Davies-Bouldin Index (Càng thấp càng tốt)", "Calinski-Harabasz Index (Càng cao càng tốt)"],
            "K-Means": [f"{silhouette_score(X, km_labels):.4f}", f"{davies_bouldin_score(X, km_labels):.4f}", f"{calinski_harabasz_score(X, km_labels):.4f}"],
            "Hierarchical": [f"{silhouette_score(X, h_labels):.4f}", f"{davies_bouldin_score(X, h_labels):.4f}", f"{calinski_harabasz_score(X, h_labels):.4f}"]
        })
        
        self.final_labeled_df = profile_base_df.copy()
        self.final_labeled_df['Cluster_Label'] = km_labels 
        profile_data = self.final_labeled_df.groupby('Cluster_Label').mean(numeric_only=True).reset_index()
        
        self.final_labeled_df['Cluster_Hierarchical'] = h_labels
        
        return fig_km, fig_h, fig_dendro, metrics, profile_data, self.final_labeled_df

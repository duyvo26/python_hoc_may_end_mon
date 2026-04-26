import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter

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
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=km_labels, cmap='viridis', edgecolors='white', alpha=0.7)
        axes[0].set_title(f'K-Means (k={n_clusters})')
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=h_labels, cmap='plasma', edgecolors='white', alpha=0.7)
        axes[1].set_title(f'Hierarchical ({linkage_type})')
        
        fig_dendro, ax_dendro = plt.subplots(figsize=(10, 5))
        Z = linkage(X, method=linkage_type)
        dendrogram(Z, ax=ax_dendro, truncate_mode='lastp', p=15)
        ax_dendro.set_title("Cấu trúc phân cấp (Dendrogram)")
        
        metrics = pd.DataFrame({
            "Chỉ số": ["Silhouette Score (Càng cao càng tốt)", "Davies-Bouldin Index (Càng thấp càng tốt)"],
            "K-Means": [f"{silhouette_score(X, km_labels):.4f}", f"{davies_bouldin_score(X, km_labels):.4f}"],
            "Hierarchical": [f"{silhouette_score(X, h_labels):.4f}", f"{davies_bouldin_score(X, h_labels):.4f}"]
        })
        
        self.final_labeled_df = profile_base_df.copy()
        self.final_labeled_df['Cluster_Label'] = km_labels 
        profile_data = self.final_labeled_df.groupby('Cluster_Label').mean(numeric_only=True).reset_index()
        
        self.final_labeled_df['Cluster_Hierarchical'] = h_labels
        
        return fig, fig_dendro, metrics, profile_data, self.final_labeled_df

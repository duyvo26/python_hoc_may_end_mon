    def analyze_k(self, processed_df, n_trials=1):
        """
        Phân tích K tối ưu bằng cách chạy thử n_trials lần và thực hiện biểu quyết (Voting).
        """
        X = processed_df.values.astype(np.float32)
        if len(X) < 3:
            fig = plt.figure(); plt.text(0.5, 0.5, "Dữ liệu quá ít", ha='center')
            return fig, fig, pd.DataFrame(), "Dữ liệu quá ít", 2, 2, []

        K_range = range(2, 11)
        limit = 11
        
        # Để lưu kết quả biểu quyết
        km_final_votes = Counter()
        h_final_votes = Counter()
        voting_history = []

        # Các trọng số (Weights)
        w_sil_km = int(os.getenv("WEIGHT_SIL_KM", 2))
        w_ch_km = int(os.getenv("WEIGHT_CH_KM", 2))
        w_elbow_km = int(os.getenv("WEIGHT_ELBOW_KM", 1))
        w_db_h = int(os.getenv("WEIGHT_DB_H", 2))
        w_sil_h = int(os.getenv("WEIGHT_SIL_H", 1))
        w_ch_h = int(os.getenv("WEIGHT_CH_H", 1))

        # Placeholder cho các chỉ số (sẽ dùng kết quả của lần chạy cuối cùng để vẽ biểu đồ)
        last_wcss = last_sil_km = last_db_km = last_ch_km = None
        last_sil_h = last_db_h = last_ch_h = None

        for t in range(n_trials):
            # K-Means metrics
            wcss = []
            sil_km = []
            db_km = []
            ch_km = []
            
            # Hierarchical metrics
            sil_h = []
            db_h = []
            ch_h = []

            for k in K_range:
                # KM
                km = KMeans(n_clusters=k, n_init='auto', random_state=42+t).fit(X)
                wcss.append(km.inertia_)
                sil_km.append(silhouette_score(X, km.labels_))
                db_km.append(davies_bouldin_score(X, km.labels_))
                ch_km.append(calinski_harabasz_score(X, km.labels_))

                # Hierarchical
                hcl = AgglomerativeClustering(n_clusters=k).fit(X)
                sil_h.append(silhouette_score(X, hcl.labels_))
                db_h.append(davies_bouldin_score(X, hcl.labels_))
                ch_h.append(calinski_harabasz_score(X, hcl.labels_))

            # Tìm K tốt nhất cho Trial này
            b_elbow = K_range[self._detect_elbow_kneedle(wcss)]
            b_sil_km = K_range[np.argmax(sil_km)]
            b_db_km = K_range[np.argmin(db_km)]
            b_ch_km = K_range[np.argmax(ch_km)]

            b_sil_h = K_range[np.argmax(sil_h)]
            b_db_h = K_range[np.argmin(db_h)]
            b_ch_h = K_range[np.argmax(ch_h)]

            # Vote
            km_trial_k = Counter([b_sil_km]*w_sil_km + [b_ch_km]*w_ch_km + [b_elbow_km]*w_elbow_km).most_common(1)[0][0]
            h_trial_k = Counter([b_sil_h]*w_sil_h + [b_db_h]*w_db_h + [b_ch_h]*w_ch_h).most_common(1)[0][0]
            
            km_final_votes[km_trial_k] += 1
            h_final_votes[h_trial_k] += 1
            
            voting_history.append({
                "Lần": t + 1,
                "K-Means Gợi ý": km_trial_k,
                "Hierarchical Gợi ý": h_trial_k
            })

            # Lưu lại data lần cuối để vẽ
            last_wcss, last_sil_km, last_db_km, last_ch_km = wcss, sil_km, db_km, ch_km
            last_sil_h, last_db_h, last_ch_h = sil_h, db_h, ch_h

        # Kết quả cuối cùng sau N lần vote
        final_k_km = km_final_votes.most_common(1)[0][0]
        final_k_h = h_final_votes.most_common(1)[0][0]

        # Vẽ biểu đồ (dùng kết quả trial cuối)
        fig_km, axes_km = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
        fig_km.suptitle(f"Phân tích K-Means (Biểu quyết cuối: K={final_k_km})")
        axes_km = axes_km.flatten()
        axes_km[0].plot(K_range, last_wcss, 'o-'); axes_km[0].set_title('Elbow (WCSS)')
        axes_km[1].plot(K_range, last_sil_km, 's-'); axes_km[1].set_title('Silhouette')
        axes_km[2].plot(K_range, last_db_km, '^-'); axes_km[2].set_title('Davies-Bouldin')
        axes_km[3].plot(K_range, last_ch_km, 'D-'); axes_km[3].set_title('Calinski-Harabasz')
        for ax in axes_km: ax.axvline(final_k_km, color='r', linestyle='--'); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_h, axes_h = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)
        fig_h.suptitle(f"Phân tích Hierarchical (Biểu quyết cuối: K={final_k_h})")
        axes_h[0].plot(K_range, last_sil_h, 's-'); axes_h[0].set_title('Silhouette')
        axes_h[1].plot(K_range, last_db_h, '^-'); axes_h[1].set_title('Davies-Bouldin')
        axes_h[2].plot(K_range, last_ch_h, 'D-'); axes_h[2].set_title('Calinski-Harabasz')
        for ax in axes_h: ax.axvline(final_k_h, color='r', linestyle='--'); ax.grid(True, alpha=0.3)
        plt.tight_layout()

        detail_df = pd.DataFrame({
            "Chỉ số (Lần cuối)": ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Elbow"],
            "K tốt nhất KM": [K_range[np.argmax(last_sil_km)], K_range[np.argmin(last_db_km)], K_range[np.argmax(last_ch_km)], K_range[self._detect_elbow_kneedle(last_wcss)]],
            "K tốt nhất H": [K_range[np.argmax(last_sil_h)], K_range[np.argmin(last_db_h)], K_range[np.argmax(last_ch_h)], "—"]
        })

        msg = f"Biểu quyết sau {n_trials} lần: K-Means={final_k_km}, Hierarchical={final_k_h}"
        return fig_km, fig_h, detail_df, msg, final_k_km, final_k_h, voting_history

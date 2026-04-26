import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from scipy import stats

class DataProcessor:
    def __init__(self):
        self.df = None
        self.processed_df = None
        self.profile_base_df = None

    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        cols = self.df.columns.tolist()
        html_table = self.df.head().to_html(classes='table table-striped', index=False)
        preview = f"<div style='overflow-x: auto; max-width: 100%;'>{html_table}</div>"
        
        # Heatmap
        numeric_df = self.df.select_dtypes(include=[np.number])
        fig_corr, ax = plt.subplots(figsize=(12, 8), dpi=200)
        if not numeric_df.empty:
            # Chỉ hiển thị số liệu nếu số cột <= 10 để tránh rối rắm
            show_annot = len(numeric_df.columns) <= 12
            sns.heatmap(numeric_df.corr(), annot=show_annot, fmt=".2f", cmap='coolwarm', center=0, 
                        linewidths=1.0, linecolor='white', annot_kws={"size": 10, "weight": "bold"}, ax=ax)
            ax.set_title("Ma trận tương quan đặc trưng (Correlation Heatmap)", fontsize=16, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, "Không có dữ liệu số để tính tương quan", ha='center')
            
        return preview, cols, fig_corr

    def preprocess_data(self, cols_to_drop, imputer_method, scaler_method, remove_outliers):
        if self.df is None:
            raise ValueError("Chưa có dữ liệu.")
            
        temp_df = self.df.drop(columns=cols_to_drop)
        
        if remove_outliers:
            numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                z_data = temp_df[numeric_cols].fillna(temp_df[numeric_cols].mean())
                z_scores = np.abs(stats.zscore(z_data))
                temp_df = temp_df[(z_scores < 3).all(axis=1)]
        
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
        categorical_cols = temp_df.select_dtypes(exclude=[np.number]).columns
        
        if imputer_method == "Drop":
            temp_df = temp_df.dropna()
        else:
            for col in numeric_cols:
                fill_val = temp_df[col].mean() if imputer_method == "Mean" else temp_df[col].median()
                temp_df[col] = temp_df[col].fillna(fill_val)
            for col in categorical_cols:
                mode_val = temp_df[col].mode()[0] if not temp_df[col].mode().empty else "Missing"
                temp_df[col] = temp_df[col].fillna(mode_val)

        # Giữ lại dataframe đã xử lý thiếu/nhiễu nhưng chưa mã hóa/chuẩn hóa để làm profile
        self.profile_base_df = temp_df.copy()

        for col in categorical_cols:
            le = LabelEncoder()
            temp_df[col] = le.fit_transform(temp_df[col].astype(str))
        
        scaler = StandardScaler() if scaler_method == "StandardScaler" else MinMaxScaler()
        scaled_data = scaler.fit_transform(temp_df)
            
        self.processed_df = pd.DataFrame(scaled_data, columns=temp_df.columns)
        return self.processed_df, self.profile_base_df

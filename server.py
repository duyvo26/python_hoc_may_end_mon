import os
import shutil
import uuid
import threading
import time
import io
import base64
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data_processor import DataProcessor
from model_manager import ModelManager

from styles import setup_scientific_plots, get_sys_info

app = Flask(__name__)
setup_scientific_plots() # Áp dụng cấu hình biểu đồ chuẩn khoa học

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Khởi tạo các module lõi
processor = DataProcessor()
model_manager = ModelManager()

# Quản lý tác vụ ngầm
tasks = {} 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sys-info')
def sys_info_api():
    info = get_sys_info()
    return jsonify({"cpu": info.get("cpu", 0), "ram": info.get("ram", 0)})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    ext = file.filename.split('.')[-1].lower()
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    preview, columns = processor.load_data(filepath)
    if processor.df is None:
        return jsonify({"error": "Không thể nạp dữ liệu"}), 400
    
    return jsonify({
        "message": "Upload thành công",
        "columns": columns,
        "preview": preview,
        "filename": file.filename
    })

@app.route('/api/heatmap', methods=['GET'])
def get_heatmap():
    if processor.df is None: return jsonify({"error": "No data"}), 400
    fig = processor.plot_correlation()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return jsonify({"image": img_str})

@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    data = request.json
    drop_cols = data.get('drop_cols', [])
    imp_method = data.get('imp_method', 'Mean')
    scl_method = data.get('scl_method', 'StandardScaler')
    remove_outliers = data.get('remove_outliers', True)
    
    df, profile_df = processor.preprocess_data(drop_cols, imp_method, scl_method, remove_outliers)
    if df is None:
        return jsonify({"error": "Lỗi tiền xử lý"}), 400
        
    return jsonify({
        "message": "Tiền xử lý hoàn tất",
        "preview": df.head(5).to_dict(orient='records'),
        "shape": df.shape
    })

@app.route('/api/analyze-k', methods=['POST'])
def start_analyze_k():
    data = request.json
    n_trials = int(data.get('n_trials', 1))
    sid = data.get('session_id', 'default_session')
    
    task_id = f"analyze_{str(uuid.uuid4())[:8]}"
    tasks[task_id] = {"status": "running", "start_time": time.time(), "message": "Đang phân tích K tối ưu..."}
    
    def run_bg(tid, sid_val):
        try:
            fig_km, fig_h, k_details, best_k_msg, k_km_suggest, k_h_suggest, v_hist = model_manager.analyze_k(
                processor.processed_df, n_trials=n_trials
            )
            
            def prepare_figs(figs):
                out = []
                for f in figs:
                    buf = io.BytesIO()
                    f.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    plt.close(f)
                    buf.seek(0)
                    out.append({"type": "image", "data": base64.b64encode(buf.read()).decode('utf-8')})
                return out

            # 1. Lưu file vào thư mục Session TRƯỚC (figure còn mở)
            session_dir = os.path.join(UPLOAD_FOLDER, sid_val)
            os.makedirs(session_dir, exist_ok=True)
            fig_km.savefig(os.path.join(session_dir, "1_Analysis_KMeans.png"), bbox_inches='tight', dpi=300)
            fig_h.savefig(os.path.join(session_dir, "1_Analysis_Hierarchical.png"), bbox_inches='tight', dpi=300)
            k_details.to_csv(os.path.join(session_dir, "1_K_Metrics_Detail.csv"), index=False)
            pd.DataFrame(v_hist).to_csv(os.path.join(session_dir, "1_Voting_History.csv"), index=False)
            processor.processed_df.to_csv(os.path.join(session_dir, "0_Processed_Data.csv"), index=False)

            # 2. Convert sang base64 để gửi lên giao diện (đóng figure sau khi lưu xong)
            def prepare_figs(figs):
                out = []
                for f in figs:
                    buf = io.BytesIO()
                    f.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    plt.close(f)
                    buf.seek(0)
                    out.append({"type": "image", "data": base64.b64encode(buf.read()).decode('utf-8')})
                return out

            plots_data = prepare_figs([fig_km, fig_h])

            result_data = {
                "plot_km": plots_data[0],
                "plot_h": plots_data[1],
                "k_details": k_details.to_dict(orient='records'),
                "voting_history": v_hist,
                "suggestion": best_k_msg,
                "k_km_suggest": int(k_km_suggest),
                "k_h_suggest": int(k_h_suggest)
            }
            tasks[tid]["result"] = result_data
            tasks[tid]["status"] = "completed"
        except Exception as e:
            tasks[tid]["status"] = "failed"
            tasks[tid]["error"] = str(e)

    threading.Thread(target=run_bg, args=(task_id, sid)).start()
    return jsonify({"task_id": task_id})

@app.route('/api/train', methods=['POST'])
def start_train():
    data = request.json
    k_km = int(data.get('k_kmeans', 3))
    k_h = int(data.get('k_hier', 3))
    linkage = data.get('linkage', 'ward')
    sid = data.get('session_id', 'default_session')
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {"status": "running", "start_time": time.time(), "message": "Đang huấn luyện mô hình..."}
    
    def run_bg(tid, sid_val):
        try:
            res_dict = model_manager.run_clustering(
                processor.processed_df, 
                processor.profile_base_df, 
                k_km, k_h, linkage
            )
            
            session_dir = os.path.join(UPLOAD_FOLDER, sid_val)
            os.makedirs(session_dir, exist_ok=True)
            
            # Lưu file 2D
            res_dict["pca2d_km"].savefig(os.path.join(session_dir, "2_PCA_KMeans_2D.png"), bbox_inches='tight', dpi=300)
            res_dict["pca2d_h"].savefig(os.path.join(session_dir, "2_PCA_Hierarchical_2D.png"), bbox_inches='tight', dpi=300)
            res_dict["dendrogram"].savefig(os.path.join(session_dir, "2_Dendrogram.png"), bbox_inches='tight', dpi=300)
            
            res_dict["metrics"].to_csv(os.path.join(session_dir, "3_Metrics.csv"), index=False)
            res_dict["profile_km"].to_csv(os.path.join(session_dir, "4_Profile_KMeans.csv"), index=False)
            res_dict["profile_h"].to_csv(os.path.join(session_dir, "4_Profile_Hierarchical.csv"), index=False)

            def prepare_figs(figs_list):
                out = []
                for f in figs_list:
                    if hasattr(f, 'to_json'):
                        out.append({"type": "plotly", "data": f.to_json()})
                    else:
                        buf = io.BytesIO()
                        f.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        plt.close(f)
                        buf.seek(0)
                        out.append({"type": "image", "data": base64.b64encode(buf.read()).decode('utf-8')})
                return out

            plots = prepare_figs([
                res_dict["pca2d_km"], res_dict["pca2d_h"], 
                res_dict["pca3d_km"], res_dict["pca3d_h"],
                res_dict["dendrogram"]
            ])

            result_data = {
                "pca2d_km": plots[0], "pca2d_h": plots[1],
                "pca3d_km": plots[2], "pca3d_h": plots[3],
                "plot_dendro": plots[4],
                "metrics": res_dict["metrics"].to_dict(orient='records'),
                "profile_km": res_dict["profile_km"].to_dict(orient='records'),
                "profile_h": res_dict["profile_h"].to_dict(orient='records')
            }
            
            tasks[tid]["result"] = result_data
            tasks[tid]["status"] = "completed"

        except Exception as e:
            tasks[tid]["status"] = "failed"
            tasks[tid]["error"] = str(e)

    threading.Thread(target=run_bg, args=(task_id, sid)).start()
    return jsonify({"task_id": task_id})

@app.route('/api/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks: return jsonify({"error": "Not found"}), 404
    task = tasks[task_id]
    response = {"status": task["status"]}
    if task["status"] == "running":
        response["elapsed"] = int(time.time() - task.get("start_time", time.time()))
    elif task["status"] == "completed":
        if "result" in task:
            response["result"] = task["result"]
        else:
            response["status"] = "running" # Chưa gán xong result thì coi như vẫn đang chạy
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")
    return jsonify(response)

@app.route('/api/chatgpt-prompt', methods=['POST'])
def get_prompt():
    data = request.json
    prompt = f"Phân tích kết quả phân cụm:\nMetrics:\n{pd.DataFrame(data['metrics']).to_string()}\n\nProfiling K-Means:\n{pd.DataFrame(data['profile_km']).to_string()}"
    return jsonify({"prompt": prompt})

@app.route('/api/export/<session_id>')
def export_results(session_id):
    session_dir = os.path.join(UPLOAD_FOLDER, session_id)
    
    print(f"[EXPORT] Session dir: {session_dir}")
    print(f"[EXPORT] Exists: {os.path.isdir(session_dir)}")
    if os.path.isdir(session_dir):
        print(f"[EXPORT] Files: {os.listdir(session_dir)}")
    
    if not os.path.isdir(session_dir) or not os.listdir(session_dir):
        return jsonify({"error": f"Không tìm thấy kết quả cho phiên {session_id}. Hãy chạy Phân tích hoặc Huấn luyện trước."}), 404
    
    zip_name = f"Full_Report_{session_id}"
    zip_path = os.path.join(UPLOAD_FOLDER, zip_name)
    shutil.make_archive(zip_path, 'zip', session_dir)
    
    return send_file(
        os.path.abspath(zip_path + ".zip"),
        as_attachment=True,
        download_name=f"{zip_name}.zip"
    )

if __name__ == '__main__':
    # Tắt use_reloader để tránh server tự khởi động lại làm mất task ngầm và dữ liệu
    app.run(debug=True, use_reloader=False, port=5000)

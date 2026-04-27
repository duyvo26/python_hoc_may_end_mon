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

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

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
    return jsonify({"info": get_sys_info()})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Đảm bảo thư mục uploads tồn tại
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Đổi tên file thành UUID để tránh trùng lặp
    ext = os.path.splitext(file.filename)[1]
    new_filename = f"{uuid.uuid4()}{ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    file.save(path)
    
    preview_html, columns = processor.load_data(path)
    if processor.df is None:
        return jsonify({"error": "Không thể nạp dữ liệu"}), 400
    
    return jsonify({
        "message": "Upload thành công",
        "columns": columns,
        "preview": processor.df.head(5).to_dict(orient='records'),
        "preview_html": preview_html,
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
    
    task_id = "analyze_" + str(uuid.uuid4())[:8]
    tasks[task_id] = {"status": "running", "start_time": time.time(), "message": "Đang phân tích các chỉ số K tối ưu..."}
    
    def run_bg(tid):
        try:
            fig_km, fig_h, k_details, best_k_msg, k_km_suggest, k_h_suggest, v_hist = model_manager.analyze_k(
                processor.processed_df, n_trials
            )
            
            def prepare_figs(figs):
                out = []
                for f in figs:
                    if hasattr(f, 'to_json'):
                        out.append({"type": "plotly", "data": f.to_json()})
                    else:
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

    threading.Thread(target=run_bg, args=(task_id,)).start()
    return jsonify({"task_id": task_id})

@app.route('/api/train', methods=['POST'])
def start_train():
    data = request.json
    k_km = int(data.get('k_kmeans', 3))
    k_h = int(data.get('k_hier', 3))
    linkage = data.get('linkage', 'ward')
    pca_dim = data.get('pca_dim', '3D')
    
    task_id = str(uuid.uuid4())[:8]
    tasks[task_id] = {"status": "running", "start_time": time.time(), "message": "Đang huấn luyện mô hình..."}
    
    def run_bg(tid):
        try:
            res = model_manager.run_clustering(
                processor.processed_df, 
                processor.profile_base_df, 
                k_km, k_h, linkage, pca_dim
            )
            
            # Helper to prepare plots (JSON for Plotly, Base64 for Matplotlib)
            def prepare_figs(figs):
                out = []
                for f in figs:
                    if hasattr(f, 'to_json'): # Plotly (Interactive)
                        out.append({"type": "plotly", "data": f.to_json()})
                    else: # Matplotlib (Static)
                        buf = io.BytesIO()
                        f.savefig(buf, format='png', bbox_inches='tight')
                        plt.close(f)
                        buf.seek(0)
                        out.append({"type": "image", "data": base64.b64encode(buf.read()).decode('utf-8')})
                return out

            plots_data = prepare_figs([res[0], res[1], res[2]])
            
            result_data = {
                "plot_km": plots_data[0],
                "plot_h": plots_data[1],
                "plot_dendro": plots_data[2],
                "metrics": res[3].to_dict(orient='records'),
                "profile_km": res[4].to_dict(orient='records'),
                "profile_h": res[5].to_dict(orient='records'),
                "task_id": tid
            }
            tasks[tid]["result"] = result_data
            tasks[tid]["status"] = "completed"
        except Exception as e:
            tasks[tid]["status"] = "failed"
            tasks[tid]["error"] = str(e)

    threading.Thread(target=run_bg, args=(task_id,)).start()
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

@app.route('/api/export/<task_id>')
def export_results(task_id):
    # Đường dẫn thư mục kết quả
    folder_path = app.config['RESULTS_FOLDER']
    zip_name = f"Full_Report_{task_id}"
    zip_path = os.path.join('uploads', zip_name)
    
    # Nén thư mục results
    shutil.make_archive(zip_path, 'zip', folder_path)
    
    return send_file(zip_path + ".zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

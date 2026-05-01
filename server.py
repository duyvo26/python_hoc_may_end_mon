import os
import shutil
import uuid
import threading
import time
import io
import base64
import zipfile
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from data_processor import DataProcessor
from model_manager import ModelManager
from report_generator import ReportGenerator

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

@app.route('/api/upload_chunk', methods=['POST'])
def upload_chunk():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    chunk = request.files['file']
    chunk_index = int(request.form.get('chunkIndex', 0))
    total_chunks = int(request.form.get('totalChunks', 1))
    file_id = request.form.get('fileId', 'temp')
    original_filename = request.form.get('fileName', 'data.csv')
    
    temp_dir = os.path.join(UPLOAD_FOLDER, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{file_id}.part")
    
    # Ghi nối tiếp chunk vào file tạm
    mode = 'ab' if chunk_index > 0 else 'wb'
    with open(temp_path, mode) as f:
        f.write(chunk.read())
        
    if chunk_index == total_chunks - 1:
        # Khi đã nhận đủ chunk, đổi tên file và xử lý
        ext = original_filename.split('.')[-1].lower()
        final_filename = f"{uuid.uuid4()}.{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, final_filename)
        shutil.move(temp_path, filepath)
        
        preview, columns = processor.load_data(filepath)
        if processor.df is None:
            return jsonify({"error": "Không thể nạp dữ liệu"}), 400
        
        # Lưu thông tin gốc để làm báo cáo vào processor
        processor.original_shape = f"{processor.df.shape[0]} hàng, {processor.df.shape[1]} cột"
        processor.df_name = original_filename
        
        return jsonify({
            "status": "completed",
            "message": "Upload thành công",
            "columns": columns,
            "preview": preview,
            "filename": original_filename
        })
        
    return jsonify({"status": "uploading"})

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
        
    # Lưu thông tin sau tiền xử lý để làm báo cáo
    processor.processed_shape = f"{df.shape[0]} hàng, {df.shape[1]} cột"
    
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
            tasks[tid]["suggestion"] = best_k_msg
            tasks[tid]["voting_history"] = v_hist if isinstance(v_hist, list) else v_hist.to_dict(orient='records')
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
            
            res_dict["metrics"].to_csv(os.path.join(session_dir, "3_Metrics.csv"), index=False)
            res_dict["profile_km"].to_csv(os.path.join(session_dir, "4_Profile_KMeans.csv"), index=False)
            res_dict["profile_h"].to_csv(os.path.join(session_dir, "4_Profile_Hierarchical.csv"), index=False)

            def prepare_figs(figs_list):
                out = []
                for f in figs_list:
                    if hasattr(f, 'to_dict'):
                        out.append({"type": "plotly", "data": f.to_dict()})
                    else:
                        buf = io.BytesIO()
                        f.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                        plt.close(f)
                        buf.seek(0)
                        out.append({"type": "image", "data": base64.b64encode(buf.read()).decode('utf-8')})
                return out

            plots = prepare_figs([
                res_dict["pca2d_km"], res_dict["pca2d_h"], 
                res_dict["pca3d_km"], res_dict["pca3d_h"]
            ])

            result_data = {
                "pca2d_km": plots[0], "pca2d_h": plots[1],
                "pca3d_km": plots[2], "pca3d_h": plots[3],
                "metrics": res_dict["metrics"].to_dict(orient='records'),
                "profile_km": res_dict["profile_km"].to_dict(orient='records'),
                "profile_h": res_dict["profile_h"].to_dict(orient='records')
            }
            
            tasks[tid]["result"] = result_data
            tasks[tid]["status"] = "completed"

            # TỰ ĐỘNG TẠO BÁO CÁO .MD VÀ .DOCX
            try:
                # Tìm thông tin từ các bước trước
                suggestion = "Kết quả phân tích dựa trên đa chỉ số."
                v_history = []
                for old_tid in tasks:
                    if old_tid.startswith("analyze") and "suggestion" in tasks[old_tid]:
                        suggestion = tasks[old_tid]["suggestion"]
                        v_history = tasks[old_tid].get("voting_history", [])
                
                d_info = tasks.get(f"data_{sid_val}", {})
                p_info = tasks.get(f"preprocess_{sid_val}", {})
                
                reporter = ReportGenerator(session_dir)
                reporter.generate(
                    dataset_name=getattr(processor, 'df_name', 'Dữ liệu người dùng'),
                    original_info=getattr(processor, 'original_shape', 'Không xác định'),
                    preprocess_info=getattr(processor, 'processed_shape', 'Chưa qua tiền xử lý'),
                    k_details=pd.read_csv(os.path.join(session_dir, "1_K_Metrics_Detail.csv")),
                    voting_history=pd.DataFrame(v_history),
                    best_k_msg=suggestion,
                    metrics_df=res_dict["metrics"],
                    profile_km=res_dict["profile_km"],
                    profile_h=res_dict["profile_h"]
                )
            except Exception as re:
                print(f"Lỗi tạo báo cáo: {re}")

        except Exception as e:
            tasks[tid]["status"] = "failed"
            tasks[tid]["error"] = str(e)

    threading.Thread(target=run_bg, args=(task_id, sid)).start()
    return jsonify({"task_id": task_id})

@app.route('/api/status/<task_id>')
def get_status(task_id):
    if task_id not in tasks: return jsonify({"error": "Not found"}), 404
    task = tasks[task_id]
    response = {
        "status": task["status"],
        "message": task.get("message", ""),
        "logs": task.get("logs", [])
    }
    if task["status"] == "running":
        response["elapsed"] = int(time.time() - task.get("start_time", time.time()))
    elif task["status"] == "completed":
        response["result"] = task.get("result")
        response["result_url"] = task.get("result_url")
    elif task["status"] == "failed":
        response["error"] = task.get("error", "Unknown error")
    return jsonify(response)

@app.route('/api/chatgpt-prompt', methods=['POST'])
def get_prompt():
    data = request.json
    prompt = f"Phân tích kết quả phân cụm:\nMetrics:\n{pd.DataFrame(data['metrics']).to_string()}\n\nProfiling K-Means:\n{pd.DataFrame(data['profile_km']).to_string()}"
    return jsonify({"prompt": prompt})

@app.route('/api/report/md/<session_id>')
def download_report_md(session_id):
    path = os.path.join(UPLOAD_FOLDER, session_id, "Full_Report.md")
    if os.path.exists(path):
        return send_file(os.path.abspath(path), as_attachment=True, download_name=f"Report_{session_id}.md")
    return jsonify({"error": "Báo cáo chưa được tạo. Hãy chạy Huấn luyện trước."}), 404

@app.route('/api/report/docx/<session_id>', methods=['GET'])
def download_docx(session_id):
    path = os.path.join(UPLOAD_FOLDER, session_id, "Full_Report.docx")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"Clustering_Report_{session_id}.docx")
    return jsonify({"error": "Báo cáo Word chưa được tạo"}), 404

@app.route('/api/batch-process', methods=['POST'])
def batch_process():
    if 'file' not in request.files:
        return jsonify({"error": "Chưa chọn tệp tin"}), 400
    
    zip_file = request.files['file']
    if not zip_file.filename.endswith('.zip'):
        return jsonify({"error": "Vui lòng tải lên tệp định dạng .zip"}), 400

    batch_id = str(uuid.uuid4())[:8]
    batch_dir = os.path.join(UPLOAD_FOLDER, f"batch_{batch_id}")
    os.makedirs(batch_dir, exist_ok=True)
    
    zip_path = os.path.join(batch_dir, "input.zip")
    zip_file.save(zip_path)
    
    task_id = f"batch_{str(uuid.uuid4())[:8]}"
    tasks[task_id] = {"status": "running", "message": "Đang giải nén dữ liệu hàng loạt..."}
    
    def run_batch_bg(tid, b_dir, z_p):
        tasks[tid]["logs"] = []
        def add_log(msg):
            print(msg)
            tasks[tid]["logs"].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
            tasks[tid]["message"] = msg

        try:
            add_log("Bắt đầu quy trình xử lý hàng loạt...")
            extract_dir = os.path.join(b_dir, "extracted")
            add_log(f"Đang giải nén tệp tin: {os.path.basename(z_p)}")
            with zipfile.ZipFile(z_p, 'r') as z:
                z.extractall(extract_dir)
            
            csv_files = []
            for root, _, files in os.walk(extract_dir):
                for f in files:
                    if f.endswith('.csv'):
                        csv_files.append(os.path.join(root, f))
            
            add_log(f"Tìm thấy {len(csv_files)} tệp tin CSV.")
            if not csv_files:
                raise Exception("Không tìm thấy file .csv nào trong tệp zip.")

            reports_dir = os.path.join(b_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            for i, csv_path in enumerate(csv_files):
                csv_name = os.path.basename(csv_path)
                add_log(f"Đang xử lý ({i+1}/{len(csv_files)}): {csv_name}")
                
                add_log(f"[{csv_name}] Bước 1: Nạp và Tiền xử lý (Auto Scaling)...")
                p = DataProcessor()
                p.load_data(csv_path)
                p.df_name = csv_name
                df_proc, df_prof = p.preprocess_data([], 'Mean', 'StandardScaler', True)
                
                s_id = f"sub_{batch_id}_{i}"
                s_dir = os.path.join(UPLOAD_FOLDER, s_id)
                os.makedirs(s_dir, exist_ok=True)
                
                add_log(f"[{csv_name}] Bước 2: Phân tích K tối ưu (10 trials)...")
                f_km, f_h, k_det, msg, k_km, k_h, v_h = model_manager.analyze_k(
                    df_proc, n_trials=10, log_callback=add_log
                )
                
                f_km.savefig(os.path.join(s_dir, "1_Analysis_KMeans.png"), bbox_inches='tight', dpi=300)
                f_h.savefig(os.path.join(s_dir, "1_Analysis_Hierarchical.png"), bbox_inches='tight', dpi=300)
                k_det.to_csv(os.path.join(s_dir, "1_K_Metrics_Detail.csv"), index=False)
                plt.close(f_km)
                plt.close(f_h)

                add_log(f"[{csv_name}] Bước 3: Huấn luyện mô hình (K={k_km}/{k_h})...")
                res = model_manager.run_clustering(df_proc, df_prof, k_km, k_h, 'ward')
                
                res["pca2d_km"].savefig(os.path.join(s_dir, "2_PCA_KMeans_2D.png"), bbox_inches='tight', dpi=300)
                res["pca2d_h"].savefig(os.path.join(s_dir, "2_PCA_Hierarchical_2D.png"), bbox_inches='tight', dpi=300)
                res["metrics"].to_csv(os.path.join(s_dir, "3_Metrics.csv"), index=False)
                res["profile_km"].to_csv(os.path.join(s_dir, "4_Profile_KMeans.csv"), index=False)
                res["profile_h"].to_csv(os.path.join(s_dir, "4_Profile_Hierarchical.csv"), index=False)
                plt.close(res["pca2d_km"])
                plt.close(res["pca2d_h"])


                add_log(f"[{csv_name}] Bước 4: Xuất báo cáo Word...")
                rep = ReportGenerator(s_dir)
                _, docx_p = rep.generate(
                    csv_name, 
                    f"{p.df.shape[0]} hàng, {p.df.shape[1]} cột",
                    f"{df_proc.shape[0]} hàng, {df_proc.shape[1]} cột",
                    k_det, pd.DataFrame(v_h), msg,
                    res["metrics"], res["profile_km"], res["profile_h"]
                )
                
                shutil.copy(docx_p, os.path.join(reports_dir, f"Bao_cao_{csv_name.replace('.csv', '')}.docx"))
                add_log(f"[{csv_name}] Hoàn tất!")

            add_log("Đang đóng gói toàn bộ báo cáo vào file ZIP...")
            final_zip = os.path.join(b_dir, f"Ket_qua_Batch_{batch_id}.zip")
            with zipfile.ZipFile(final_zip, 'w') as fz:
                for root, _, files in os.walk(reports_dir):
                    for f in files:
                        fz.write(os.path.join(root, f), f)
            
            tasks[tid]["status"] = "completed"
            tasks[tid]["result_url"] = f"/api/batch/download/{batch_id}"
            add_log("TOÀN BỘ QUY TRÌNH HOÀN TẤT!")
            
        except Exception as e:
            tasks[tid]["status"] = "failed"
            tasks[tid]["error"] = str(e)

    threading.Thread(target=run_batch_bg, args=(task_id, batch_dir, zip_path)).start()
    return jsonify({"task_id": task_id})

@app.route('/api/batch/download/<batch_id>', methods=['GET'])
def download_batch(batch_id):
    path = os.path.join(UPLOAD_FOLDER, f"batch_{batch_id}", f"Ket_qua_Batch_{batch_id}.zip")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name=f"Batch_Results_{batch_id}.zip")
    return jsonify({"error": "Tệp tin không tồn tại"}), 404

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

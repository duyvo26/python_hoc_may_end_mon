import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def setup_styles():
    """Cấu hình thẩm mỹ chuẩn Báo cáo Khoa học (Academic Report)"""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 300
    sns.set_theme(style="ticks", rc={"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]})
    sns.set_palette("colorblind")

def get_sys_info():
    if not PSUTIL_AVAILABLE:
        return "⚠️ Chưa cài psutil"
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    return f"🖥️ CPU: {cpu}% | 🧠 RAM: {ram}%"

JS_COPY_RICH = """
(text, html) => {
    if (!text || text.trim().length === 0) return;
    try {
        const typeHtml = "text/html";
        const typeText = "text/plain";
        const blobHtml = new Blob([html], { type: typeHtml });
        const blobText = new Blob([text], { type: typeText });
        const data = [new ClipboardItem({ [typeHtml]: blobHtml, [typeText]: blobText })];
        navigator.clipboard.write(data).then(() => {
            alert("📋 Đã copy bảng! Bạn có thể dán vào Word/Excel dưới định dạng bảng chuẩn.");
        });
    } catch (err) {
        navigator.clipboard.writeText(text).then(() => {
            alert("📋 Đã sao chép (Dạng văn bản).");
        });
    }
}
"""

JS_SCROLL = "(id) => { setTimeout(() => { const el = document.getElementById(id); if (el) el.scrollIntoView({behavior: 'smooth', block: 'center'}); }, 100); }"

import matplotlib.pyplot as plt
import seaborn as sns

def setup_scientific_plots():
    """Thiết lập cấu hình biểu đồ chuẩn khoa học (Times New Roman, 300 DPI)."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 300
    sns.set_theme(style="ticks", rc={"font.family": "serif", "font.serif": ["Times New Roman"]})
    sns.set_palette("colorblind")

def get_sys_info():
    """Lấy thông tin hệ thống cho Dashboard. Trả về dict {cpu, ram}."""
    try:
        import psutil
        return {"cpu": psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}
    except ImportError:
        return {"cpu": 0, "ram": 0}

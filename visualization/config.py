import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# ========== Academic-quality figure styling ==========
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.2,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
})

# Color palette for academic figures
COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#E94F37',    # Vermillion red
    'tertiary': '#44AF69',     # Green
    'quaternary': '#F18F01',   # Orange
    'gray': '#6C757D',
}

# Palette from https://mycolor.space/?hex=%232E86AB&sub=1
# Using "Generic Gradient" scheme for smooth color transitions
GENERIC_GRADIENT = ['#2E86AB', '#00A3B9', '#00BFB4', '#57D89F', '#A7EC84', '#F9F871']

# Using "Classy Palette" for sophisticated color combinations
CLASSY_PALETTE = ['#2E86AB', '#364954', '#99AEBB', '#A06D95', '#6C3C62']

# Using "Threedom" for distinct triadic colors
THREEDOM = ['#2E86AB', '#C15D7A', '#638830']

# Using "Pin Palette" for vibrant combinations
PIN_PALETTE = ['#2E86AB', '#00C0FF', '#E6F4F1', '#EF972C']

# Combined rich palette from multiple mycolor.space schemes
BASE_PALETTE = [
    '#2E86AB',  # Base color (Steel blue)
    '#00A3B9',  # Generic Gradient 2
    '#57D89F',  # Generic Gradient 4
    '#A7EC84',  # Generic Gradient 5
    '#C15D7A',  # Threedom 2
    '#638830',  # Threedom 3
    '#EF972C',  # Pin Palette 4
    '#A06D95',  # Classy Palette 4
    '#6C3C62',  # Classy Palette 5
    '#00C0FF',  # Pin Palette 2
    '#99AEBB',  # Classy Palette 3
    '#F9F871',  # Generic Gradient 6
]


def get_palette(n: int):
    """Get color palette from mycolor.space schemes."""
    if n <= len(BASE_PALETTE):
        return BASE_PALETTE[:n]
    return [BASE_PALETTE[i % len(BASE_PALETTE)] for i in range(n)]


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiment_data"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_figure(fig, filename: str):
    """统一保存路径到集中文件夹 figures/comprehensive."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved: {out_path.relative_to(ROOT)}")
    return out_path


# ========== 拟合得到的参数 ==========
P0 = 0.199            # W, 基线功耗 (屏幕关)
P_s0 = 0.302          # W, 屏幕固定功耗 (亮度0)
alpha_s = 0.505       # W, 屏幕亮度斜率
alpha_c = 1.628       # W @ 100% CPU
alpha_gpu = 2.768     # W @ 100% GPU
alpha_w = 0.049       # W, WiFi
alpha_m = 0.098       # W, 蜂窝
alpha_g = 0.073       # W, GPS

Q_max_Ah = 6.0        # 手机电池标称容量 (Ah)
V_nom = 3.85          # 标称电压 (V)

# ========== ECM 参数（一阶 Thevenin） ==========
ECM = {
    'R0': 0.045,       # 欧姆内阻 (Ohm)
    'Rct': 0.09,       # 电荷转移电阻 (Ohm)
    'Cp': 2400.0,      # 极化电容 (F)
    'V_cut': 3.0,      # 截止电压 (V)
    'eta_min': 0.88,   # 最低效率
    'alpha_eta': 0.020,  # 效率 vs |I| 斜率
    'beta_eta': 0.004,   # 效率 vs |ΔT| 斜率
    'T_ref': 25.0,
}

# ========== OCV 五参数拟合系数 (Oxford pseudo-OCV) ==========
OCV_COEF = [4.115648, -0.646723, 0.541718, 0.250783, -0.046021]  # a0~a4

# ========== 温度/老化参数 ==========
K_T = 2.249e-4        # 温度系数 /°C²
Q_25 = 2.968          # 25°C 参考容量 (Ah) - 用于温度修正演示
LAMBDA_NORMAL = 1.85e-4   # 正常老化 λ /cycle
BETA_NORMAL = 1.0         # 正常老化 β
LAMBDA_ACCEL = 3.64e-3    # 加速老化 λ
BETA_ACCEL = 0.97         # 加速老化 β


def ocv_from_soc(soc: float) -> float:
    """OCV 五参数模型：OCV(s) = a0 + a1*s + a2*s² + a3*ln(s) + a4*ln(1-s)"""
    s = np.clip(soc, 1e-4, 1 - 1e-4)  # SOC 裁剪，避免 log(0)
    a = OCV_COEF
    return a[0] + a[1] * s + a[2] * s**2 + a[3] * np.log(s) + a[4] * np.log(1 - s)


def compute_efficiency(I: float, T: float, ecm: dict = ECM) -> float:
    """效率约束：η = max(η_min, 1 - α_η|I| - β_η|T - T_ref|)"""
    eta = 1 - ecm['alpha_eta'] * abs(I) - ecm['beta_eta'] * abs(T - ecm['T_ref'])
    return max(ecm['eta_min'], eta)


def solve_current_from_power(P: float, V_oc: float, v_p: float, R0: float) -> float:
    """
    求解二次方程 P = I * (V_oc - v_p - R0 * I) 得到电流 I
    判别式保护：若 P > (V_oc - v_p)² / (4*R0)，则限制到最大可行功率
    返回较小的正根（放电情况）
    """
    V_eff = V_oc - v_p
    P_max = V_eff**2 / (4 * R0)
    
    if P > P_max:
        P = P_max  # 功率限幅
    
    # 二次方程: R0*I² - V_eff*I + P = 0
    disc = V_eff**2 - 4 * R0 * P
    if disc < 0:
        disc = 0  # 数值保护
    
    sqrt_disc = np.sqrt(disc)
    I1 = (V_eff - sqrt_disc) / (2 * R0)  # 较小根
    I2 = (V_eff + sqrt_disc) / (2 * R0)  # 较大根
    
    # 放电时选较小正根（更稳定工作点）
    return I1 if I1 > 0 else I2


def temperature_capacity_factor(T: float, k_T: float = K_T, T_ref: float = 25.0) -> float:
    """温度容量修正因子：1 - k_T * (T - T_ref)²"""
    return 1 - k_T * (T - T_ref)**2


def aging_capacity_factor(n_cycles: int, lam: float = LAMBDA_NORMAL, beta: float = BETA_NORMAL) -> float:
    """老化容量修正因子（非线性）：1 - λ * n^β"""
    if n_cycles <= 0:
        return 1.0
    return max(0.0, 1 - lam * (n_cycles ** beta))


def predict_power(row: pd.Series):
    """根据一行数据预测瞬时功率 (W) - 基础功耗模型"""
    u_s = 1 if row.get('screen', 'off') == 'on' else 0
    brightness = row.get('brightness', 0)
    B = brightness / 255.0 if brightness else 0
    P_screen = u_s * (P_s0 + alpha_s * B)

    cpu_util = row.get('cpu_util_pct', 0)
    L_cpu = cpu_util / 100.0 if cpu_util else 0
    P_cpu = alpha_c * L_cpu

    gpu_util = row.get('gpu_util_pct', 0)
    L_gpu = gpu_util / 100.0 if gpu_util else 0
    P_gpu = alpha_gpu * L_gpu

    u_w = 1 if row.get('wifi_state', 'off') == 'on' else 0
    u_m = 1 if row.get('mobile_state', 'off') == 'on' else 0
    P_net = alpha_w * u_w + alpha_m * u_m

    u_g = 1 if row.get('gps', 'off') == 'on' else 0
    P_gps = alpha_g * u_g

    return P0 + P_screen + P_cpu + P_gpu + P_net + P_gps


def simulate_discharge_ecm(df: pd.DataFrame, n_cycles: int = 0, use_aging: bool = True):
    """
    ECM 增强放电仿真：
    - OCV(SOC) 非线性映射
    - 一阶 Thevenin 极化电压动态
    - 效率约束
    - 功率判别式保护
    - TTE 事件检测 (V <= V_cut)
    
    返回: (q_pred, v_pred, vp_trace, eta_trace, tte_idx)
    """
    df = df.sort_values('elapsed_sec').reset_index(drop=True)
    t = df['elapsed_sec'].values
    n = len(df)
    
    ecm = ECM
    R0, Rct, Cp, V_cut = ecm['R0'], ecm['Rct'], ecm['Cp'], ecm['V_cut']
    
    # 初始状态
    q0_mAh = df['charge_mAh'].iloc[0]
    Q_max_mAh = Q_max_Ah * 1000
    
    # 老化修正
    if use_aging and n_cycles > 0:
        aging_factor = aging_capacity_factor(n_cycles)
        Q_eff_mAh = Q_max_mAh * aging_factor
    else:
        Q_eff_mAh = Q_max_mAh
    
    # 温度修正（取平均温度）
    T_avg = df['temp_C'].mean() if 'temp_C' in df.columns else 25.0
    temp_factor = temperature_capacity_factor(T_avg)
    Q_eff_mAh *= temp_factor
    
    # 状态数组
    q_pred = np.zeros(n)
    v_pred = np.zeros(n)
    vp_trace = np.zeros(n)
    eta_trace = np.zeros(n)
    
    q_pred[0] = q0_mAh
    soc = q_pred[0] / Q_eff_mAh
    V_oc = ocv_from_soc(soc)
    v_p = 0.0  # 初始极化电压
    v_pred[0] = V_oc - v_p
    vp_trace[0] = v_p
    eta_trace[0] = 1.0
    
    tte_idx = None  # TTE 事件索引
    
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        if dt <= 0:
            dt = 1.0
        
        # 功率需求
        P_load = predict_power(df.iloc[i - 1])
        
        # 当前 SOC 和 OCV
        soc = np.clip(q_pred[i - 1] / Q_eff_mAh, 1e-4, 1 - 1e-4)
        V_oc = ocv_from_soc(soc)
        
        # 求解电流（带判别式保护）
        I = solve_current_from_power(P_load, V_oc, v_p, R0)
        
        # 效率约束
        T_now = df['temp_C'].iloc[i - 1] if 'temp_C' in df.columns else 25.0
        eta = compute_efficiency(I, T_now)
        
        # 有效电流（考虑效率损耗）
        I_eff = I / eta
        
        # 极化电压动态更新 (一阶 RC)
        dv_p = (-v_p / (Rct * Cp) + I_eff / Cp) * dt
        v_p = v_p + dv_p
        
        # 端电压
        V_t = V_oc - R0 * I_eff - v_p
        
        # 容量消耗 (mAh)
        dq = I_eff * dt / 3.6
        q_pred[i] = q_pred[i - 1] - dq
        
        v_pred[i] = V_t
        vp_trace[i] = v_p
        eta_trace[i] = eta
        
        # TTE 事件检测
        if V_t <= V_cut and tte_idx is None:
            tte_idx = i
    
    return q_pred, v_pred, vp_trace, eta_trace, tte_idx


def simulate_discharge(df: pd.DataFrame):
    """兼容接口：调用 ECM 增强仿真，仅返回电量预测"""
    q_pred, _, _, _, _ = simulate_discharge_ecm(df, n_cycles=0, use_aging=False)
    return q_pred


def _safe_mode(series: Optional[pd.Series], default='off'):
    if series is None or series.dropna().empty:
        return default
    return series.mode().iloc[0]


def compute_scene_power(df: pd.DataFrame):
    t = df['elapsed_sec'].values
    q = df['charge_mAh'].values
    slope = np.polyfit(t, q, 1)[0]
    current_A = -slope * 3.6
    voltage_V = df['voltage_mV'].mean() / 1000
    return current_A * voltage_V


def estimate_scene_model_power(df: pd.DataFrame):
    brightness = df['brightness'].mean() if 'brightness' in df else 0
    cpu_util = df['cpu_util_pct'].mean() if 'cpu_util_pct' in df else 0
    gpu_util = df['gpu_util_pct'].mean() if 'gpu_util_pct' in df else 0
    screen_state = _safe_mode(df.get('screen', pd.Series(dtype=str)), default='off')
    wifi_state = _safe_mode(df.get('wifi_state', pd.Series(dtype=str)), default='off')
    mobile_state = _safe_mode(df.get('mobile_state', pd.Series(dtype=str)), default='off')
    gps_state = _safe_mode(df.get('gps', pd.Series(dtype=str)), default='off')

    u_s = 1 if screen_state == 'on' else 0
    u_w = 1 if wifi_state == 'on' else 0
    u_m = 1 if mobile_state == 'on' else 0
    u_g = 1 if gps_state == 'on' else 0

    B = brightness / 255.0 if brightness else 0
    L_cpu = cpu_util / 100.0 if cpu_util else 0
    L_gpu = gpu_util / 100.0 if gpu_util else 0

    P_screen = u_s * (P_s0 + alpha_s * B)
    P_cpu = alpha_c * L_cpu
    P_gpu = alpha_gpu * L_gpu
    P_net = alpha_w * u_w + alpha_m * u_m
    P_gps = alpha_g * u_g
    return P0 + P_screen + P_cpu + P_gpu + P_net + P_gps


def _format_scene_label(scene_name: str) -> str:
    label = scene_name.replace('scene_', '').replace('_', ' ')
    label = label.replace('pct', '%')
    return label.title()


def _extract_scene_settings(df: pd.DataFrame):
    brightness = df['brightness'].mean() if 'brightness' in df else 0
    cpu_util = df['cpu_util_pct'].mean() if 'cpu_util_pct' in df else 0
    gpu_util = df['gpu_util_pct'].mean() if 'gpu_util_pct' in df else 0
    screen_state = _safe_mode(df.get('screen', pd.Series(dtype=str)), default='off')
    wifi_state = _safe_mode(df.get('wifi_state', pd.Series(dtype=str)), default='off')
    mobile_state = _safe_mode(df.get('mobile_state', pd.Series(dtype=str)), default='off')
    gps_state = _safe_mode(df.get('gps', pd.Series(dtype=str)), default='off')
    temp_mean = df['temp_C'].mean() if 'temp_C' in df else np.nan

    return {
        'u_s': 1 if screen_state == 'on' else 0,
        'B': brightness / 255.0 if brightness else 0,
        'L_cpu': cpu_util / 100.0 if cpu_util else 0,
        'L_gpu': gpu_util / 100.0 if gpu_util else 0,
        'u_w': 1 if wifi_state == 'on' else 0,
        'u_m': 1 if mobile_state == 'on' else 0,
        'u_g': 1 if gps_state == 'on' else 0,
        'brightness_raw': brightness,
        'cpu_raw': cpu_util,
        'gpu_raw': gpu_util,
        'temp_C': temp_mean,
    }


__all__ = [
    'plt', 'sns', 'np', 'pd',
    'COLORS', 'BASE_PALETTE', 'GENERIC_GRADIENT', 'CLASSY_PALETTE', 'THREEDOM', 'PIN_PALETTE',
    'get_palette', 'ROOT', 'DATA', 'OUTPUT_DIR', '_save_figure',
    'P0', 'P_s0', 'alpha_s', 'alpha_c', 'alpha_gpu', 'alpha_w', 'alpha_m', 'alpha_g',
    'Q_max_Ah', 'V_nom',
    # ECM 参数
    'ECM', 'OCV_COEF', 'K_T', 'Q_25', 'LAMBDA_NORMAL', 'BETA_NORMAL', 'LAMBDA_ACCEL', 'BETA_ACCEL',
    # ECM 函数
    'ocv_from_soc', 'compute_efficiency', 'solve_current_from_power',
    'temperature_capacity_factor', 'aging_capacity_factor',
    'predict_power', 'simulate_discharge', 'simulate_discharge_ecm',
    '_safe_mode', 'compute_scene_power', 'estimate_scene_model_power',
    '_format_scene_label', '_extract_scene_settings'
]

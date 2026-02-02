import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from .config import COLORS, _save_figure, get_palette

# ========== 温度/老化参数（来自 拟合过程.tex） ==========
# Wisconsin MultiTemp 数据集（Panasonic 18650PF）
WISCONSIN_TEMP_DATA = {
    'temp_C': [-20, -10, 0, 10, 25],
    'capacity_Ah': [1.629, 2.334, 2.727, 2.886, 2.968],
}
K_T = 2.249e-4  # /°C^2, 温度系数
Q_25 = 2.968    # Ah, 25°C 参考容量

# NASA 老化数据（加速条件，非线性拟合）
NASA_AGING_DATA = {
    'battery': ['B0005', 'B0006', 'B0007', 'B0018'],
    'lambda': [8.14e-4, 7.05e-3, 9.59e-4, 5.72e-3],
    'beta': [1.17, 0.80, 1.11, 0.80],
    'cycles': [168, 168, 168, 132],
    'capacity_fade_pct': [28.6, 41.7, 24.3, 27.7],
}
LAMBDA_NASA_AVG = 3.64e-3  # 平均 λ
BETA_NASA_AVG = 0.97       # 平均 β

# Toyota/Stanford 正常循环数据（近似线性 β≈1）
LAMBDA_NORMAL = 1.85e-4  # /cycle
BETA_NORMAL = 1.0

# ========== OCV-SOC 参数（Oxford pseudo-OCV 拟合） ==========
OCV_COEF = [4.115648, -0.646723, 0.541718, 0.250783, -0.046021]  # a0~a4


def plot_temperature_capacity_curve():
    """Plot temperature vs. capacity curve with Wisconsin data and model fit."""
    temps = np.array(WISCONSIN_TEMP_DATA['temp_C'])
    caps = np.array(WISCONSIN_TEMP_DATA['capacity_Ah'])
    q_ratio = caps / Q_25

    T_model = np.linspace(-25, 30, 200)
    Q_ratio_model = 1 - K_T * (T_model - 25)**2

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1 = axes[0]
    ax1.scatter(temps, caps, s=100, color=COLORS['primary'], edgecolor='black',
                linewidth=1.2, zorder=5, label='Wisconsin data')
    ax1.plot(T_model, Q_ratio_model * Q_25, color=COLORS['secondary'], lw=2.5,
             label=rf'Model: $Q = Q_{{25}}[1 - k_T(T-25)^2]$')
    ax1.axhline(Q_25, color='gray', ls='--', lw=1.0, alpha=0.7, label=f'$Q_{{25}}$ = {Q_25:.3f} Ah')
    ax1.axvline(25, color='gray', ls=':', lw=1.0, alpha=0.7)
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Discharge Capacity (Ah)', fontsize=12)
    ax1.set_title('(a) Temperature–Capacity Relationship', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, frameon=True)
    ax1.set_xlim(-28, 32)
    ax1.set_ylim(1.4, 3.2)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax1, offset=4)

    ax2 = axes[1]
    ax2.scatter(temps, q_ratio, s=100, color=COLORS['tertiary'], edgecolor='black',
                linewidth=1.2, zorder=5, label='Data (normalized)')
    ax2.plot(T_model, Q_ratio_model, color=COLORS['secondary'], lw=2.5,
             label=rf'Model: $k_T = {K_T:.3e}$ /°C²')
    ax2.axhline(1.0, color='gray', ls='--', lw=1.0, alpha=0.7)
    ax2.axvline(25, color='gray', ls=':', lw=1.0, alpha=0.7)
    for t, qr in zip(temps, q_ratio):
        ax2.annotate(f'{qr:.3f}', (t, qr), textcoords='offset points',
                     xytext=(5, 8), fontsize=9, color='black')
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel(r'$Q(T) / Q_{25}$', fontsize=12)
    ax2.set_title('(b) Normalized Capacity & Fitted Model', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, frameon=True)
    ax2.set_xlim(-28, 32)
    ax2.set_ylim(0.45, 1.1)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.35)
    ax2.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax2, offset=4)

    fig.suptitle(f'Temperature Coefficient Fit (Wisconsin MultiTemp, $k_T = {K_T:.3e}$ /°C²)',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, 'temperature_capacity_fit.png')


def plot_aging_cycle_curve():
    """Plot aging curves: NASA (accelerated) and Toyota (normal) data."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1 = axes[0]
    nasa_bats = NASA_AGING_DATA['battery']
    nasa_lambda = np.array(NASA_AGING_DATA['lambda'])
    nasa_beta = np.array(NASA_AGING_DATA['beta'])
    palette_nasa = get_palette(len(nasa_bats))
    x_pos = np.arange(len(nasa_bats))
    width = 0.38
    bars_lam = ax1.bar(x_pos - width/2, nasa_lambda * 1000, width, color=palette_nasa, edgecolor='black', linewidth=1.0, label=r'$\lambda$ (×10⁻³)')
    bars_beta = ax1.bar(x_pos + width/2, nasa_beta, width, color=[COLORS['quaternary']]*len(nasa_bats), edgecolor='black', linewidth=1.0, label=r'$\beta$')
    for bar, val in zip(bars_lam, nasa_lambda * 1000):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='medium')
    for bar, val in zip(bars_beta, nasa_beta):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.03,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='medium')
    ax1.axhline(LAMBDA_NASA_AVG * 1000, color=COLORS['secondary'], ls='--', lw=2.0,
                label=f'Avg λ = {LAMBDA_NASA_AVG*1000:.2f}')
    ax1.axhline(BETA_NASA_AVG, color=COLORS['gray'], ls=':', lw=1.5,
                label=f'Avg β = {BETA_NASA_AVG:.2f}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(nasa_bats)
    ax1.set_xlabel('Battery ID', fontsize=12)
    ax1.set_ylabel(r'$\lambda$ (×10⁻³) / $\beta$', fontsize=12)
    ax1.set_title('(a) NASA Nonlinear Aging (λ, β)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, frameon=True)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(axis='y', which='major', alpha=0.35)
    ax1.grid(axis='y', which='minor', alpha=0.12)
    sns.despine(ax=ax1, offset=4)

    ax2 = axes[1]
    cycles = np.linspace(1, 600, 200)  # start at 1 to avoid 0^beta

    soh_normal = 1 - LAMBDA_NORMAL * np.power(cycles, BETA_NORMAL)
    ax2.plot(cycles, soh_normal * 100, color=COLORS['tertiary'], lw=2.5,
             label=rf'Normal: $\lambda$={LAMBDA_NORMAL:.2e}, $\beta$={BETA_NORMAL:.1f}')

    soh_accel = 1 - LAMBDA_NASA_AVG * np.power(cycles, BETA_NASA_AVG)
    soh_accel = np.clip(soh_accel, 0, 1)  # prevent negative
    ax2.plot(cycles, soh_accel * 100, color=COLORS['secondary'], lw=2.5, ls='--',
             label=rf'Accel (NASA): $\lambda$={LAMBDA_NASA_AVG:.2e}, $\beta$={BETA_NASA_AVG:.2f}')

    ax2.axhline(80, color='gray', ls=':', lw=1.5, alpha=0.8, label='EOL threshold (80%)')
    ax2.axhline(100, color='gray', ls='--', lw=1.0, alpha=0.5)

    fade_500_normal = LAMBDA_NORMAL * (500 ** BETA_NORMAL) * 100
    ax2.annotate(f'500 cycles: {100 - fade_500_normal:.1f}%',
                 (500, 100 - fade_500_normal), textcoords='offset points',
                 xytext=(-60, -20), fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    ax2.set_xlabel('Cycle Number', fontsize=12)
    ax2.set_ylabel('State of Health (%)', fontsize=12)
    ax2.set_title('(b) Aging Model Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=9, frameon=True)
    ax2.set_xlim(0, 600)
    ax2.set_ylim(0, 105)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.35)
    ax2.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax2, offset=4)

    fig.suptitle('Aging Coefficient Fitting (NASA + Toyota Data Sources)',
                 fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, 'aging_cycle_fit.png')


def plot_combined_capacity_model():
    """更专业的组合容量模型可视化：曲线 + 等高线 + 寿命标注。"""
    temps = np.array([-20, -10, 0, 10, 25])
    cycles = np.linspace(0, 600, 240)

    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.24)
    ax_curve = fig.add_subplot(gs[0, 0])
    ax_contour = fig.add_subplot(gs[0, 1])

    palette = sns.color_palette('coolwarm', n_colors=len(temps))

    # 主曲线面板：不同温度下的 SOH 随循环衰减 + EOL 提前量（非线性）
    eol_cycle_points = []
    for temp, color in zip(temps, palette):
        temp_factor = 1 - K_T * (temp - 25)**2
        soh = temp_factor * (1 - LAMBDA_NORMAL * np.power(cycles, BETA_NORMAL))
        label = f'T = {temp}°C'
        ax_curve.plot(cycles, soh * 100, color=color, lw=2.2, label=label)

        # 计算达到 80% SOH 的循环数（若初始低于 80%，则记为 0）
        # 解 temp_factor*(1-λ n^β)=0.8 => n=(1-0.8/temp_factor)/λ)^(1/β)
        if temp_factor > 0.8:
            eol_n = ((1 - 0.8 / temp_factor) / LAMBDA_NORMAL) ** (1 / BETA_NORMAL)
        else:
            eol_n = 0
        if eol_n > 0:
            eol_n = min(eol_n, cycles.max())
            ax_curve.scatter(eol_n, 80, color=color, edgecolor='black', zorder=6, s=45)
            eol_cycle_points.append((temp, eol_n))
            ax_curve.annotate(f'{eol_n:.0f} cyc',
                              (eol_n, 80), xytext=(8, 10), textcoords='offset points',
                              fontsize=9, color=color, weight='bold',
                              arrowprops=dict(arrowstyle='->', color=color, lw=1.1))

        # 高亮“可用区间” (>=80%)
        valid_mask = soh * 100 >= 80
        if valid_mask.any():
            ax_curve.fill_between(cycles, 80, soh * 100, where=valid_mask,
                                   color=color, alpha=0.08, linewidth=0)

    ax_curve.axhline(80, color='gray', ls=':', lw=1.5, alpha=0.9, label='EOL 80%')
    ax_curve.set_xlim(0, cycles.max())
    ax_curve.set_ylim(40, 105)
    ax_curve.set_xlabel('Cycle Number', fontsize=12)
    ax_curve.set_ylabel('Effective State of Health (%)', fontsize=12)
    ax_curve.set_title(rf'Combined Capacity Model: $Q(T,n)/Q_{{\max,0}} = [1 - k_T(T-25)^2](1 - \lambda n^{{\beta}})$',
                       fontsize=13, fontweight='bold', pad=10)
    ax_curve.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_curve.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_curve.grid(True, which='major', alpha=0.32)
    ax_curve.grid(True, which='minor', alpha=0.12)
    ax_curve.legend(loc='lower left', fontsize=9, frameon=True, ncol=2)

    # 次坐标轴：换算回容量 (Ah)
    def soh_to_ah(y):
        return y / 100 * Q_25

    def ah_to_soh(y):
        return y / Q_25 * 100

    ax_ah = ax_curve.twinx()
    ax_ah.set_ylim(ax_curve.get_ylim())
    ax_ah.set_ylabel('Capacity (Ah)', fontsize=11)
    yticks = ax_curve.get_yticks()
    ax_ah.set_yticks(yticks)
    ax_ah.set_yticklabels([f'{soh_to_ah(y):.2f}' for y in yticks])
    sns.despine(ax=ax_curve, right=False, offset=4)

    # 右侧等高线/热力图：温度-循环二维衰减图（非线性）
    temp_grid = np.linspace(-25, 40, 140)
    cycle_grid = np.linspace(1, 600, 240)  # start at 1
    Tm, Nm = np.meshgrid(temp_grid, cycle_grid)
    soh_grid = (1 - K_T * (Tm - 25)**2) * (1 - LAMBDA_NORMAL * np.power(Nm, BETA_NORMAL)) * 100

    cmap = sns.color_palette('coolwarm', as_cmap=True)
    cf = ax_contour.contourf(Tm, Nm, soh_grid, levels=np.linspace(40, 105, 18), cmap=cmap, alpha=0.92)
    cs = ax_contour.contour(Tm, Nm, soh_grid, levels=[60, 70, 80, 90], colors='k', linewidths=1.0, linestyles='--')
    ax_contour.clabel(cs, fmt='%d%%', fontsize=9)
    cbar = fig.colorbar(cf, ax=ax_contour, pad=0.02, aspect=28)
    cbar.set_label('SOH (%)', fontsize=11)

    ax_contour.set_xlabel('Temperature (°C)', fontsize=12)
    ax_contour.set_ylabel('Cycle Number', fontsize=12)
    ax_contour.set_title('Temperature × Cycle Impact (contour view)', fontsize=12, fontweight='bold', pad=8)
    ax_contour.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_contour.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_contour.grid(True, which='major', alpha=0.28)
    ax_contour.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax_contour, offset=4)

    # 文本注释框
    textstr = (f'$k_T = {K_T:.3e}$ /°C²\n'
               f'$\\lambda = {LAMBDA_NORMAL:.2e}$, $\\beta = {BETA_NORMAL:.1f}$\n'
               f'$Q_{{25}} = {Q_25:.3f}$ Ah')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85)
    ax_curve.text(0.98, 0.97, textstr, transform=ax_curve.transAxes, fontsize=10,
                  verticalalignment='top', horizontalalignment='right', bbox=props)

    # 汇总 EOL 循环表
    if eol_cycle_points:
        lines = ["EOL @80% cycles:"] + [f"T={t:+.0f}°C: {n:.0f}" for t, n in eol_cycle_points]
        ax_curve.text(0.02, 0.98, "\n".join(lines), transform=ax_curve.transAxes,
                      va='top', ha='left', fontsize=9.2,
                      bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.8', alpha=0.9))

    # 在等高线图标注参考温度 25°C 以及峰值区域
    ax_contour.axvline(25, color='k', ls=':', lw=1.2, alpha=0.8)
    ax_contour.scatter([25], [0], color='k', s=26, zorder=5, marker='o', edgecolor='white', linewidth=0.8)
    ax_contour.annotate('Ref temp 25°C', xy=(25, 0), xytext=(5, 30), textcoords='offset points',
                        fontsize=9, arrowprops=dict(arrowstyle='->', color='k', lw=1.0))

    # 更紧凑的版面，避免 tight_layout 警告
    fig.subplots_adjust(left=0.07, right=0.96, top=0.95, bottom=0.08, wspace=0.22)
    _save_figure(fig, 'combined_capacity_model.png')


def plot_ocv_soc_curve():
    """Plot fitted OCV-SOC curve using Oxford pseudo-OCV coefficients."""
    a = np.array(OCV_COEF)
    soc = np.linspace(0.01, 0.99, 300)
    ocv = a[0] + a[1] * soc + a[2] * soc**2 + a[3] * np.log(soc) + a[4] * np.log(1 - soc)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(soc, ocv, color=COLORS['primary'], lw=2.5, label='Model fit')
    ax.set_xlabel('SOC', fontsize=12)
    ax.set_ylabel('OCV (V)', fontsize=12)
    ax.set_title('OCV–SOC Curve (Oxford pseudo-OCV)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(2.6, 4.4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which='major', alpha=0.35)
    ax.grid(True, which='minor', alpha=0.12)

    # 公式文本
    formula = (r'$\mathrm{OCV}(s) = a_0 + a_1 s + a_2 s^2 + a_3 \ln s + a_4 \ln(1-s)$' + '\n'
               + f'$[a_0..a_4] = [{a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}, {a[3]:.2f}, {a[4]:.2f}]$')
    ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.legend(loc='lower right', fontsize=10)
    sns.despine(ax=ax, offset=4)
    fig.tight_layout()
    _save_figure(fig, 'ocv_soc_curve.png')


__all__ = [
    'plot_temperature_capacity_curve',
    'plot_aging_cycle_curve',
    'plot_combined_capacity_model',
    'plot_ocv_soc_curve',
]

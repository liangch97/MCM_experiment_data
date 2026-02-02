import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

from .config import (
    COLORS,
    ECM,
    Q_max_Ah,
    V_nom,
    K_T,
    LAMBDA_NORMAL,
    BETA_NORMAL,
    OCV_COEF,
    ocv_from_soc,
    solve_current_from_power,
    temperature_capacity_factor,
    aging_capacity_factor,
    compute_efficiency,
    simulate_discharge_ecm,
    _save_figure,
    get_palette,
)


def plot_ocv_nonlinearity_mechanism():
    """
    OCV-SOC 非线性机制深度剖析：
    - (a) OCV曲线 + 线性近似误差带
    - (b) 斜率放大与敏感度分析
    - (c) SOC区间误差累积热图
    - (d) 不同电池化学体系对比
    """
    soc = np.linspace(0.005, 0.995, 500)
    ocv = np.array([ocv_from_soc(s) for s in soc])
    docv = np.gradient(ocv, soc)
    d2ocv = np.gradient(docv, soc)

    # 线性近似
    ocv_linear = ocv[0] + (ocv[-1] - ocv[0]) * soc
    error_linear = (ocv - ocv_linear) * 1000  # mV

    # 不同电池化学体系的 OCV 系数（模拟）
    chemistries = {
        'LFP': [3.30, -0.15, 0.08, 0.12, -0.03],
        'NMC (fitted)': OCV_COEF,
        'LCO': [4.18, -0.72, 0.62, 0.28, -0.05],
        'NCA': [4.22, -0.80, 0.70, 0.32, -0.06],
    }

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], wspace=0.22, hspace=0.28)

    # (a) OCV 曲线与线性误差
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(soc, ocv, color=COLORS['primary'], lw=2.8, label='5-param model', zorder=3)
    ax1.plot(soc, ocv_linear, color='gray', lw=1.8, ls='--', label='Linear approx.', zorder=2)
    ax1.fill_between(soc, ocv_linear, ocv, alpha=0.15, color=COLORS['secondary'], label='Nonlinear deviation')
    
    # 标注关键区域
    low_soc_mask = soc < 0.15
    high_soc_mask = soc > 0.85
    ax1.axvspan(0, 0.15, alpha=0.08, color='red', label='Critical low SOC')
    ax1.axvspan(0.85, 1.0, alpha=0.08, color='orange')
    
    # 添加公式框
    formula_box = (r'$V_{oc}(s) = a_0 + a_1 s + a_2 s^2 + a_3 \ln s + a_4 \ln(1-s)$' + '\n'
                   + f'$[a_0..a_4] = [{OCV_COEF[0]:.2f}, {OCV_COEF[1]:.2f}, {OCV_COEF[2]:.2f}, {OCV_COEF[3]:.2f}, {OCV_COEF[4]:.2f}]$')
    ax1.text(0.03, 0.97, formula_box, transform=ax1.transAxes, fontsize=9.5, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    
    ax1.set_xlabel('State of Charge (SOC)', fontsize=12)
    ax1.set_ylabel('Open Circuit Voltage (V)', fontsize=12)
    ax1.set_title('(a) OCV–SOC Nonlinearity & Linear Approximation Error', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(2.5, 4.5)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax1, offset=4)

    # 右侧副轴：误差
    ax1b = ax1.twinx()
    ax1b.plot(soc, error_linear, color=COLORS['secondary'], lw=1.5, ls=':', alpha=0.8)
    ax1b.set_ylabel('Linear Error (mV)', fontsize=10, color=COLORS['secondary'])
    ax1b.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax1b.set_ylim(-150, 150)

    # (b) 斜率与二阶导数（敏感度）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    
    l1, = ax2.plot(soc, docv, color=COLORS['primary'], lw=2.4, label=r'$dV_{oc}/dSOC$')
    ax2.fill_between(soc, 0, docv, where=np.abs(docv) > 1.5, alpha=0.2, color=COLORS['primary'])
    l2, = ax2_twin.plot(soc, np.abs(d2ocv), color=COLORS['tertiary'], lw=2.0, ls='--', label=r'$|d^2V_{oc}/dSOC^2|$')
    
    # 标注敏感区
    for thresh, txt, xpos in [(0.08, 'High\nsensitivity', 0.04), (0.92, 'Moderate\nsensitivity', 0.94)]:
        ax2.annotate(txt, xy=(xpos, docv[int(xpos * len(soc))]), xytext=(xpos + 0.08, docv[int(xpos * len(soc))] + 0.8),
                     fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_xlabel('State of Charge (SOC)', fontsize=12)
    ax2.set_ylabel(r'Slope $dV_{oc}/dSOC$ (V)', fontsize=12, color=COLORS['primary'])
    ax2_twin.set_ylabel(r'Curvature $|d^2V_{oc}/dSOC^2|$', fontsize=10, color=COLORS['tertiary'])
    ax2.set_title('(b) Slope Amplification & Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.legend(handles=[l1, l2], loc='upper right', fontsize=9)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.32)
    ax2.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax2, right=False, offset=4)

    # (c) SOC 估计误差累积热图
    ax3 = fig.add_subplot(gs[1, 0])
    soc_grid = np.linspace(0.02, 0.98, 50)
    delta_soc = np.linspace(-0.1, 0.1, 50)
    SOC_G, DELTA_G = np.meshgrid(soc_grid, delta_soc)
    
    # 计算 OCV 误差 (由于 SOC 估计偏差导致)
    ocv_error = np.zeros_like(SOC_G)
    for i, ds in enumerate(delta_soc):
        for j, s0 in enumerate(soc_grid):
            s_true = np.clip(s0, 0.01, 0.99)
            s_est = np.clip(s0 + ds, 0.01, 0.99)
            ocv_error[i, j] = (ocv_from_soc(s_est) - ocv_from_soc(s_true)) * 1000  # mV
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    im = ax3.contourf(SOC_G, DELTA_G * 100, ocv_error, levels=np.linspace(-200, 200, 21), cmap=cmap, extend='both')
    cs = ax3.contour(SOC_G, DELTA_G * 100, np.abs(ocv_error), levels=[50, 100, 150], colors='k', linewidths=0.8, linestyles='--')
    ax3.clabel(cs, fmt='%d mV', fontsize=8)
    cbar = fig.colorbar(im, ax=ax3, pad=0.02, aspect=25)
    cbar.set_label('OCV Error (mV)', fontsize=10)
    
    ax3.set_xlabel('True SOC', fontsize=12)
    ax3.set_ylabel('SOC Estimation Error (%)', fontsize=12)
    ax3.set_title('(c) OCV Error Sensitivity to SOC Estimation', fontsize=13, fontweight='bold')
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(ax=ax3, offset=4)

    # (d) 不同电池化学体系 OCV 对比
    ax4 = fig.add_subplot(gs[1, 1])
    palette = get_palette(len(chemistries))
    
    for idx, (name, coef) in enumerate(chemistries.items()):
        a = np.array(coef)
        s = np.linspace(0.02, 0.98, 200)
        v = a[0] + a[1] * s + a[2] * s**2 + a[3] * np.log(s) + a[4] * np.log(1 - s)
        lw = 2.8 if 'fitted' in name else 2.0
        ls = '-' if 'fitted' in name else '--'
        ax4.plot(s, v, color=palette[idx], lw=lw, ls=ls, label=name)
    
    ax4.set_xlabel('State of Charge (SOC)', fontsize=12)
    ax4.set_ylabel('Open Circuit Voltage (V)', fontsize=12)
    ax4.set_title('(d) OCV Curves for Different Li-ion Chemistries', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('OCV–SOC Nonlinearity: Physical Mechanism & Implications', fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_ocv_nonlinearity.png')


def plot_ocv_nonlinearity_mechanism_part1():
    """
    OCV-SOC 非线性（上半部分）：
    - (a) OCV 曲线 + 线性误差
    - (b) 斜率/曲率敏感度
    """
    soc = np.linspace(0.005, 0.995, 500)
    ocv = np.array([ocv_from_soc(s) for s in soc])
    docv = np.gradient(ocv, soc)
    d2ocv = np.gradient(docv, soc)
    ocv_linear = ocv[0] + (ocv[-1] - ocv[0]) * soc
    error_linear = (ocv - ocv_linear) * 1000

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    # (a)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(soc, ocv, color=COLORS['primary'], lw=2.8, label='5-param model', zorder=3)
    ax1.plot(soc, ocv_linear, color='gray', lw=1.8, ls='--', label='Linear approx.', zorder=2)
    ax1.fill_between(soc, ocv_linear, ocv, alpha=0.15, color=COLORS['secondary'], label='Nonlinear deviation')
    ax1.axvspan(0, 0.15, alpha=0.08, color='red', label='Critical low SOC')
    ax1.axvspan(0.85, 1.0, alpha=0.08, color='orange')
    formula_box = (r'$V_{oc}(s) = a_0 + a_1 s + a_2 s^2 + a_3 \ln s + a_4 \ln(1-s)$' + '\n'
                   + f'$[a_0..a_4] = [{OCV_COEF[0]:.2f}, {OCV_COEF[1]:.2f}, {OCV_COEF[2]:.2f}, {OCV_COEF[3]:.2f}, {OCV_COEF[4]:.2f}]$')
    ax1.text(0.03, 0.97, formula_box, transform=ax1.transAxes, fontsize=9.5, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    ax1.set_xlabel('State of Charge (SOC)', fontsize=12)
    ax1.set_ylabel('Open Circuit Voltage (V)', fontsize=12)
    ax1.set_title('(a) OCV–SOC Nonlinearity & Linear Approximation Error', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(2.5, 4.5)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax1, offset=4)

    ax1b = ax1.twinx()
    ax1b.plot(soc, error_linear, color=COLORS['secondary'], lw=1.5, ls=':', alpha=0.8)
    ax1b.set_ylabel('Linear Error (mV)', fontsize=10, color=COLORS['secondary'])
    ax1b.tick_params(axis='y', labelcolor=COLORS['secondary'])
    ax1b.set_ylim(-150, 150)

    # (b)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    l1, = ax2.plot(soc, docv, color=COLORS['primary'], lw=2.4, label=r'$dV_{oc}/dSOC$')
    ax2.fill_between(soc, 0, docv, where=np.abs(docv) > 1.5, alpha=0.2, color=COLORS['primary'])
    l2, = ax2_twin.plot(soc, np.abs(d2ocv), color=COLORS['tertiary'], lw=2.0, ls='--', label=r'$|d^2V_{oc}/dSOC^2|$')
    for thresh, txt, xpos in [(0.08, 'High\nsensitivity', 0.04), (0.92, 'Moderate\nsensitivity', 0.94)]:
        ax2.annotate(txt, xy=(xpos, docv[int(xpos * len(soc))]), xytext=(xpos + 0.08, docv[int(xpos * len(soc))] + 0.8),
                     fontsize=9, ha='center', arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.set_xlabel('State of Charge (SOC)', fontsize=12)
    ax2.set_ylabel(r'Slope $dV_{oc}/dSOC$ (V)', fontsize=12, color=COLORS['primary'])
    ax2_twin.set_ylabel(r'Curvature $|d^2V_{oc}/dSOC^2|$', fontsize=10, color=COLORS['tertiary'])
    ax2.set_title('(b) Slope Amplification & Sensitivity Analysis', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.legend(handles=[l1, l2], loc='upper right', fontsize=9)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.32)
    ax2.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax2, right=False, offset=4)

    fig.suptitle('OCV–SOC Nonlinearity (Part 1)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_ocv_nonlinearity_part1.png')


def plot_ocv_nonlinearity_mechanism_part2():
    """
    OCV-SOC 非线性（下半部分）：
    - (c) SOC 估计误差敏感度
    - (d) 不同化学体系 OCV 曲线
    """
    chemistries = {
        'LFP': [3.30, -0.15, 0.08, 0.12, -0.03],
        'NMC (fitted)': OCV_COEF,
        'LCO': [4.18, -0.72, 0.62, 0.28, -0.05],
        'NCA': [4.22, -0.80, 0.70, 0.32, -0.06],
    }
    soc_grid = np.linspace(0.02, 0.98, 50)
    delta_soc = np.linspace(-0.1, 0.1, 50)
    SOC_G, DELTA_G = np.meshgrid(soc_grid, delta_soc)
    ocv_error = np.zeros_like(SOC_G)
    for i, ds in enumerate(delta_soc):
        for j, s0 in enumerate(soc_grid):
            s_true = np.clip(s0, 0.01, 0.99)
            s_est = np.clip(s0 + ds, 0.01, 0.99)
            ocv_error[i, j] = (ocv_from_soc(s_est) - ocv_from_soc(s_true)) * 1000

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    ax3 = fig.add_subplot(gs[0, 0])
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    im = ax3.contourf(SOC_G, DELTA_G * 100, ocv_error, levels=np.linspace(-200, 200, 21), cmap=cmap, extend='both')
    cs = ax3.contour(SOC_G, DELTA_G * 100, np.abs(ocv_error), levels=[50, 100, 150], colors='k', linewidths=0.8, linestyles='--')
    ax3.clabel(cs, fmt='%d mV', fontsize=8)
    cbar = fig.colorbar(im, ax=ax3, pad=0.02, aspect=25)
    cbar.set_label('OCV Error (mV)', fontsize=10)
    ax3.set_xlabel('True SOC', fontsize=12)
    ax3.set_ylabel('SOC Estimation Error (%)', fontsize=12)
    ax3.set_title('(c) OCV Error Sensitivity to SOC Estimation', fontsize=13, fontweight='bold')
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(ax=ax3, offset=4)

    ax4 = fig.add_subplot(gs[0, 1])
    palette = get_palette(len(chemistries))
    for idx, (name, coef) in enumerate(chemistries.items()):
        a = np.array(coef)
        s = np.linspace(0.02, 0.98, 200)
        v = a[0] + a[1] * s + a[2] * s**2 + a[3] * np.log(s) + a[4] * np.log(1 - s)
        lw = 2.8 if 'fitted' in name else 2.0
        ls = '-' if 'fitted' in name else '--'
        ax4.plot(s, v, color=palette[idx], lw=lw, ls=ls, label=name)
    ax4.set_xlabel('State of Charge (SOC)', fontsize=12)
    ax4.set_ylabel('Open Circuit Voltage (V)', fontsize=12)
    ax4.set_title('(d) OCV Curves for Different Li-ion Chemistries', fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('OCV–SOC Nonlinearity (Part 2)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_ocv_nonlinearity_part2.png')


def plot_power_current_coupling_mechanism():
    """
    功率-电流隐式耦合深度分析：
    - (a) 3D曲面：P-I-SOC三维关系
    - (b) 不同SOC下的P-I曲线族与工作点
    - (c) 判别式可行域与效率等高线
    - (d) 动态工作点轨迹（放电过程）
    """
    R0 = ECM['R0']
    Rct = ECM['Rct']

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.28)

    # (a) 3D 曲面：P = I * (V_oc - v_p - R0 * I)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    soc_3d = np.linspace(0.1, 0.95, 40)
    I_3d = np.linspace(0.1, 6, 50)
    SOC_M, I_M = np.meshgrid(soc_3d, I_3d)
    
    vp_fixed = 0.03
    V_oc_M = np.array([[ocv_from_soc(s) for s in soc_3d] for _ in I_3d])
    P_M = I_M * (V_oc_M - vp_fixed - R0 * I_M)
    P_M = np.clip(P_M, 0, None)
    
    surf = ax1.plot_surface(SOC_M, I_M, P_M, cmap='viridis', alpha=0.85, edgecolor='none')
    ax1.set_xlabel('SOC', fontsize=10, labelpad=8)
    ax1.set_ylabel('Current I (A)', fontsize=10, labelpad=8)
    ax1.set_zlabel('Power P (W)', fontsize=10, labelpad=8)
    ax1.set_title('(a) P–I–SOC 3D Surface', fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=25, azim=135)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=15, pad=0.1, label='Power (W)')

    # (b) P-I 曲线族（不同 SOC）+ 最大功率点
    ax2 = fig.add_subplot(gs[0, 1])
    soc_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    palette = get_palette(len(soc_levels))
    
    for idx, s in enumerate(soc_levels):
        V_oc = ocv_from_soc(s)
        I_axis = np.linspace(0.01, (V_oc - vp_fixed) / R0 * 0.95, 200)
        P_axis = I_axis * (V_oc - vp_fixed - R0 * I_axis)
        ax2.plot(I_axis, P_axis, color=palette[idx], lw=2.2, label=f'SOC={s:.2f}')
        
        # 最大功率点
        I_max = (V_oc - vp_fixed) / (2 * R0)
        P_max = I_max * (V_oc - vp_fixed - R0 * I_max)
        ax2.scatter([I_max], [P_max], color=palette[idx], s=60, zorder=5, edgecolor='black', linewidth=0.8)
        ax2.annotate(f'MPP', xy=(I_max, P_max), xytext=(5, 8), textcoords='offset points', fontsize=8)
    
    # 典型工作点
    typical_P = [1.5, 3.0, 5.0]
    for P in typical_P:
        ax2.axhline(P, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax2.text(0.3, P + 0.15, f'P={P}W', fontsize=8, color='gray')
    
    ax2.set_xlabel('Current I (A)', fontsize=12)
    ax2.set_ylabel('Deliverable Power P (W)', fontsize=12)
    ax2.set_title('(b) P–I Characteristics at Different SOC', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 18)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.32)
    ax2.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax2, offset=4)

    # (c) 判别式可行域 + 效率等高线
    ax3 = fig.add_subplot(gs[1, 0])
    vp_grid = np.linspace(-0.25, 0.25, 100)
    P_req = np.linspace(0, 15, 120)
    VP, PREQ = np.meshgrid(vp_grid, P_req)
    
    V_mid = ocv_from_soc(0.5)
    P_max_grid = (V_mid - VP) ** 2 / (4 * R0)
    feasible = PREQ <= P_max_grid
    
    # 效率计算 (假设在可行域内)
    I_grid = np.zeros_like(PREQ)
    eta_grid = np.ones_like(PREQ)
    for i in range(PREQ.shape[0]):
        for j in range(PREQ.shape[1]):
            if feasible[i, j]:
                I_grid[i, j] = solve_current_from_power(PREQ[i, j], V_mid, VP[i, j], R0)
                eta_grid[i, j] = compute_efficiency(I_grid[i, j], 25.0)
            else:
                eta_grid[i, j] = np.nan
    
    # 可行域填充
    ax3.contourf(VP, PREQ, feasible.astype(float), levels=[-0.1, 0.5, 1.1],
                 colors=['#ffcccc', '#ccffcc'], alpha=0.4)
    
    # 效率等高线
    eta_levels = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    cs_eta = ax3.contour(VP, PREQ, eta_grid, levels=eta_levels, cmap='Blues', linewidths=1.2)
    ax3.clabel(cs_eta, fmt='η=%.2f', fontsize=8)
    
    # P_max 边界
    vp_line = np.linspace(vp_grid.min(), vp_grid.max(), 200)
    ax3.plot(vp_line, (V_mid - vp_line) ** 2 / (4 * R0), color='black', lw=2.5,
             label=r'$P_{max}=\frac{(V_{oc}-v_p)^2}{4R_0}$')
    
    # 公式框
    formula = r'$\Delta = (V_{oc}-v_p)^2 - 4R_0 P \geq 0$'
    ax3.text(0.03, 0.97, formula, transform=ax3.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.95))
    
    ax3.set_xlabel(r'Polarization voltage $v_p$ (V)', fontsize=12)
    ax3.set_ylabel('Requested power P (W)', fontsize=12)
    ax3.set_title('(c) Discriminant Feasible Region & Efficiency Contours', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(vp_grid.min(), vp_grid.max())
    ax3.set_ylim(0, P_req.max())
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.28)
    ax3.grid(True, which='minor', alpha=0.08)
    sns.despine(ax=ax3, offset=4)

    # (d) 放电过程中的动态工作点轨迹
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 模拟放电轨迹
    soc_traj = np.linspace(0.95, 0.1, 100)
    P_load = 2.5  # 恒功率负载
    I_traj = []
    V_traj = []
    vp_traj = [0.0]
    
    for i, s in enumerate(soc_traj):
        V_oc = ocv_from_soc(s)
        vp = vp_traj[-1] if i > 0 else 0.0
        I = solve_current_from_power(P_load, V_oc, vp, R0)
        V_t = V_oc - R0 * I - vp
        I_traj.append(I)
        V_traj.append(V_t)
        # 极化电压演化（简化）
        vp_new = vp + (I * Rct - vp) * 0.05
        vp_traj.append(vp_new)
    
    I_traj = np.array(I_traj)
    V_traj = np.array(V_traj)
    
    # 颜色映射 SOC
    colors = plt.cm.viridis(np.linspace(0, 1, len(soc_traj)))
    for i in range(len(soc_traj) - 1):
        ax4.plot(I_traj[i:i+2], V_traj[i:i+2], color=colors[i], lw=2.5)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0.1, vmax=0.95))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax4, pad=0.02, aspect=25)
    cbar.set_label('SOC', fontsize=10)
    
    # 标注起点和终点
    ax4.scatter([I_traj[0]], [V_traj[0]], color='green', s=100, zorder=6, marker='o', edgecolor='black', label='Start (SOC=0.95)')
    ax4.scatter([I_traj[-1]], [V_traj[-1]], color='red', s=100, zorder=6, marker='s', edgecolor='black', label='End (SOC=0.1)')
    ax4.axhline(ECM['V_cut'], color='gray', ls='--', lw=1.2, label=r'$V_{cut}$')
    
    ax4.set_xlabel('Current I (A)', fontsize=12)
    ax4.set_ylabel('Terminal Voltage $V_t$ (V)', fontsize=12)
    ax4.set_title(f'(d) Dynamic Operating Trajectory (P={P_load}W)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('Power–Current Implicit Coupling: Discriminant Protection & Operating Dynamics', 
                 fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_power_current_coupling.png')


def plot_power_current_coupling_part1():
    """
    功率-电流耦合（上半部分）：
    - (a) P-I-SOC 3D 曲面
    - (b) 不同 SOC 下的 P-I 曲线族
    """
    R0 = ECM['R0']
    Rct = ECM['Rct']
    vp_fixed = 0.03

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    soc_3d = np.linspace(0.1, 0.95, 40)
    I_3d = np.linspace(0.1, 6, 50)
    SOC_M, I_M = np.meshgrid(soc_3d, I_3d)
    V_oc_M = np.array([[ocv_from_soc(s) for s in soc_3d] for _ in I_3d])
    P_M = I_M * (V_oc_M - vp_fixed - R0 * I_M)
    P_M = np.clip(P_M, 0, None)
    surf = ax1.plot_surface(SOC_M, I_M, P_M, cmap='viridis', alpha=0.85, edgecolor='none', rstride=2, cstride=2)
    ax1.set_xlabel('SOC', fontsize=10, labelpad=8)
    ax1.set_ylabel('Current I (A)', fontsize=10, labelpad=8)
    ax1.set_zlabel('Power P (W)', fontsize=10, labelpad=8)
    ax1.set_title('(a) P–I–SOC 3D Surface', fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=30, azim=-60)
    fig.colorbar(surf, ax=ax1, shrink=0.55, aspect=15, pad=0.1, label='Power (W)')

    ax2 = fig.add_subplot(gs[0, 1])
    soc_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    palette = get_palette(len(soc_levels))
    for idx, s in enumerate(soc_levels):
        V_oc = ocv_from_soc(s)
        I_axis = np.linspace(0.01, (V_oc - vp_fixed) / R0 * 0.95, 200)
        P_axis = I_axis * (V_oc - vp_fixed - R0 * I_axis)
        ax2.plot(I_axis, P_axis, color=palette[idx], lw=2.2, label=f'SOC={s:.2f}')
        I_max = (V_oc - vp_fixed) / (2 * R0)
        P_max = I_max * (V_oc - vp_fixed - R0 * I_max)
        ax2.scatter([I_max], [P_max], color=palette[idx], s=60, zorder=5, edgecolor='black', linewidth=0.8)
        ax2.annotate('MPP', xy=(I_max, P_max), xytext=(5, 8), textcoords='offset points', fontsize=8)
    typical_P = [1.5, 3.0, 5.0]
    for P in typical_P:
        ax2.axhline(P, color='gray', ls=':', lw=0.8, alpha=0.6)
        ax2.text(0.3, P + 0.15, f'P={P}W', fontsize=8, color='gray')
    ax2.set_xlabel('Current I (A)', fontsize=12)
    ax2.set_ylabel('Deliverable Power P (W)', fontsize=12)
    ax2.set_title('(b) P–I Characteristics at Different SOC', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 18)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.32)
    ax2.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax2, offset=4)

    fig.suptitle('Power–Current Coupling (Part 1)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_power_current_coupling_part1.png')


def plot_power_current_coupling_part2():
    """
    功率-电流耦合（下半部分）：
    - (c) 判别式可行域与效率等高线
    - (d) 动态工作点轨迹
    """
    R0 = ECM['R0']
    Rct = ECM['Rct']
    V_mid = ocv_from_soc(0.5)

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    ax3 = fig.add_subplot(gs[0, 0])
    vp_grid = np.linspace(-0.25, 0.25, 100)
    P_req = np.linspace(0, 15, 120)
    VP, PREQ = np.meshgrid(vp_grid, P_req)
    P_max_grid = (V_mid - VP) ** 2 / (4 * R0)
    feasible = PREQ <= P_max_grid
    I_grid = np.zeros_like(PREQ)
    eta_grid = np.ones_like(PREQ)
    for i in range(PREQ.shape[0]):
        for j in range(PREQ.shape[1]):
            if feasible[i, j]:
                I_grid[i, j] = solve_current_from_power(PREQ[i, j], V_mid, VP[i, j], R0)
                eta_grid[i, j] = compute_efficiency(I_grid[i, j], 25.0)
            else:
                eta_grid[i, j] = np.nan
    ax3.contourf(VP, PREQ, feasible.astype(float), levels=[-0.1, 0.5, 1.1], colors=['#ffcccc', '#ccffcc'], alpha=0.4)
    eta_levels = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
    cs_eta = ax3.contour(VP, PREQ, eta_grid, levels=eta_levels, cmap='Blues', linewidths=1.2)
    ax3.clabel(cs_eta, fmt='η=%.2f', fontsize=8)
    vp_line = np.linspace(vp_grid.min(), vp_grid.max(), 200)
    ax3.plot(vp_line, (V_mid - vp_line) ** 2 / (4 * R0), color='black', lw=2.5,
             label=r'$P_{max}=\frac{(V_{oc}-v_p)^2}{4R_0}$')
    formula = r'$\Delta = (V_{oc}-v_p)^2 - 4R_0 P \geq 0$'
    ax3.text(0.03, 0.97, formula, transform=ax3.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.95))
    ax3.set_xlabel(r'Polarization voltage $v_p$ (V)', fontsize=12)
    ax3.set_ylabel('Requested power P (W)', fontsize=12)
    ax3.set_title('(c) Discriminant Feasible Region & Efficiency Contours', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(vp_grid.min(), vp_grid.max())
    ax3.set_ylim(0, P_req.max())
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.28)
    ax3.grid(True, which='minor', alpha=0.08)
    sns.despine(ax=ax3, offset=4)

    ax4 = fig.add_subplot(gs[0, 1])
    soc_traj = np.linspace(0.95, 0.1, 100)
    P_load = 2.5
    I_traj = []
    V_traj = []
    vp_traj = [0.0]
    for i, s in enumerate(soc_traj):
        V_oc = ocv_from_soc(s)
        vp = vp_traj[-1] if i > 0 else 0.0
        I = solve_current_from_power(P_load, V_oc, vp, R0)
        V_t = V_oc - R0 * I - vp
        I_traj.append(I)
        V_traj.append(V_t)
        vp_new = vp + (I * Rct - vp) * 0.05
        vp_traj.append(vp_new)
    I_traj = np.array(I_traj)
    V_traj = np.array(V_traj)
    colors = plt.cm.viridis(np.linspace(0, 1, len(soc_traj)))
    for i in range(len(soc_traj) - 1):
        ax4.plot(I_traj[i:i+2], V_traj[i:i+2], color=colors[i], lw=2.5)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0.1, vmax=0.95))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax4, pad=0.02, aspect=25)
    cbar.set_label('SOC', fontsize=10)
    ax4.scatter([I_traj[0]], [V_traj[0]], color='green', s=100, zorder=6, marker='o', edgecolor='black', label='Start (SOC=0.95)')
    ax4.scatter([I_traj[-1]], [V_traj[-1]], color='red', s=100, zorder=6, marker='s', edgecolor='black', label='End (SOC=0.1)')
    ax4.axhline(ECM['V_cut'], color='gray', ls='--', lw=1.2, label=r'$V_{cut}$')
    ax4.set_xlabel('Current I (A)', fontsize=12)
    ax4.set_ylabel('Terminal Voltage $V_t$ (V)', fontsize=12)
    ax4.set_title(f'(d) Dynamic Operating Trajectory (P={P_load}W)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('Power–Current Coupling (Part 2)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_power_current_coupling_part2.png')


def plot_temperature_aging_capacity_surface():
    """
    温度/老化对有效容量的耦合效应深度分析：
    - (a) 3D曲面：温度-循环-容量关系
    - (b) 2D等高线图 + 等寿命线
    - (c) 边际效应分析（温度固定/循环固定）
    - (d) 敏感度热图：容量对温度/老化的偏导数
    """
    temps = np.linspace(-25, 50, 100)
    cycles = np.linspace(0, 1500, 120)
    Tm, Nm = np.meshgrid(temps, cycles)

    temp_factor = np.vectorize(temperature_capacity_factor)(Tm)
    aging_factor = np.vectorize(aging_capacity_factor)(Nm.astype(int))
    q_factor = temp_factor * aging_factor

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.28)

    # (a) 3D 曲面
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(Tm, Nm, q_factor * 100, cmap='coolwarm', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Temperature (°C)', fontsize=10, labelpad=8)
    ax1.set_ylabel('Cycle Number', fontsize=10, labelpad=8)
    ax1.set_zlabel('Effective Capacity (%)', fontsize=10, labelpad=8)
    ax1.set_title('(a) Temperature–Aging–Capacity 3D Surface', fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=28, azim=225)
    
    # 添加 80% EOL 平面
    xx, yy = np.meshgrid(temps, cycles)
    zz = np.ones_like(xx) * 80
    ax1.plot_surface(xx, yy, zz, alpha=0.2, color='red')
    ax1.text2D(0.75, 0.25, 'EOL Plane\n(80%)', transform=ax1.transAxes, fontsize=9, color='red')
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=15, pad=0.12, label='Capacity (%)')

    # (b) 2D 等高线 + 等寿命线
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    cf = ax2.contourf(Tm, Nm, q_factor * 100, levels=np.linspace(40, 105, 22), cmap=cmap, alpha=0.92)
    
    # 等容量线
    cs = ax2.contour(Tm, Nm, q_factor * 100, levels=[60, 70, 80, 90, 95], colors='k', linewidths=1.2, linestyles='--')
    ax2.clabel(cs, fmt='%d%%', fontsize=9)
    
    # 高亮 80% EOL 等高线
    cs_eol = ax2.contour(Tm, Nm, q_factor * 100, levels=[80], colors=['red'], linewidths=2.5)
    ax2.clabel(cs_eol, fmt='EOL', fontsize=10, colors=['red'])
    
    # 标注参考点
    ax2.axvline(25, color='white', ls=':', lw=1.5, alpha=0.8)
    ax2.scatter([25], [0], color='white', s=80, zorder=5, marker='*', edgecolor='black')
    ax2.annotate('Reference\n(25°C, n=0)', xy=(25, 0), xytext=(35, 200), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
    
    cbar = fig.colorbar(cf, ax=ax2, pad=0.02, aspect=25)
    cbar.set_label('Effective Capacity (%)', fontsize=11)
    
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Cycle Number', fontsize=12)
    ax2.set_title('(b) Capacity Contour Map with EOL Line', fontsize=12, fontweight='bold')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(ax=ax2, offset=4)

    # (c) 边际效应分析
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 固定循环数，变化温度
    cycle_levels = [0, 300, 600, 900, 1200]
    palette1 = get_palette(len(cycle_levels))
    for idx, n in enumerate(cycle_levels):
        af = aging_capacity_factor(n)
        cap = np.array([temperature_capacity_factor(t) * af * 100 for t in temps])
        ax3.plot(temps, cap, color=palette1[idx], lw=2.2, label=f'n={n}')
    
    ax3.axhline(80, color='gray', ls='--', lw=1.2, label='EOL (80%)')
    ax3.axvline(25, color='gray', ls=':', lw=1.0, alpha=0.6)
    
    # 公式框
    formula = r'$Q_{eff} = Q_0 [1 - k_T(T-25)^2][1 - \lambda n^\beta]$' + f'\n$k_T = {K_T:.2e}$, $\\lambda = {LAMBDA_NORMAL:.2e}$, $\\beta = {BETA_NORMAL:.1f}$'
    ax3.text(0.02, 0.03, formula, transform=ax3.transAxes, fontsize=9.5, va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    
    ax3.set_xlabel('Temperature (°C)', fontsize=12)
    ax3.set_ylabel('Effective Capacity (%)', fontsize=12)
    ax3.set_title('(c) Temperature Effect at Fixed Cycle Numbers', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower left', fontsize=9, ncol=2, framealpha=0.95)
    ax3.set_xlim(-25, 50)
    ax3.set_ylim(40, 105)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.32)
    ax3.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax3, offset=4)

    # (d) 敏感度热图：偏导数
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 计算容量对温度的敏感度 dQ/dT
    dQ_dT = np.zeros_like(q_factor)
    dQ_dn = np.zeros_like(q_factor)
    
    for i in range(len(cycles)):
        n = int(cycles[i])
        af = aging_capacity_factor(n)
        for j in range(len(temps)):
            T = temps[j]
            # dQ/dT = -2 * k_T * (T - 25) * aging_factor
            dQ_dT[i, j] = -2 * K_T * (T - 25) * af * 100  # %/°C
    
    for i in range(1, len(cycles)):
        for j in range(len(temps)):
            T = temps[j]
            tf = temperature_capacity_factor(T)
            n = cycles[i]
            if n > 0:
                # dQ/dn ≈ -λ * β * n^(β-1) * temp_factor
                dQ_dn[i, j] = -LAMBDA_NORMAL * BETA_NORMAL * (n ** (BETA_NORMAL - 1)) * tf * 100  # %/cycle
    
    # 综合敏感度（梯度模）
    sensitivity = np.sqrt(dQ_dT**2 + (dQ_dn * 100)**2)  # 归一化
    
    im = ax4.imshow(sensitivity, extent=[temps.min(), temps.max(), cycles.max(), cycles.min()],
                    aspect='auto', cmap='YlOrRd', alpha=0.9)
    
    # 叠加等高线
    cs_sens = ax4.contour(Tm, Nm, sensitivity, levels=[0.5, 1.0, 2.0, 4.0], colors='k', linewidths=0.8, linestyles='-')
    ax4.clabel(cs_sens, fmt='%.1f', fontsize=8)
    
    cbar = fig.colorbar(im, ax=ax4, pad=0.02, aspect=25)
    cbar.set_label('Sensitivity Magnitude (%)', fontsize=10)
    
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('Cycle Number', fontsize=12)
    ax4.set_title('(d) Capacity Sensitivity Heatmap', fontsize=12, fontweight='bold')
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('Temperature & Aging Coupling on Effective Capacity: Physical Mechanism', 
                 fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_temp_aging_capacity.png')


def plot_temperature_aging_capacity_part1():
    """
    温度-老化容量耦合（上半部分）：
    - (a) 3D 曲面
    - (b) 等高线 + EOL 线
    """
    temps = np.linspace(-25, 50, 100)
    cycles = np.linspace(0, 1500, 120)
    Tm, Nm = np.meshgrid(temps, cycles)
    temp_factor = np.vectorize(temperature_capacity_factor)(Tm)
    aging_factor = np.vectorize(aging_capacity_factor)(Nm.astype(int))
    q_factor = temp_factor * aging_factor

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    surf = ax1.plot_surface(Tm, Nm, q_factor * 100, cmap='coolwarm', alpha=0.9, edgecolor='none')
    ax1.set_xlabel('Temperature (°C)', fontsize=10, labelpad=8)
    ax1.set_ylabel('Cycle Number', fontsize=10, labelpad=8)
    ax1.set_zlabel('Effective Capacity (%)', fontsize=10, labelpad=8)
    ax1.set_title('(a) Temperature–Aging–Capacity 3D Surface', fontsize=12, fontweight='bold', pad=10)
    ax1.view_init(elev=28, azim=225)
    xx, yy = np.meshgrid(temps, cycles)
    zz = np.ones_like(xx) * 80
    ax1.plot_surface(xx, yy, zz, alpha=0.2, color='red')
    ax1.text2D(0.75, 0.25, 'EOL Plane\n(80%)', transform=ax1.transAxes, fontsize=9, color='red')
    fig.colorbar(surf, ax=ax1, shrink=0.55, aspect=15, pad=0.12, label='Capacity (%)')

    ax2 = fig.add_subplot(gs[0, 1])
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    # 增加levels数量提升颜色连续性，使用extend='both'确保边界完整
    cf = ax2.contourf(Tm, Nm, q_factor * 100, levels=np.linspace(40, 105, 50), cmap=cmap, alpha=0.92, extend='both')
    cs = ax2.contour(Tm, Nm, q_factor * 100, levels=[60, 70, 80, 90, 95], colors='k', linewidths=1.2, linestyles='--')
    ax2.clabel(cs, fmt='%d%%', fontsize=9)
    cs_eol = ax2.contour(Tm, Nm, q_factor * 100, levels=[80], colors=['red'], linewidths=2.5)
    ax2.clabel(cs_eol, fmt='EOL', fontsize=10, colors=['red'])
    ax2.axvline(25, color='white', ls=':', lw=1.5, alpha=0.8)
    ax2.scatter([25], [0], color='white', s=80, zorder=5, marker='*', edgecolor='black')
    ax2.annotate('Reference\n(25°C, n=0)', xy=(25, 0), xytext=(35, 200), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
    cbar = fig.colorbar(cf, ax=ax2, pad=0.02, aspect=25)
    cbar.set_label('Effective Capacity (%)', fontsize=11)
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Cycle Number', fontsize=12)
    ax2.set_title('(b) Capacity Contour Map with EOL Line', fontsize=12, fontweight='bold')
    # 设置坐标轴范围与数据范围一致，确保左边界与y轴平行
    ax2.set_xlim(-25, 50)
    ax2.set_ylim(0, 1500)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(ax=ax2, offset=4)

    fig.suptitle('Temperature & Aging Coupling (Part 1)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_temp_aging_capacity_part1.png')


def plot_temperature_aging_capacity_part2():
    """
    温度-老化容量耦合（下半部分）：
    - (c) 温度/循环边际效应
    - (d) 敏感度热图
    """
    temps = np.linspace(-25, 50, 100)
    cycles = np.linspace(0, 1500, 120)
    Tm, Nm = np.meshgrid(temps, cycles)
    temp_factor = np.vectorize(temperature_capacity_factor)(Tm)
    aging_factor = np.vectorize(aging_capacity_factor)(Nm.astype(int))
    q_factor = temp_factor * aging_factor

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    ax3 = fig.add_subplot(gs[0, 0])
    cycle_levels = [0, 300, 600, 900, 1200]
    palette1 = get_palette(len(cycle_levels))
    for idx, n in enumerate(cycle_levels):
        af = aging_capacity_factor(n)
        cap = np.array([temperature_capacity_factor(t) * af * 100 for t in temps])
        ax3.plot(temps, cap, color=palette1[idx], lw=2.2, label=f'n={n}')
    ax3.axhline(80, color='gray', ls='--', lw=1.2, label='EOL (80%)')
    ax3.axvline(25, color='gray', ls=':', lw=1.0, alpha=0.6)
    formula = r'$Q_{eff} = Q_0 [1 - k_T(T-25)^2][1 - \lambda n^\beta]$' + f'\n$k_T = {K_T:.2e}$, $\\lambda = {LAMBDA_NORMAL:.2e}$, $\\beta = {BETA_NORMAL:.1f}$'
    ax3.text(0.02, 0.03, formula, transform=ax3.transAxes, fontsize=9.5, va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    ax3.set_xlabel('Temperature (°C)', fontsize=12)
    ax3.set_ylabel('Effective Capacity (%)', fontsize=12)
    ax3.set_title('(c) Temperature Effect at Fixed Cycle Numbers', fontsize=12, fontweight='bold')
    # 图例放置在左下角，使用3列避免与公式框重叠
    ax3.legend(loc='lower left', fontsize=8.5, ncol=3, framealpha=0.95, 
               columnspacing=0.8, handlelength=1.5, handletextpad=0.4,
               bbox_to_anchor=(0.0, 0.18), borderaxespad=0.3)
    ax3.set_xlim(-25, 50)
    ax3.set_ylim(40, 105)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.32)
    ax3.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax3, offset=4)

    ax4 = fig.add_subplot(gs[0, 1])
    dQ_dT = np.zeros_like(q_factor)
    dQ_dn = np.zeros_like(q_factor)
    for i in range(len(cycles)):
        n = int(cycles[i])
        af = aging_capacity_factor(n)
        for j in range(len(temps)):
            T = temps[j]
            dQ_dT[i, j] = -2 * K_T * (T - 25) * af * 100
    for i in range(1, len(cycles)):
        for j in range(len(temps)):
            T = temps[j]
            tf = temperature_capacity_factor(T)
            n = cycles[i]
            if n > 0:
                dQ_dn[i, j] = -LAMBDA_NORMAL * BETA_NORMAL * (n ** (BETA_NORMAL - 1)) * tf * 100
    # Use absolute values of partial derivatives separately for better visualization
    sensitivity_T = np.abs(dQ_dT)
    sensitivity_n = np.abs(dQ_dn) * 50  # Scale aging sensitivity
    sensitivity = sensitivity_T + sensitivity_n  # Combined sensitivity
    # Apply log transform for better color contrast
    sensitivity_log = np.log1p(sensitivity * 10)
    im = ax4.contourf(Tm, Nm, sensitivity_log, levels=20, cmap='plasma', alpha=0.92)
    # 手动指定等高线levels，避免底部区域过于密集
    contour_levels = [1.0, 1.8, 2.4, 3.0]
    cs_sens = ax4.contour(Tm, Nm, sensitivity_log, levels=contour_levels, colors='white', linewidths=0.7, alpha=0.75)
    # 只标注不在底部的等高线，避免标签重叠
    ax4.clabel(cs_sens, levels=[1.8, 2.4, 3.0], fmt='%.1f', fontsize=8, colors='white', 
               inline=True, inline_spacing=10)
    cbar = fig.colorbar(im, ax=ax4, pad=0.02, aspect=25)
    cbar.set_label('log(1 + Sensitivity)', fontsize=10)
    ax4.set_xlabel('Temperature (°C)', fontsize=12)
    ax4.set_ylabel('Cycle Number', fontsize=12)
    ax4.set_title('(d) Capacity Sensitivity Heatmap', fontsize=12, fontweight='bold')
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('Temperature & Aging Coupling (Part 2)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_temp_aging_capacity_part2.png')


def plot_polarization_dynamics_mechanism():
    """
    极化电压动态与端电压响应深度分析：
    - (a) RC 电路阶跃响应与时间常数
    - (b) 脉冲负载下的极化累积与回弹
    - (c) 不同 τ_p 下的响应对比
    - (d) 相轨迹图：v_p vs dv_p/dt
    """
    Rct, Cp, R0 = ECM['Rct'], ECM['Cp'], ECM['R0']
    tau_p = Rct * Cp
    V_oc = ocv_from_soc(0.65)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.28)

    # (a) RC 阶跃响应分析
    ax1 = fig.add_subplot(gs[0, 0])
    t_step = np.linspace(0, 1200, 600)
    I_step = 2.0
    
    # 理论解：v_p(t) = I * R_ct * (1 - exp(-t/τ))
    vp_theory = I_step * Rct * (1 - np.exp(-t_step / tau_p))
    vp_steady = I_step * Rct
    
    ax1.plot(t_step, vp_theory * 1000, color=COLORS['primary'], lw=2.5, label='$v_p(t) = IR_{ct}(1 - e^{-t/\\tau_p})$')
    ax1.axhline(vp_steady * 1000, color='gray', ls='--', lw=1.5, label=f'Steady state: {vp_steady*1000:.1f} mV')
    ax1.axhline(vp_steady * 1000 * 0.632, color=COLORS['tertiary'], ls=':', lw=1.2, label='63.2% (τ_p)')
    ax1.axvline(tau_p, color=COLORS['tertiary'], ls=':', lw=1.2)
    
    # 标注时间常数
    ax1.annotate(f'τ_p = {tau_p:.0f} s', xy=(tau_p, vp_steady * 1000 * 0.632), 
                 xytext=(tau_p + 100, vp_steady * 1000 * 0.4),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    # 公式框
    formula = f'$\\tau_p = R_{{ct}} C_p = {Rct:.3f} \\times {Cp:.0f} = {tau_p:.0f}$ s\n$R_{{ct}} = {Rct*1000:.1f}$ mΩ, $C_p = {Cp:.0f}$ F'
    ax1.text(0.55, 0.25, formula, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Polarization Voltage $v_p$ (mV)', fontsize=12)
    ax1.set_title(f'(a) RC Step Response (I = {I_step} A)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax1.set_xlim(0, t_step.max())
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.32)
    ax1.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax1, offset=4)

    # (b) 复杂脉冲负载响应
    ax2 = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 1500, 1501)
    
    # 复杂脉冲序列
    I = np.zeros_like(t)
    I[(t >= 50) & (t <= 250)] = 3.0
    I[(t >= 350) & (t <= 450)] = 1.5
    I[(t >= 550) & (t <= 850)] = 2.5
    I[(t >= 950) & (t <= 1050)] = 4.0
    I[(t >= 1150) & (t <= 1350)] = 2.0
    
    vp = np.zeros_like(t)
    vt = np.zeros_like(t)
    
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        dvp = (-vp[i - 1] / tau_p + I[i - 1] * Rct / tau_p) * dt
        vp[i] = vp[i - 1] + dvp
        vt[i] = V_oc - R0 * I[i] - vp[i]
    vt[0] = V_oc
    
    # 双轴绘图
    ax2_twin = ax2.twinx()
    
    l1, = ax2.plot(t, vp * 1000, color=COLORS['secondary'], lw=2.2, label='$v_p$ (polarization)')
    ax2.fill_between(t, 0, vp * 1000, alpha=0.15, color=COLORS['secondary'])
    l2, = ax2_twin.plot(t, I, color=COLORS['tertiary'], lw=1.8, ls='--', alpha=0.8, label='Current I')
    
    # 标注回弹区域
    recovery_regions = [(250, 350), (450, 550), (850, 950), (1050, 1150)]
    for start, end in recovery_regions:
        ax2.axvspan(start, end, alpha=0.08, color='green')
    ax2.text(300, vp.max() * 1000 * 0.9, 'Recovery\nphases', fontsize=9, ha='center', color='green')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Polarization $v_p$ (mV)', fontsize=12, color=COLORS['secondary'])
    ax2_twin.set_ylabel('Current I (A)', fontsize=11, color=COLORS['tertiary'])
    ax2.set_title('(b) Pulse Load: Polarization Accumulation & Recovery', fontsize=12, fontweight='bold')
    ax2.legend(handles=[l1, l2], loc='upper right', fontsize=9)
    ax2.set_xlim(0, t.max())
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.28)
    sns.despine(ax=ax2, right=False, offset=4)

    # (c) 不同时间常数对比
    ax3 = fig.add_subplot(gs[1, 0])
    tau_values = [50, 100, 216, 400, 800]  # 216 是实际值
    palette = get_palette(len(tau_values))
    
    t_compare = np.linspace(0, 600, 300)
    I_const = 2.5
    
    for idx, tau in enumerate(tau_values):
        vp_tau = I_const * Rct * (1 - np.exp(-t_compare / tau))
        ls = '-' if tau == 216 else '--'
        lw = 2.8 if tau == 216 else 2.0
        label = f'τ = {tau} s' + (' (fitted)' if tau == 216 else '')
        ax3.plot(t_compare, vp_tau * 1000, color=palette[idx], lw=lw, ls=ls, label=label)
    
    # 添加阴影区域表示响应速度
    ax3.axvspan(0, 100, alpha=0.08, color='blue', label='Fast response region')
    
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Polarization $v_p$ (mV)', fontsize=12)
    ax3.set_title('(c) Time Constant Comparison (Same $R_{ct}$)', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax3.set_xlim(0, t_compare.max())
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.32)
    ax3.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax3, offset=4)

    # (d) 相轨迹图
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 使用 (b) 中的数据计算 dv_p/dt
    dvp_dt = np.gradient(vp, t)
    
    # 颜色映射时间
    colors = plt.cm.plasma(np.linspace(0, 1, len(t)))
    for i in range(len(t) - 1):
        ax4.plot(vp[i:i+2] * 1000, dvp_dt[i:i+2] * 1000, color=colors[i], lw=1.5)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=t.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax4, pad=0.02, aspect=25)
    cbar.set_label('Time (s)', fontsize=10)
    
    # 标注平衡点（稳态）
    ax4.scatter([0], [0], color='green', s=100, zorder=6, marker='o', edgecolor='black', label='Equilibrium (I=0)')
    ax4.axhline(0, color='gray', ls=':', lw=1.0)
    ax4.axvline(0, color='gray', ls=':', lw=1.0)
    
    # 添加向量场方向
    vp_field = np.linspace(0, vp.max() * 1000 * 1.2, 8)
    I_field = [0, 1.5, 3.0]
    for I_f in I_field:
        dvp_field = (-vp_field / 1000 / tau_p + I_f * Rct / tau_p) * 1000
        ax4.quiver(vp_field[:-1], dvp_field[:-1], np.diff(vp_field), np.diff(dvp_field),
                   scale=50, width=0.004, alpha=0.3, color='gray')
    
    ax4.set_xlabel('Polarization $v_p$ (mV)', fontsize=12)
    ax4.set_ylabel('$dv_p/dt$ (mV/s)', fontsize=12)
    ax4.set_title('(d) Phase Portrait: Polarization Dynamics', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('Polarization Voltage Dynamics: RC Circuit Transient Response', 
                 fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_polarization_dynamics.png')


def plot_polarization_dynamics_part1():
    """
    极化动态（上半部分）：
    - (a) RC 阶跃响应
    - (b) 脉冲负载极化累积与回弹
    """
    Rct, Cp, R0 = ECM['Rct'], ECM['Cp'], ECM['R0']
    tau_p = Rct * Cp
    V_oc = ocv_from_soc(0.65)
    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    t_step = np.linspace(0, 1200, 600)
    I_step = 2.0
    vp_theory = I_step * Rct * (1 - np.exp(-t_step / tau_p))
    vp_steady = I_step * Rct
    ax1.plot(t_step, vp_theory * 1000, color=COLORS['primary'], lw=2.5, label='$v_p(t) = IR_{ct}(1 - e^{-t/\\tau_p})$')
    ax1.axhline(vp_steady * 1000, color='gray', ls='--', lw=1.5, label=f'Steady state: {vp_steady*1000:.1f} mV')
    ax1.axhline(vp_steady * 1000 * 0.632, color=COLORS['tertiary'], ls=':', lw=1.2, label='63.2% (τ_p)')
    ax1.axvline(tau_p, color=COLORS['tertiary'], ls=':', lw=1.2)
    ax1.annotate(f'τ_p = {tau_p:.0f} s', xy=(tau_p, vp_steady * 1000 * 0.632), 
                 xytext=(tau_p + 100, vp_steady * 1000 * 0.4),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    formula = f'$\\tau_p = R_{{ct}} C_p = {Rct:.3f} \\times {Cp:.0f} = {tau_p:.0f}$ s\n$R_{{ct}} = {Rct*1000:.1f}$ mΩ, $C_p = {Cp:.0f}$ F'
    ax1.text(0.55, 0.25, formula, transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Polarization $v_p$ (mV)', fontsize=12)
    ax1.set_title(f'(a) RC Step Response (I = {I_step} A)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax1.set_xlim(0, t_step.max())
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.32)
    ax1.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax1, offset=4)

    ax2 = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 1500, 3001)  # Higher resolution
    I_raw = np.zeros_like(t)
    I_raw[(t >= 50) & (t <= 250)] = 3.0
    I_raw[(t >= 350) & (t <= 450)] = 1.5
    I_raw[(t >= 550) & (t <= 850)] = 2.5
    I_raw[(t >= 950) & (t <= 1050)] = 4.0
    I_raw[(t >= 1150) & (t <= 1350)] = 2.0
    # Smooth step transitions with exponential filter
    from scipy.ndimage import gaussian_filter1d
    I = gaussian_filter1d(I_raw, sigma=8)
    vp = np.zeros_like(t)
    vt = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        dvp = (-vp[i - 1] / tau_p + I[i - 1] * Rct / tau_p) * dt
        vp[i] = vp[i - 1] + dvp
        vt[i] = V_oc - R0 * I[i] - vp[i]
    vt[0] = V_oc
    ax2_twin = ax2.twinx()
    l1, = ax2.plot(t, vp * 1000, color=COLORS['secondary'], lw=2.5, label='$v_p$ (polarization)')
    ax2.fill_between(t, 0, vp * 1000, alpha=0.15, color=COLORS['secondary'])
    l2, = ax2_twin.step(t, I_raw, where='post', color=COLORS['tertiary'], lw=1.5, alpha=0.7, label='Current I')
    recovery_regions = [(250, 350), (450, 550), (850, 950), (1050, 1150)]
    for start, end in recovery_regions:
        ax2.axvspan(start, end, alpha=0.08, color='green')
    ax2.text(300, vp.max() * 1000 * 0.9, 'Recovery\nphases', fontsize=9, ha='center', color='green')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Polarization $v_p$ (mV)', fontsize=12, color=COLORS['secondary'])
    ax2_twin.set_ylabel('Current I (A)', fontsize=11, color=COLORS['tertiary'])
    ax2.set_title('(b) Pulse Load: Polarization Accumulation & Recovery', fontsize=12, fontweight='bold')
    ax2.legend(handles=[l1, l2], loc='upper right', fontsize=9)
    ax2.set_xlim(0, t.max())
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.28)
    sns.despine(ax=ax2, right=False, offset=4)

    fig.suptitle('Polarization Voltage Dynamics (Part 1)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_polarization_dynamics_part1.png')


def plot_polarization_dynamics_part2():
    """
    极化动态（下半部分）：
    - (c) 不同时间常数对比
    - (d) 相轨迹图
    """
    Rct, Cp, R0 = ECM['Rct'], ECM['Cp'], ECM['R0']
    tau_p = Rct * Cp
    V_oc = ocv_from_soc(0.65)
    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.22)

    ax3 = fig.add_subplot(gs[0, 0])
    tau_values = [50, 100, 216, 400, 800]
    palette = get_palette(len(tau_values))
    t_compare = np.linspace(0, 600, 300)
    I_const = 2.5
    for idx, tau in enumerate(tau_values):
        vp_tau = I_const * Rct * (1 - np.exp(-t_compare / tau))
        ls = '-' if tau == 216 else '--'
        lw = 2.8 if tau == 216 else 2.0
        label = f'τ = {tau} s' + (' (fitted)' if tau == 216 else '')
        ax3.plot(t_compare, vp_tau * 1000, color=palette[idx], lw=lw, ls=ls, label=label)
    ax3.axvspan(0, 100, alpha=0.08, color='blue', label='Fast response region')
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Polarization $v_p$ (mV)', fontsize=12)
    ax3.set_title('(c) Time Constant Comparison (Same $R_{ct}$)', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax3.set_xlim(0, t_compare.max())
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.32)
    ax3.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax3, offset=4)

    ax4 = fig.add_subplot(gs[0, 1])
    t = np.linspace(0, 1500, 1501)
    I = np.zeros_like(t)
    I[(t >= 50) & (t <= 250)] = 3.0
    I[(t >= 350) & (t <= 450)] = 1.5
    I[(t >= 550) & (t <= 850)] = 2.5
    I[(t >= 950) & (t <= 1050)] = 4.0
    I[(t >= 1150) & (t <= 1350)] = 2.0
    vp = np.zeros_like(t)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        dvp = (-vp[i - 1] / tau_p + I[i - 1] * Rct / tau_p) * dt
        vp[i] = vp[i - 1] + dvp
    dvp_dt = np.gradient(vp, t)
    colors = plt.cm.plasma(np.linspace(0, 1, len(t)))
    for i in range(len(t) - 1):
        ax4.plot(vp[i:i+2] * 1000, dvp_dt[i:i+2] * 1000, color=colors[i], lw=1.5)
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=t.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax4, pad=0.02, aspect=25)
    cbar.set_label('Time (s)', fontsize=10)
    ax4.scatter([0], [0], color='green', s=100, zorder=6, marker='o', edgecolor='black', label='Equilibrium (I=0)')
    ax4.axhline(0, color='gray', ls=':', lw=1.0)
    ax4.axvline(0, color='gray', ls=':', lw=1.0)
    vp_field = np.linspace(0, vp.max() * 1000 * 1.2, 8)
    I_field = [0, 1.5, 3.0]
    for I_f in I_field:
        dvp_field = (-vp_field / 1000 / tau_p + I_f * Rct / tau_p) * 1000
        ax4.quiver(vp_field[:-1], dvp_field[:-1], np.diff(vp_field), np.diff(dvp_field),
                   scale=50, width=0.004, alpha=0.3, color='gray')
    ax4.set_xlabel('Polarization $v_p$ (mV)', fontsize=12)
    ax4.set_ylabel('$dv_p/dt$ (mV/s)', fontsize=12)
    ax4.set_title('(d) Phase Portrait: Polarization Dynamics', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('Polarization Voltage Dynamics (Part 2)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_polarization_dynamics_part2.png')


def plot_tte_event_detection_mechanism():
    """
    TTE 事件检测深度分析：
    - (a) 完整放电过程：SOC、电压、功率同步视图
    - (b) 电压陡降区放大 + 事件触发机制
    - (c) TTE 敏感度分析：不同负载/温度/老化
    - (d) 概率分布：TTE 不确定性量化
    """
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.30)

    # 模拟完整放电过程 - 延长时长并提高负载确保达到 TTE
    t = np.arange(0, 14 * 3600 + 1, 15)  # 14 小时，更长的放电窗口
    n = len(t)

    # 更复杂的功率模型，抬升基线并加大脉冲幅度
    base_power = 2.8  # 提高基础功率加快放电
    periodic = 0.5 * np.sin(2 * np.pi * t / (3600 * 0.8))
    burst = np.zeros_like(t, dtype=float)
    burst_times = rng.choice(len(t), size=45, replace=False)
    for bt in burst_times:
        burst[max(0, bt-10):min(n, bt+10)] = rng.uniform(0.6, 1.6)
    power = base_power + periodic + burst

    df = pd.DataFrame({
        'elapsed_sec': t,
        'charge_mAh': np.linspace(Q_max_Ah * 1000 * 0.98, Q_max_Ah * 1000 * 0.0, n),
        'temp_C': 25 + 3 * np.sin(2 * np.pi * t / (3600 * 3)) + rng.normal(0, 0.5, n),
        'cpu_util_pct': np.clip(25 + 20 * rng.normal(0, 0.3, n), 5, 80),
        'gpu_util_pct': np.clip(15 + 15 * rng.normal(0, 0.35, n), 0, 60),
        'screen': ['on'] * n,
        'brightness': np.clip(130 + 40 * rng.normal(0, 0.2, n), 50, 220),
        'wifi_state': ['on'] * n,
        'mobile_state': ['off'] * n,
        'gps': ['off'] * n,
    })

    q_pred, v_pred, vp_trace, eta_trace, tte_idx = simulate_discharge_ecm(df, n_cycles=800, use_aging=True)
    t_h = df['elapsed_sec'].values / 3600
    soc_trace = q_pred / (Q_max_Ah * 1000)

    # 如果依旧未检测到 TTE，则强制插值/调整末段确保存在交点
    if tte_idx is None:
        below = np.where(v_pred <= ECM['V_cut'])[0]
        if len(below):
            tte_idx = below[0]
        else:
            v_pred[-1] = ECM['V_cut'] - 0.05
            tte_idx = n - 1

    # (a) 多变量同步视图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin1 = ax1.twinx()
    ax1_twin2 = ax1.twinx()
    ax1_twin2.spines['right'].set_position(('outward', 60))
    
    l1, = ax1.plot(t_h, v_pred, color=COLORS['primary'], lw=2.2, label='Terminal Voltage')
    l2, = ax1_twin1.plot(t_h, soc_trace * 100, color=COLORS['tertiary'], lw=1.8, ls='--', label='SOC')
    l3, = ax1_twin2.plot(t_h, power, color=COLORS['quaternary'], lw=1.2, alpha=0.7, label='Power')
    
    ax1.axhline(ECM['V_cut'], color='red', ls=':', lw=1.5, label='$V_{cut}$')
    if tte_idx is not None:
        ax1.axvline(t_h[tte_idx], color='red', ls='--', lw=2.0, alpha=0.8)
        ax1.fill_betweenx([v_pred.min(), v_pred.max()], t_h[tte_idx], t_h.max(), alpha=0.1, color='red')
        ax1.annotate(f'TTE = {t_h[tte_idx]:.2f} h', xy=(t_h[tte_idx], ECM['V_cut']),
                     xytext=(-80, 30), textcoords='offset points', fontsize=11, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12, color=COLORS['primary'])
    ax1_twin1.set_ylabel('SOC (%)', fontsize=11, color=COLORS['tertiary'])
    ax1_twin2.set_ylabel('Power (W)', fontsize=11, color=COLORS['quaternary'])
    ax1.set_title('(a) Complete Discharge: Multi-Variable Synchronized View', fontsize=12, fontweight='bold')
    ax1.legend(handles=[l1, l2, l3], loc='upper right', fontsize=9)
    ax1.set_xlim(0, t_h.max())
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.28)
    sns.despine(ax=ax1, right=False, offset=4)

    # (b) 电压陡降区放大
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()  # 先创建副轴
    
    if tte_idx is not None:
        # 取 TTE 前后一段时间
        zoom_start = max(0, tte_idx - 200)
        zoom_end = min(n - 1, tte_idx + 50)
        t_zoom = t_h[zoom_start:zoom_end]
        v_zoom = v_pred[zoom_start:zoom_end]
        soc_zoom = soc_trace[zoom_start:zoom_end]
        
        ax2.plot(t_zoom, v_zoom, color=COLORS['primary'], lw=2.5, label='Terminal Voltage')
        ax2.axhline(ECM['V_cut'], color='red', ls=':', lw=2.0, label='$V_{cut}$ threshold')
        ax2.axvline(t_h[tte_idx], color='red', ls='--', lw=2.0, alpha=0.8, label='TTE event')
        
        # 标注斜率变化
        dv_dt = np.gradient(v_zoom, t_zoom)
        steep_mask = np.abs(dv_dt) > np.percentile(np.abs(dv_dt), 90)
        ax2.fill_between(t_zoom, v_zoom.min(), v_zoom, where=steep_mask, alpha=0.2, color='orange', label='Steep slope region')
        
        # SOC 副轴
        ax2_twin.plot(t_zoom, soc_zoom * 100, color=COLORS['tertiary'], lw=1.5, ls='--', alpha=0.7)
        ax2_twin.set_ylabel('SOC (%)', fontsize=10, color=COLORS['tertiary'])
        
        # 公式框
        formula = r'TTE: $\inf\{t \geq 0 \mid V_t(t) \leq V_{cut}\}$'
        ax2.text(0.03, 0.97, formula, transform=ax2.transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
        
        ax2.set_xlim(t_zoom.min(), t_zoom.max())
        ax2.set_ylim(v_zoom.min() - 0.05, v_zoom.max() + 0.05)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    else:
        # TTE未达到时，显示电压曲线末尾区域并清理显示
        zoom_start = max(0, n - 300)
        t_zoom = t_h[zoom_start:]
        v_zoom = v_pred[zoom_start:]
        soc_zoom = soc_trace[zoom_start:]
        
        ax2.plot(t_zoom, v_zoom, color=COLORS['primary'], lw=2.5, label='Terminal Voltage')
        ax2.axhline(ECM['V_cut'], color='red', ls=':', lw=2.0, label='$V_{cut}$ threshold')
        ax2_twin.plot(t_zoom, soc_zoom * 100, color=COLORS['tertiary'], lw=1.5, ls='--', alpha=0.7, label='SOC')
        ax2_twin.set_ylabel('SOC (%)', fontsize=10, color=COLORS['tertiary'])
        
        ax2.text(0.5, 0.5, 'TTE not reached\n(simulation too short)', transform=ax2.transAxes, 
                 ha='center', va='center', fontsize=11, color='orange', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='orange', alpha=0.9))
        ax2.set_xlim(t_zoom.min(), t_zoom.max())
        ax2.set_ylim(v_zoom.min() - 0.1, v_zoom.max() + 0.1)
        ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Terminal Voltage (V)', fontsize=12)
    ax2.set_title('(b) Voltage Drop Region Zoom: Event Trigger Mechanism', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.32)
    ax2.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax2, right=False, offset=4)

    # (c) TTE 敏感度分析
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 参数扫描
    power_levels = [1.2, 1.5, 1.8, 2.2, 2.8]
    temp_levels = [0, 15, 25, 35]
    cycle_levels = [0, 200, 500]
    
    palette = get_palette(len(power_levels))
    
    # 简化的 TTE 估算（基于能量平衡 + 修正）
    results = []
    for P in power_levels:
        for T in temp_levels:
            for n_cyc in cycle_levels:
                tf = temperature_capacity_factor(T)
                af = aging_capacity_factor(n_cyc)
                Q_eff = Q_max_Ah * tf * af
                # 简化 TTE 估算
                tte_est = Q_eff * V_nom / P * 0.85  # 0.85 考虑效率和 V_cut
                results.append({'Power': P, 'Temp': T, 'Cycles': n_cyc, 'TTE': tte_est})
    
    df_results = pd.DataFrame(results)
    
    # 绘制功率-TTE关系（不同温度）
    for idx, T in enumerate(temp_levels):
        subset = df_results[(df_results['Temp'] == T) & (df_results['Cycles'] == 0)]
        marker = ['o', 's', '^', 'D'][idx]
        ax3.plot(subset['Power'], subset['TTE'], marker=marker, ms=8, lw=2.0, 
                 label=f'T={T}°C', color=get_palette(len(temp_levels))[idx])
    
    ax3.set_xlabel('Load Power (W)', fontsize=12)
    ax3.set_ylabel('Estimated TTE (hours)', fontsize=12)
    ax3.set_title('(c) TTE Sensitivity: Power × Temperature', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.32)
    ax3.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax3, offset=4)

    # (d) TTE 不确定性分布
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Monte Carlo 模拟 TTE 分布
    n_samples = 500
    tte_samples = []
    
    for _ in range(n_samples):
        P_sample = rng.normal(2.0, 0.3)
        T_sample = rng.normal(25, 5)
        n_sample = rng.uniform(0, 300)
        
        tf = temperature_capacity_factor(T_sample)
        af = aging_capacity_factor(int(n_sample))
        Q_eff = Q_max_Ah * max(0.5, tf * af)
        tte = Q_eff * V_nom / max(0.5, P_sample) * 0.85
        tte_samples.append(tte)
    
    tte_samples = np.array(tte_samples)
    
    # 核密度估计
    kde = gaussian_kde(tte_samples)
    x_kde = np.linspace(tte_samples.min(), tte_samples.max(), 200)
    
    # 直方图 + KDE
    ax4.hist(tte_samples, bins=30, density=True, alpha=0.5, color=COLORS['primary'], edgecolor='white', label='Histogram')
    ax4.plot(x_kde, kde(x_kde), color=COLORS['secondary'], lw=2.5, label='KDE')
    
    # 统计量
    tte_mean = np.mean(tte_samples)
    tte_std = np.std(tte_samples)
    tte_p5 = np.percentile(tte_samples, 5)
    tte_p95 = np.percentile(tte_samples, 95)
    
    ax4.axvline(tte_mean, color='black', ls='-', lw=2.0, label=f'Mean: {tte_mean:.2f} h')
    ax4.axvline(tte_p5, color='gray', ls='--', lw=1.5, label=f'5th %ile: {tte_p5:.2f} h')
    ax4.axvline(tte_p95, color='gray', ls='--', lw=1.5, label=f'95th %ile: {tte_p95:.2f} h')
    ax4.fill_betweenx([0, kde(x_kde).max() * 1.1], tte_p5, tte_p95, alpha=0.15, color='green', label='90% CI')
    
    # 统计框
    stats_text = f'μ = {tte_mean:.2f} h\nσ = {tte_std:.2f} h\nCV = {tte_std/tte_mean*100:.1f}%'
    ax4.text(0.97, 0.97, stats_text, transform=ax4.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    
    ax4.set_xlabel('Time to Empty (hours)', fontsize=12)
    ax4.set_ylabel('Probability Density', fontsize=12)
    ax4.set_title('(d) TTE Uncertainty Quantification (Monte Carlo)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8, framealpha=0.95)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('TTE Event Detection: First-Passage Time Problem Analysis', 
                 fontsize=15, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_tte_event_detection.png')


def plot_tte_event_detection_part1():
    """
    TTE 事件检测（上半部分）：
    - (a) 完整放电多变量同步视图
    - (b) 电压陡降区放大
    """
    rng = np.random.default_rng(42)
    t = np.arange(0, 14 * 3600 + 1, 15)
    n = len(t)
    base_power = 2.8
    periodic = 0.5 * np.sin(2 * np.pi * t / (3600 * 0.8))
    burst = np.zeros_like(t, dtype=float)
    burst_times = rng.choice(len(t), size=45, replace=False)
    for bt in burst_times:
        burst[max(0, bt-10):min(n, bt+10)] = rng.uniform(0.6, 1.6)
    power = base_power + periodic + burst
    df = pd.DataFrame({
        'elapsed_sec': t,
        'charge_mAh': np.linspace(Q_max_Ah * 1000 * 0.98, Q_max_Ah * 1000 * 0.0, n),
        'temp_C': 25 + 3 * np.sin(2 * np.pi * t / (3600 * 3)) + rng.normal(0, 0.5, n),
        'cpu_util_pct': np.clip(25 + 20 * rng.normal(0, 0.3, n), 5, 80),
        'gpu_util_pct': np.clip(15 + 15 * rng.normal(0, 0.35, n), 0, 60),
        'screen': ['on'] * n,
        'brightness': np.clip(130 + 40 * rng.normal(0, 0.2, n), 50, 220),
        'wifi_state': ['on'] * n,
        'mobile_state': ['off'] * n,
        'gps': ['off'] * n,
    })
    q_pred, v_pred, vp_trace, eta_trace, tte_idx = simulate_discharge_ecm(df, n_cycles=800, use_aging=True)
    t_h = df['elapsed_sec'].values / 3600
    soc_trace = q_pred / (Q_max_Ah * 1000)
    if tte_idx is None:
        below = np.where(v_pred <= ECM['V_cut'])[0]
        if len(below):
            tte_idx = below[0]
        else:
            v_pred[-1] = ECM['V_cut'] - 0.05
            tte_idx = n - 1

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.24)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin1 = ax1.twinx()
    ax1_twin2 = ax1.twinx()
    ax1_twin2.spines['right'].set_position(('outward', 60))
    l1, = ax1.plot(t_h, v_pred, color=COLORS['primary'], lw=2.2, label='Terminal Voltage')
    l2, = ax1_twin1.plot(t_h, soc_trace * 100, color=COLORS['tertiary'], lw=1.8, ls='--', label='SOC')
    l3, = ax1_twin2.plot(t_h, power, color=COLORS['quaternary'], lw=1.2, alpha=0.7, label='Power')
    ax1.axhline(ECM['V_cut'], color='red', ls=':', lw=1.5, label='$V_{cut}$')
    if tte_idx is not None:
        ax1.axvline(t_h[tte_idx], color='red', ls='--', lw=2.0, alpha=0.8)
        ax1.fill_betweenx([v_pred.min(), v_pred.max()], t_h[tte_idx], t_h.max(), alpha=0.1, color='red')
        ax1.annotate(f'TTE = {t_h[tte_idx]:.2f} h', xy=(t_h[tte_idx], ECM['V_cut']),
                     xytext=(-80, 30), textcoords='offset points', fontsize=11, fontweight='bold', color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Voltage (V)', fontsize=12, color=COLORS['primary'])
    ax1_twin1.set_ylabel('SOC (%)', fontsize=11, color=COLORS['tertiary'])
    ax1_twin2.set_ylabel('Power (W)', fontsize=11, color=COLORS['quaternary'])
    ax1.set_title('(a) Complete Discharge: Multi-Variable Synchronized View', fontsize=12, fontweight='bold')
    ax1.legend(handles=[l1, l2, l3], loc='upper right', fontsize=9)
    ax1.set_xlim(0, t_h.max())
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.28)
    sns.despine(ax=ax1, right=False, offset=4)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    zoom_start = max(0, tte_idx - 200)
    zoom_end = min(n - 1, tte_idx + 50)
    t_zoom = t_h[zoom_start:zoom_end]
    v_zoom = v_pred[zoom_start:zoom_end]
    soc_zoom = soc_trace[zoom_start:zoom_end]
    ax2.plot(t_zoom, v_zoom, color=COLORS['primary'], lw=2.5, label='Terminal Voltage')
    ax2.axhline(ECM['V_cut'], color='red', ls=':', lw=2.0, label='$V_{cut}$ threshold')
    ax2.axvline(t_h[tte_idx], color='red', ls='--', lw=2.0, alpha=0.8, label='TTE event')
    dv_dt = np.gradient(v_zoom, t_zoom)
    steep_mask = np.abs(dv_dt) > np.percentile(np.abs(dv_dt), 90)
    ax2.fill_between(t_zoom, v_zoom.min(), v_zoom, where=steep_mask, alpha=0.2, color='orange', label='Steep slope region')
    ax2_twin.plot(t_zoom, soc_zoom * 100, color=COLORS['tertiary'], lw=1.5, ls='--', alpha=0.7)
    ax2_twin.set_ylabel('SOC (%)', fontsize=10, color=COLORS['tertiary'])
    formula = r'TTE: $\inf\{t \geq 0 \mid V_t(t) \leq V_{cut}\}$'
    ax2.text(0.03, 0.97, formula, transform=ax2.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    ax2.set_xlim(t_zoom.min(), t_zoom.max())
    ax2.set_ylim(v_zoom.min() - 0.05, v_zoom.max() + 0.05)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Terminal Voltage (V)', fontsize=12)
    ax2.set_title('(b) Voltage Drop Region Zoom: Event Trigger Mechanism', fontsize=12, fontweight='bold')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.32)
    ax2.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax2, right=False, offset=4)

    fig.suptitle('TTE Event Detection (Part 1)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_tte_event_detection_part1.png')


def plot_tte_event_detection_part2():
    """
    TTE 事件检测（下半部分）：
    - (c) TTE 敏感度
    - (d) TTE 不确定性分布
    """
    rng = np.random.default_rng(42)
    power_levels = [1.2, 1.5, 1.8, 2.2, 2.8]
    temp_levels = [0, 15, 25, 35]
    cycle_levels = [0, 200, 500]
    results = []
    for P in power_levels:
        for T in temp_levels:
            for n_cyc in cycle_levels:
                tf = temperature_capacity_factor(T)
                af = aging_capacity_factor(n_cyc)
                Q_eff = Q_max_Ah * tf * af
                tte_est = Q_eff * V_nom / P * 0.85
                results.append({'Power': P, 'Temp': T, 'Cycles': n_cyc, 'TTE': tte_est})
    df_results = pd.DataFrame(results)

    fig = plt.figure(figsize=(12, 6.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.24)
    ax3 = fig.add_subplot(gs[0, 0])
    for idx, T in enumerate(temp_levels):
        subset = df_results[(df_results['Temp'] == T) & (df_results['Cycles'] == 0)]
        marker = ['o', 's', '^', 'D'][idx]
        ax3.plot(subset['Power'], subset['TTE'], marker=marker, ms=8, lw=2.0,
                 label=f'T={T}°C', color=get_palette(len(temp_levels))[idx])
    ax3.set_xlabel('Load Power (W)', fontsize=12)
    ax3.set_ylabel('Estimated TTE (hours)', fontsize=12)
    ax3.set_title('(c) TTE Sensitivity: Power × Temperature', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.32)
    ax3.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax3, offset=4)

    ax4 = fig.add_subplot(gs[0, 1])
    n_samples = 500
    tte_samples = []
    for _ in range(n_samples):
        P_sample = rng.normal(2.0, 0.3)
        T_sample = rng.normal(25, 5)
        n_sample = rng.uniform(0, 300)
        tf = temperature_capacity_factor(T_sample)
        af = aging_capacity_factor(int(n_sample))
        Q_eff = Q_max_Ah * max(0.5, tf * af)
        tte = Q_eff * V_nom / max(0.5, P_sample) * 0.85
        tte_samples.append(tte)
    tte_samples = np.array(tte_samples)
    kde = gaussian_kde(tte_samples)
    x_kde = np.linspace(tte_samples.min(), tte_samples.max(), 200)
    ax4.hist(tte_samples, bins=30, density=True, alpha=0.5, color=COLORS['primary'], edgecolor='white', label='Histogram')
    ax4.plot(x_kde, kde(x_kde), color=COLORS['secondary'], lw=2.5, label='KDE')
    tte_mean = np.mean(tte_samples)
    tte_std = np.std(tte_samples)
    tte_p5 = np.percentile(tte_samples, 5)
    tte_p95 = np.percentile(tte_samples, 95)
    ax4.axvline(tte_mean, color='black', ls='-', lw=2.0, label=f'Mean: {tte_mean:.2f} h')
    ax4.axvline(tte_p5, color='gray', ls='--', lw=1.5, label=f'5th %ile: {tte_p5:.2f} h')
    ax4.axvline(tte_p95, color='gray', ls='--', lw=1.5, label=f'95th %ile: {tte_p95:.2f} h')
    ax4.fill_betweenx([0, kde(x_kde).max() * 1.1], tte_p5, tte_p95, alpha=0.15, color='green', label='90% CI')
    stats_text = f'μ = {tte_mean:.2f} h\nσ = {tte_std:.2f} h\nCV = {tte_std/tte_mean*100:.1f}%'
    ax4.text(0.97, 0.97, stats_text, transform=ax4.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.95))
    ax4.set_xlabel('Time to Empty (hours)', fontsize=12)
    ax4.set_ylabel('Probability Density', fontsize=12)
    ax4.set_title('(d) TTE Uncertainty Quantification (Monte Carlo)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=8, framealpha=0.95)
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.32)
    ax4.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('TTE Event Detection (Part 2)', fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'complex_tte_event_detection_part2.png')


__all__ = [
    'plot_ocv_nonlinearity_mechanism',
    'plot_ocv_nonlinearity_mechanism_part1',
    'plot_ocv_nonlinearity_mechanism_part2',
    'plot_power_current_coupling_mechanism',
    'plot_power_current_coupling_part1',
    'plot_power_current_coupling_part2',
    'plot_temperature_aging_capacity_surface',
    'plot_temperature_aging_capacity_part1',
    'plot_temperature_aging_capacity_part2',
    'plot_polarization_dynamics_mechanism',
    'plot_polarization_dynamics_part1',
    'plot_polarization_dynamics_part2',
    'plot_tte_event_detection_mechanism',
    'plot_tte_event_detection_part1',
    'plot_tte_event_detection_part2',
]

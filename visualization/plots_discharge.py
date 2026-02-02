import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap

from .config import (
    COLORS, DATA, V_nom, predict_power, simulate_discharge, simulate_discharge_ecm,
    ECM, _save_figure, get_palette
)


def plot_long_discharge():
    """Long-discharge curve comparison - Academic quality"""
    scene = DATA / "scene_long_discharge_combined"
    if not scene.exists():
        print("Long-discharge scene not found")
        return

    df = pd.read_csv(scene / "battery_monitor_log.csv")
    df = df.sort_values('elapsed_sec').reset_index(drop=True)

    t_h = df['elapsed_sec'].values / 3600
    q_actual = df['charge_mAh'].values
    q_pred = simulate_discharge(df)

    rmse = np.sqrt(np.mean((q_actual - q_pred) ** 2))
    final_err = abs(q_actual[-1] - q_pred[-1])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2.5, 1]})
    plt.subplots_adjust(hspace=0.08)

    ax1 = axes[0]
    ax1.plot(t_h, q_actual, lw=2.5, color=COLORS['primary'],
             label='Measured charge', zorder=3)
    ax1.plot(t_h, q_pred, lw=2.2, ls='--', color=COLORS['secondary'],
             label='Model prediction', zorder=4)
    ax1.fill_between(t_h, q_actual, q_pred, alpha=0.12, color=COLORS['gray'])

    ax1.set_ylabel('Remaining Charge (mAh)', fontsize=12)
    ax1.set_title(f'Long-Discharge Comparison\n(RMSE = {rmse:.1f} mAh, Final error = {final_err:.1f} mAh)',
                  fontsize=13, pad=8)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.4)
    ax1.grid(True, which='minor', alpha=0.15)
    ax1.set_xlim(0, t_h[-1] * 1.02)
    sns.despine(ax=ax1, offset=5)

    ax2 = axes[1]
    error = q_actual - q_pred
    ax2.plot(t_h, error, lw=1.8, color=COLORS['tertiary'], zorder=3)
    ax2.axhline(0, color='black', ls='-', lw=1.0, zorder=2)
    ax2.fill_between(t_h, 0, error, where=(error >= 0), alpha=0.3,
                     color=COLORS['tertiary'], interpolate=True)
    ax2.fill_between(t_h, 0, error, where=(error < 0), alpha=0.3,
                     color=COLORS['secondary'], interpolate=True)

    ax2.set_xlabel('Time (h)', fontsize=12)
    ax2.set_ylabel('Residual (mAh)', fontsize=12)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.4)
    ax2.grid(True, which='minor', alpha=0.15)
    sns.despine(ax=ax2, offset=5)

    fig.tight_layout()
    _save_figure(fig, 'long_discharge_compare.png')

    print(f"\n=== Long-discharge validation ===")
    print(f"Duration: {t_h[-1]:.2f} h")
    print(f"Initial charge: {q_actual[0]:.1f} mAh")
    print(f"Measured final: {q_actual[-1]:.1f} mAh")
    print(f"Predicted final: {q_pred[-1]:.1f} mAh")
    print(f"RMSE: {rmse:.1f} mAh")
    print(f"Final error: {final_err:.1f} mAh ({100*final_err/q_actual[0]:.2f}%)")


def plot_long_discharge_multivariate():
    """
    长放电多变量视图：功率曲线 + 特征热图
    """
    file_path = DATA / 'scene_long_discharge_combined' / 'battery_monitor_log.csv'
    if not file_path.exists():
        print("Long discharge data not found")
        return
    df = pd.read_csv(file_path)
    if 'elapsed_sec' not in df or 'charge_mAh' not in df:
        print("Required columns missing in long discharge data")
        return

    df['model_power_W'] = df.apply(predict_power, axis=1)
    time_hours = df['elapsed_sec'] / 3600

    # 使用 GridSpec 精确控制布局 - 只有两个子图
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.8], hspace=0.25)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # === (a) 功率曲线 ===
    power_values = df['model_power_W'].values
    p_min, p_max = power_values.min(), power_values.max()
    p_range = p_max - p_min if p_max > p_min else 1.0
    
    # 平滑曲线
    window = max(5, len(df) // 40)
    power_smooth = pd.Series(power_values).rolling(window=window, center=True, min_periods=1).mean()

    ax1.fill_between(time_hours, power_values, p_min, color=COLORS['secondary'], alpha=0.1)
    ax1.plot(time_hours, power_values, color=COLORS['secondary'], lw=0.6, alpha=0.4, label='Raw')
    ax1.plot(time_hours, power_smooth, color=COLORS['secondary'], lw=2.2, label='Smoothed')
    
    # 统计信息
    avg_power = power_values.mean()
    ax1.axhline(avg_power, color=COLORS['tertiary'], ls='--', lw=1.5, 
                label=f'Mean: {avg_power:.2f} W')

    ax1.set_ylabel('Model Power (W)', fontsize=12)
    ax1.set_title('(a) Power Consumption Profile', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=3)
    ax1.set_ylim(max(0, p_min - p_range * 0.1), p_max + p_range * 0.15)
    ax1.set_xlim(time_hours.min(), time_hours.max())
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.3, linestyle='--')
    plt.setp(ax1.get_xticklabels(), visible=False)
    sns.despine(ax=ax1, offset=5)

    # === (b) 特征热图 ===
    feature_cols = ['brightness', 'cpu_util_pct', 'gpu_util_pct', 'temp_C']
    feature_labels = ['Brightness', 'CPU (%)', 'GPU (%)', 'Temp (°C)']

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    heat_data = df[feature_cols].copy()
    # 归一化
    for col in feature_cols:
        col_min, col_max = heat_data[col].min(), heat_data[col].max()
        if col_max > col_min:
            heat_data[col] = (heat_data[col] - col_min) / (col_max - col_min)
        else:
            heat_data[col] = 0.5
    
    heat_normalized = heat_data.T

    # 下采样
    n_bins = min(100, len(df))
    step = max(1, len(df) // n_bins)
    heat_downsampled = heat_normalized.iloc[:, ::step]

    cmap_custom = LinearSegmentedColormap.from_list(
        'custom_ylgnbu', ['#ffffcc', '#a1dab4', '#41b6c4', '#225ea8', '#0c2c84'])

    im = ax2.imshow(heat_downsampled.values, aspect='auto', cmap=cmap_custom, 
                    interpolation='bilinear', vmin=0, vmax=1)

    ax2.set_yticks(range(len(feature_labels)))
    ax2.set_yticklabels(feature_labels, fontsize=11)

    # X轴标签
    time_vals = time_hours.iloc[::step].values
    n_labels = 10
    label_indices = np.linspace(0, len(time_vals) - 1, n_labels, dtype=int)
    ax2.set_xticks(label_indices)
    ax2.set_xticklabels([f'{time_vals[i]:.1f}' for i in label_indices], fontsize=10)

    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_title('(b) Feature Intensity Heatmap (Normalized)', fontsize=13, fontweight='bold')

    # 颜色条
    cbar = fig.colorbar(im, ax=ax2, orientation='vertical', shrink=0.8, aspect=15, pad=0.02)
    cbar.set_label('Normalized Value', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    _save_figure(fig, 'long_discharge_multivariate.png')
    _save_figure(fig, 'long_discharge_multivariate.png')


def plot_long_discharge_ecm_states():
    """
    ECM 增强放电仿真：展示 OCV、极化电压、端电压、效率与 TTE 事件
    四面板高密度视图
    """
    scene = DATA / "scene_long_discharge_combined"
    if not scene.exists():
        print("Long-discharge scene not found for ECM states")
        return

    df = pd.read_csv(scene / "battery_monitor_log.csv")
    df = df.sort_values('elapsed_sec').reset_index(drop=True)

    t_h = df['elapsed_sec'].values / 3600
    q_actual = df['charge_mAh'].values
    v_actual = df['voltage_mV'].values / 1000 if 'voltage_mV' in df.columns else None

    # ECM 增强仿真
    q_pred, v_pred, vp_trace, eta_trace, tte_idx = simulate_discharge_ecm(df, n_cycles=0, use_aging=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) 容量对比
    ax1 = axes[0, 0]
    ax1.plot(t_h, q_actual, lw=2.2, color=COLORS['primary'], label='Measured')
    ax1.plot(t_h, q_pred, lw=2.0, ls='--', color=COLORS['secondary'], label='ECM Model')
    ax1.fill_between(t_h, q_actual, q_pred, alpha=0.1, color=COLORS['gray'])
    rmse = np.sqrt(np.mean((q_actual - q_pred) ** 2))
    ax1.set_ylabel('Charge (mAh)', fontsize=11)
    ax1.set_xlabel('Time (h)', fontsize=11)
    ax1.set_title(f'(a) Capacity: Measured vs ECM (RMSE={rmse:.1f} mAh)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax1, offset=4)

    # (b) 端电压 & 极化电压
    ax2 = axes[0, 1]
    ax2.plot(t_h, v_pred, lw=2.0, color=COLORS['primary'], label='Terminal voltage $V_t$')
    ax2.plot(t_h, vp_trace, lw=1.5, color=COLORS['quaternary'], label='Polarization $v_p$')
    if v_actual is not None:
        ax2.plot(t_h, v_actual, lw=1.2, ls=':', color=COLORS['gray'], alpha=0.7, label='Measured V')
    ax2.axhline(ECM['V_cut'], color=COLORS['secondary'], ls='--', lw=1.2, label=f"$V_{{cut}}$={ECM['V_cut']} V")
    if tte_idx is not None:
        ax2.axvline(t_h[tte_idx], color=COLORS['secondary'], ls=':', lw=1.5, alpha=0.8)
        ax2.scatter([t_h[tte_idx]], [v_pred[tte_idx]], s=60, color=COLORS['secondary'],
                    edgecolor='black', zorder=6, label=f'TTE @ {t_h[tte_idx]:.2f} h')
    ax2.set_ylabel('Voltage (V)', fontsize=11)
    ax2.set_xlabel('Time (h)', fontsize=11)
    ax2.set_title('(b) Terminal & Polarization Voltage', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.35)
    ax2.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax2, offset=4)

    # (c) 效率轨迹
    ax3 = axes[1, 0]
    ax3.plot(t_h, eta_trace * 100, lw=2.0, color=COLORS['tertiary'])
    ax3.axhline(ECM['eta_min'] * 100, color='gray', ls='--', lw=1.2, label=f"$\\eta_{{min}}$={ECM['eta_min']*100:.0f}%")
    ax3.fill_between(t_h, ECM['eta_min'] * 100, eta_trace * 100, alpha=0.15, color=COLORS['tertiary'])
    ax3.set_ylabel('Efficiency η (%)', fontsize=11)
    ax3.set_xlabel('Time (h)', fontsize=11)
    ax3.set_title('(c) Discharge Efficiency', fontsize=12, fontweight='bold')
    ax3.set_ylim(ECM['eta_min'] * 100 - 2, 102)
    ax3.legend(loc='lower left', fontsize=9)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.35)
    ax3.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax3, offset=4)

    # (d) 残差
    ax4 = axes[1, 1]
    residual = q_actual - q_pred
    ax4.plot(t_h, residual, lw=1.0, color=COLORS['primary'])
    ax4.axhline(0, color='black', lw=0.8)
    ax4.fill_between(t_h, 0, residual, where=(residual >= 0), alpha=0.20, color=COLORS['tertiary'])
    ax4.fill_between(t_h, 0, residual, where=(residual < 0), alpha=0.20, color=COLORS['secondary'])
    ax4.set_ylabel('Residual (mAh)', fontsize=11)
    ax4.set_xlabel('Time (h)', fontsize=11)
    ax4.set_title('(d) Capacity Residual (Actual − Model)', fontsize=12, fontweight='bold')
    ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(True, which='major', alpha=0.35)
    ax4.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax4, offset=4)

    fig.suptitle('ECM Enhanced Discharge Simulation (OCV + Thevenin + Efficiency + TTE)',
                 fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, 'long_discharge_ecm_states.png')

    print(f"\n=== ECM Discharge Validation ===")
    print(f"Duration: {t_h[-1]:.2f} h")
    print(f"RMSE: {rmse:.1f} mAh")
    if tte_idx is not None:
        print(f"TTE event detected at: {t_h[tte_idx]:.2f} h (V={v_pred[tte_idx]:.3f} V)")
    else:
        print(f"No TTE event (V > V_cut throughout)")
    print(f"Efficiency range: {eta_trace.min()*100:.1f}% - {eta_trace.max()*100:.1f}%")


__all__ = [
    'plot_long_discharge',
    'plot_long_discharge_multivariate',
    'plot_long_discharge_ecm_states',
]

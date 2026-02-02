import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from .config import COLORS, DATA, V_nom, Q_max_Ah, compute_scene_power, get_palette, _save_figure


def plot_tte_room_temp_comparison():
    """TTE scenario comparison at room temperature."""
    scenes = [
        "scene_baseline_off",
        "scene_brightness_000",
        "scene_brightness_050",
        "scene_brightness_100",
        "scene_brightness_150",
        "scene_brightness_200",
        "scene_brightness_255",
        "scene_cpu_20pct",
        "scene_cpu_60pct",
        "scene_cpu_80pct",
        "scene_gpu_40pct",
        "scene_gpu_80pct",
        "scene_wifi_compare",
        "scene_gps_compare",
        "scene_mobile_compare",
        "scene_synth_high_load",
    ]

    records = []
    for scene_name in scenes:
        scene_dir = DATA / scene_name
        if not scene_dir.exists():
            continue
        df = pd.read_csv(scene_dir / "battery_monitor_log.csv")
        temp_mean = df['temp_C'].mean() if 'temp_C' in df else np.nan
        if not (22 <= temp_mean <= 32):
            continue
        p_meas = compute_scene_power(df)
        soc0 = df['level_pct'].iloc[0] / 100 if 'level_pct' in df else 1.0
        tte_h = soc0 * V_nom * Q_max_Ah / p_meas if p_meas else np.nan
        label = scene_name.replace('scene_', '').replace('_', ' ')
        records.append((label, tte_h, p_meas, temp_mean))

    if not records:
        print("No room-temperature scenes found for TTE comparison")
        return

    df_plot = pd.DataFrame(records, columns=['scene', 'tte_h', 'power_W', 'temp_C'])
    df_plot = df_plot.sort_values('tte_h', ascending=True)

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    colors = get_palette(len(df_plot))
    bars = ax.barh(df_plot['scene'], df_plot['tte_h'], color=colors, edgecolor='black', linewidth=0.8)
    for bar, tte, pwr, temp in zip(bars, df_plot['tte_h'], df_plot['power_W'], df_plot['temp_C']):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f"{tte:.2f} h | {pwr:.2f} W | {temp:.1f}°C",
                va='center', fontsize=8)

    ax.set_xlabel('TTE (hours)', fontsize=11)
    ax.set_ylabel('Scenario', fontsize=11)
    ax.set_title('TTE Scenario Comparison (Room Temperature)', fontsize=13, fontweight='bold')
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='x', which='major', alpha=0.35)
    ax.grid(axis='x', which='minor', alpha=0.12)
    sns.despine(ax=ax, left=True, bottom=False)

    fig.tight_layout()
    _save_figure(fig, 'tte_room_temp_compare.png')


def plot_tte_temperature_comparison():
    """TTE comparison across temperature bins (controlled for power)."""
    data_file = DATA / "MCM2026A题锂电池数据表：master_modeling_table.csv"
    if not data_file.exists():
        print("Master modeling table not found")
        return

    df = pd.read_csv(data_file)
    if 'temp_c' not in df or 't_empty_h_est' not in df:
        print("Required columns for TTE-temperature comparison not found")
        return

    df = df.dropna(subset=['temp_c', 't_empty_h_est', 'P_total_uW'])

    median_power = df['P_total_uW'].median()
    power_lower = median_power * 0.7
    power_upper = median_power * 1.3
    df_filtered = df[(df['P_total_uW'] >= power_lower) & (df['P_total_uW'] <= power_upper)].copy()

    bins = np.arange(np.floor(df_filtered['temp_c'].min() / 3) * 3,
                     np.ceil(df_filtered['temp_c'].max() / 3) * 3 + 3, 3)
    df_filtered['temp_bin'] = pd.cut(df_filtered['temp_c'], bins=bins, include_lowest=True)

    summary = (df_filtered.groupby('temp_bin')
                 .agg(tte_h=('t_empty_h_est', 'mean'),
                      tte_std=('t_empty_h_est', 'std'),
                      power_avg=('P_total_uW', 'mean'),
                      count=('t_empty_h_est', 'size'))
                 .reset_index()
                 .dropna())

    summary['temp_mid'] = summary['temp_bin'].apply(lambda b: (b.left + b.right) / 2)
    summary['tte_se'] = summary['tte_std'] / np.sqrt(summary['count'])
    summary = summary[summary['count'] >= 10].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1 = axes[0]
    color = COLORS['primary']
    ax1.plot(summary['temp_mid'], summary['tte_h'], color=color, lw=2.5, marker='o', markersize=7, zorder=3)
    ax1.fill_between(summary['temp_mid'],
                     summary['tte_h'] - 1.96 * summary['tte_se'],
                     summary['tte_h'] + 1.96 * summary['tte_se'],
                     color=color, alpha=0.2, label='95% CI')

    if len(summary) >= 3:
        z = np.polyfit(summary['temp_mid'], summary['tte_h'], 2)
        x_smooth = np.linspace(summary['temp_mid'].min(), summary['temp_mid'].max(), 100)
        y_smooth = np.polyval(z, x_smooth)
        ax1.plot(x_smooth, y_smooth, color=COLORS['secondary'], lw=1.8, ls='--', alpha=0.7, label='Quadratic fit')

    ax1.set_xlabel('Temperature (°C)', fontsize=11)
    ax1.set_ylabel('Estimated TTE (hours)', fontsize=11)
    ax1.set_title(f'(a) TTE vs Temperature\n(Power: {power_lower/1e6:.1f}-{power_upper/1e6:.1f} W)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, frameon=True)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax1, offset=3)

    ax2 = axes[1]
    colors_bar = get_palette(len(summary))
    bars = ax2.bar(summary['temp_mid'], summary['count'], width=2.5, color=colors_bar,
                   edgecolor='black', linewidth=0.8, alpha=0.85)
    for bar, count, pwr in zip(bars, summary['count'], summary['power_avg']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f'n={int(count)}\n{pwr/1e6:.1f}W', ha='center', va='bottom', fontsize=8)

    ax2.set_xlabel('Temperature (°C)', fontsize=11)
    ax2.set_ylabel('Sample Count', fontsize=11)
    ax2.set_title('(b) Sample Distribution by Temperature', fontsize=12, fontweight='bold')
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(axis='y', which='major', alpha=0.35)
    ax2.grid(axis='y', which='minor', alpha=0.12)
    sns.despine(ax=ax2, offset=3)

    fig.tight_layout()
    _save_figure(fig, 'tte_temperature_compare.png')


def _load_voltage_trace():
    """Try to load a scene with voltage trace; fallback to synthetic if none."""
    scene_dirs = sorted([p for p in DATA.glob('scene_*') if p.is_dir()])
    for scene_dir in scene_dirs:
        csv_file = scene_dir / 'battery_monitor_log.csv'
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        if {'elapsed_sec', 'voltage_mV'}.issubset(df.columns):
            return df.sort_values('elapsed_sec').reset_index(drop=True), scene_dir.name

    # Synthetic fallback: linear decay from 4.15V to 2.9V with mild noise
    t = np.linspace(0, 7200, 240)
    v = 4.15 - 1.25 * (t / t.max()) + 0.02 * np.sin(2 * np.pi * t / t.max() * 3)
    df = pd.DataFrame({'elapsed_sec': t, 'voltage_mV': v * 1000})
    return df, 'synthetic'


def _collect_tte_samples():
    records = []
    scene_dirs = sorted([p for p in DATA.glob('scene_*') if p.is_dir()])
    for scene_dir in scene_dirs:
        csv_file = scene_dir / 'battery_monitor_log.csv'
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        if 'elapsed_sec' not in df or 'charge_mAh' not in df:
            continue
        p_meas = compute_scene_power(df)
        soc0 = df['level_pct'].iloc[0] / 100 if 'level_pct' in df else df['charge_mAh'].iloc[0] / 1000 / Q_max_Ah
        if p_meas <= 0 or soc0 <= 0:
            continue
        tte_h = soc0 * V_nom * Q_max_Ah / p_meas
        temp_mean = df['temp_C'].mean() if 'temp_C' in df else np.nan
        records.append((scene_dir.name, tte_h, p_meas, temp_mean))
    return records


def plot_tte_event_and_hazard():
    """Event detection at V_cut and hazard-style analysis of TTE across scenes."""
    df, scene_name = _load_voltage_trace()
    t_h = df['elapsed_sec'].values / 3600
    v = df['voltage_mV'].values / 1000
    V_cut = 3.0

    # Detect crossing via linear interpolation
    idx = np.where(v <= V_cut)[0]
    if len(idx) == 0:
        t_cross = t_h[-1]
        v_cross = v[-1]
    else:
        k = idx[0]
        if k == 0:
            t_cross = t_h[0]
            v_cross = v[0]
        else:
            t1, t0 = t_h[k], t_h[k-1]
            v1, v0 = v[k], v[k-1]
            frac = (V_cut - v0) / (v1 - v0)
            t_cross = t0 + frac * (t1 - t0)
            v_cross = V_cut

    records = _collect_tte_samples()
    df_tte = pd.DataFrame(records, columns=['scene', 'tte_h', 'power_W', 'temp_C']) if records else pd.DataFrame()
    df_tte = df_tte.dropna(subset=['tte_h']) if not df_tte.empty else df_tte

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8], height_ratios=[1, 1], wspace=0.25, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # (a) Event detection timeline - zoomed on V_cut region
    ax1.plot(t_h, v, color=COLORS['primary'], lw=2.0, label='Voltage trace')
    ax1.axhline(V_cut, color=COLORS['secondary'], ls='--', lw=1.5, label=f'$V_{{cut}}$ = {V_cut:.1f} V')
    ax1.scatter([t_cross], [v_cross], color=COLORS['secondary'], edgecolor='black', zorder=6, s=80,
                marker='X', label=f'TTE = {t_cross:.2f} h')
    ax1.fill_between(t_h, v, V_cut, where=(v <= V_cut), color=COLORS['secondary'], alpha=0.2)
    ax1.set_xlabel('Time (h)', fontsize=11)
    ax1.set_ylabel('Voltage (V)', fontsize=11)
    ax1.set_title('(a) Voltage trajectory & TTE event', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, t_h.max() * 1.02)
    ax1.set_ylim(min(v.min(), V_cut) - 0.1, v.max() + 0.1)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.legend(loc='upper right', fontsize=9)
    sns.despine(ax=ax1, offset=4)

    # (b) TTE vs Power scatter
    if not df_tte.empty:
        ax2.scatter(df_tte['power_W'], df_tte['tte_h'], s=70, c=COLORS['tertiary'],
                    edgecolors='black', linewidths=0.8, alpha=0.85)
        # Fit inverse relationship: TTE ∝ 1/P
        if len(df_tte) > 2:
            p_fit = np.linspace(df_tte['power_W'].min(), df_tte['power_W'].max(), 100)
            k_fit = (df_tte['tte_h'] * df_tte['power_W']).mean()  # E = P * t
            tte_fit = k_fit / p_fit
            ax2.plot(p_fit, tte_fit, color=COLORS['secondary'], lw=2.0, ls='--', 
                     label=f'$TTE \\approx {k_fit:.1f}/P$')
        ax2.set_xlabel('Power (W)', fontsize=11)
        ax2.set_ylabel('TTE (h)', fontsize=11)
        ax2.set_title('(b) TTE vs Power', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.grid(True, which='major', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=11, transform=ax2.transAxes)
    sns.despine(ax=ax2, offset=4)

    # (c) TTE distribution histogram + KDE
    if not df_tte.empty and len(df_tte) > 3:
        sns.histplot(df_tte['tte_h'], ax=ax3, kde=True, color=COLORS['primary'], 
                     edgecolor='white', alpha=0.7, bins=min(15, len(df_tte)))
        ax3.axvline(df_tte['tte_h'].median(), color=COLORS['secondary'], ls='--', lw=2.0,
                    label=f'Median = {df_tte["tte_h"].median():.2f} h')
        ax3.set_xlabel('TTE (h)', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title('(c) TTE distribution', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=11, transform=ax3.transAxes)
    sns.despine(ax=ax3, offset=4)

    # (d) Survival curve (Kaplan-Meier style)
    if not df_tte.empty:
        tte_sorted = np.sort(df_tte['tte_h'].values)
        surv = 1 - np.arange(1, len(tte_sorted) + 1) / len(tte_sorted)
        # Add initial point
        tte_plot = np.concatenate([[0], tte_sorted])
        surv_plot = np.concatenate([[1.0], surv])
        ax4.step(tte_plot, surv_plot, where='post', color=COLORS['tertiary'], lw=2.5)
        ax4.fill_between(tte_plot, surv_plot, step='post', color=COLORS['tertiary'], alpha=0.2)
        # Mark 50% survival
        median_idx = np.searchsorted(1 - surv_plot, 0.5)
        if median_idx < len(tte_plot):
            ax4.axhline(0.5, color='gray', ls=':', lw=1.2)
            ax4.axvline(tte_plot[median_idx], color='gray', ls=':', lw=1.2)
        ax4.set_xlabel('TTE (h)', fontsize=11)
        ax4.set_ylabel('Survival probability', fontsize=11)
        ax4.set_title('(d) Survival curve', fontsize=12, fontweight='bold')
        ax4.set_xlim(0, tte_sorted.max() * 1.1)
        ax4.set_ylim(-0.02, 1.05)
        ax4.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax4.grid(True, which='major', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=11, transform=ax4.transAxes)
    sns.despine(ax=ax4, offset=4)
    sns.despine(ax=ax2, offset=4)

    fig.tight_layout()
    _save_figure(fig, 'tte_event_hazard.png')


def plot_tte_step_sensitivity():
    """
    TTE积分步长敏感性分析 - 专业版
    左图：不同步长下的放电曲线叠加对比
    右图：收敛阶分析（log-log误差图）
    """
    # 电池参数
    capacity_Ah = 2.5
    discharge_rate_A = 1.2
    
    # 理论TTE
    theoretical_tte_h = capacity_Ah / discharge_rate_A
    
    # 不同步长
    step_sizes = [1, 5, 15, 30, 60, 120, 300]
    
    def simulate_discharge_curve(dt_sec, return_curve=False):
        """欧拉积分模拟放电曲线"""
        q_As = capacity_Ah * 3600
        q_initial = q_As
        t_sec = 0.0
        times = [0.0]
        socs = [1.0]
        
        while q_As > 0:
            q_As -= discharge_rate_A * dt_sec
            t_sec += dt_sec
            times.append(t_sec)
            socs.append(max(0, q_As / q_initial))
        
        tte_h = t_sec / 3600
        if return_curve:
            return tte_h, np.array(times) / 3600, np.array(socs)
        return tte_h
    
    # 创建图表
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.28)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # === (a) 放电曲线叠加对比 ===
    # 选择几个代表性步长展示曲线
    display_steps = [1, 30, 120, 300]
    colors_curve = get_palette(len(display_steps))
    
    for i, dt in enumerate(display_steps):
        _, t_curve, soc_curve = simulate_discharge_curve(dt, return_curve=True)
        label = f'dt={dt}s' if dt < 60 else f'dt={dt//60}min'
        ax1.step(t_curve, soc_curve * 100, where='post', lw=2.0, 
                 color=colors_curve[i], alpha=0.85, label=label)
    
    # 理论截止点
    ax1.axvline(theoretical_tte_h, color='black', ls='--', lw=2.0, 
                label=f'Theoretical TTE={theoretical_tte_h:.3f}h')
    ax1.axhline(0, color='gray', ls='-', lw=0.8, alpha=0.5)
    
    # 放大显示TTE附近区域的插图
    axins = ax1.inset_axes([0.55, 0.35, 0.42, 0.45])
    for i, dt in enumerate(display_steps):
        _, t_curve, soc_curve = simulate_discharge_curve(dt, return_curve=True)
        axins.step(t_curve, soc_curve * 100, where='post', lw=1.8, 
                   color=colors_curve[i], alpha=0.85)
    axins.axvline(theoretical_tte_h, color='black', ls='--', lw=1.5)
    axins.set_xlim(theoretical_tte_h - 0.15, theoretical_tte_h + 0.15)
    axins.set_ylim(-5, 15)
    axins.set_xlabel('Time (h)', fontsize=8)
    axins.set_ylabel('SOC (%)', fontsize=8)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.3)
    axins.set_title('Zoom: TTE region', fontsize=9)
    
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('State of Charge (%)', fontsize=12)
    ax1.set_title('(a) Discharge Curves at Different Step Sizes', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, theoretical_tte_h * 1.15)
    ax1.set_ylim(-5, 105)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35, ls='--')
    sns.despine(ax=ax1, offset=5)
    
    # === (b) 收敛阶分析 (log-log) ===
    tte_estimates = [simulate_discharge_curve(dt) for dt in step_sizes]
    errors_abs = [abs(est - theoretical_tte_h) for est in tte_estimates]
    errors_abs = [max(e, 1e-6) for e in errors_abs]  # 避免log(0)
    
    # 绘制误差点
    ax2.loglog(step_sizes, errors_abs, 's-', color=COLORS['secondary'], 
               markersize=10, lw=2.5, markeredgecolor='black', 
               markeredgewidth=1.2, label='Numerical error', zorder=5)
    
    # 拟合收敛阶 (只用大步长点拟合，避免机器精度影响)
    fit_mask = np.array(errors_abs) > 1e-4
    if sum(fit_mask) >= 2:
        log_dt = np.log(np.array(step_sizes)[fit_mask])
        log_err = np.log(np.array(errors_abs)[fit_mask])
        slope, intercept = np.polyfit(log_dt, log_err, 1)
        
        # 绘制拟合线
        dt_fit = np.logspace(np.log10(min(step_sizes)), np.log10(max(step_sizes)), 50)
        err_fit = np.exp(intercept) * dt_fit ** slope
        ax2.loglog(dt_fit, err_fit, '--', color=COLORS['tertiary'], lw=2.0, 
                   alpha=0.8, label=f'Fit: $O(\\Delta t^{{{slope:.2f}}})$')
    
    # 参考线: O(dt)
    dt_ref = np.array([1, 300])
    ax2.loglog(dt_ref, dt_ref * 1e-4, ':', color='gray', lw=1.5, alpha=0.7, label='$O(\\Delta t)$ ref')
    
    # 1%误差阈值线
    threshold_1pct = theoretical_tte_h * 0.01
    ax2.axhline(threshold_1pct, color='red', ls=':', lw=1.5, alpha=0.7, 
                label=f'1% threshold ({threshold_1pct*60:.1f} min)')
    
    ax2.set_xlabel('Integration Step Size (s)', fontsize=12)
    ax2.set_ylabel('Absolute Error (hours)', fontsize=12)
    ax2.set_title('(b) Convergence Order Analysis', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, which='major', alpha=0.35, ls='--')
    ax2.grid(True, which='minor', alpha=0.15)
    sns.despine(ax=ax2, offset=5)
    
    fig.tight_layout()
    _save_figure(fig, 'tte_step_sensitivity.png')
    _save_figure(fig, 'tte_step_sensitivity.png')


__all__ = [
    'plot_tte_room_temp_comparison',
    'plot_tte_temperature_comparison',
    'plot_tte_event_and_hazard',
    'plot_tte_step_sensitivity',
]

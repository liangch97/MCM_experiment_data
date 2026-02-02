from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from .config import (
    COLORS, DATA, P0, P_s0, alpha_s, alpha_c, alpha_gpu, alpha_w, alpha_m, alpha_g,
    _save_figure, get_palette
)


def plot_power_components_comparison():
    """Power model components: 4-panel comparison figure."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ========== Panel (a): Screen Brightness ==========
    ax1 = axes[0, 0]
    levels = [0, 50, 100, 150, 200, 255]
    brightness_data = []
    for lvl in levels:
        scene = DATA / f"scene_brightness_{lvl:03d}"
        if scene.exists():
            df = pd.read_csv(scene / "battery_monitor_log.csv")
            t = df['elapsed_sec'].values
            q = df['charge_mAh'].values
            slope = np.polyfit(t, q, 1)[0]
            current_A = -slope * 3.6
            voltage_V = df['voltage_mV'].mean() / 1000
            power_W = current_A * voltage_V
            brightness_data.append((lvl / 255 * 100, power_W))

    if brightness_data:
        x_br = np.array([d[0] for d in brightness_data])
        y_br = np.array([d[1] for d in brightness_data])
        ax1.scatter(x_br, y_br, s=80, c=COLORS['primary'],
                    edgecolors='black', linewidths=1.0, zorder=5, label='Measured')
        x_fit = np.linspace(0, 100, 100)
        y_fit = P0 + P_s0 + alpha_s * (x_fit / 100)
        ax1.plot(x_fit, y_fit, color=COLORS['secondary'], lw=2.2,
                 label=rf'Fit: $\alpha_s$={alpha_s:.3f} W')
        ax1.fill_between(x_fit, y_fit - 0.02, y_fit + 0.02,
                         color=COLORS['secondary'], alpha=0.12)

    ax1.set_xlabel('Screen Brightness (%)', fontsize=11)
    ax1.set_ylabel('Power (W)', fontsize=11)
    ax1.set_title('(a) Screen Brightness Effect', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(-5, 105)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax1, offset=3)

    # ========== Panel (b): CPU Utilization ==========
    ax2 = axes[0, 1]
    cpu_levels = [5, 20, 40, 60, 80]
    cpu_data = []
    for lvl in cpu_levels:
        scene = DATA / f"scene_cpu_{lvl:02d}pct"
        if scene.exists():
            df = pd.read_csv(scene / "battery_monitor_log.csv")
            t = df['elapsed_sec'].values
            q = df['charge_mAh'].values
            slope = np.polyfit(t, q, 1)[0]
            current_A = -slope * 3.6
            voltage_V = df['voltage_mV'].mean() / 1000
            power_W = current_A * voltage_V
            cpu_util = df['cpu_util_pct'].mean()
            b = df['brightness'].mean() / 255
            cpu_data.append((cpu_util, power_W, b))

    if cpu_data:
        x_cpu = np.array([d[0] for d in cpu_data])
        y_cpu = np.array([d[1] for d in cpu_data])
        b_avg = np.mean([d[2] for d in cpu_data])
        ax2.scatter(x_cpu, y_cpu, s=80, c=COLORS['tertiary'],
                    edgecolors='black', linewidths=1.0, zorder=5, label='Measured')
        x_fit = np.linspace(0, 100, 100)
        y_fit = P0 + P_s0 + alpha_s * b_avg + alpha_c * (x_fit / 100)
        ax2.plot(x_fit, y_fit, color=COLORS['secondary'], lw=2.2,
                 label=rf'Fit: $\alpha_c$={alpha_c:.3f} W')
        ax2.fill_between(x_fit, y_fit - 0.03, y_fit + 0.03,
                         color=COLORS['secondary'], alpha=0.12)

    ax2.set_xlabel('CPU Utilization (%)', fontsize=11)
    ax2.set_ylabel('Power (W)', fontsize=11)
    ax2.set_title('(b) CPU Load Effect', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(-5, 105)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.35)
    ax2.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax2, offset=3)

    # ========== Panel (c): GPU ==========
    ax3 = axes[1, 0]
    gpu_levels = [20, 40, 60, 80]
    gpu_data = []
    for lvl in gpu_levels:
        scene = DATA / f"scene_gpu_{lvl:02d}pct"
        if scene.exists():
            df = pd.read_csv(scene / "battery_monitor_log.csv")
            t = df['elapsed_sec'].values
            q = df['charge_mAh'].values
            slope = np.polyfit(t, q, 1)[0]
            current_A = -slope * 3.6
            voltage_V = df['voltage_mV'].mean() / 1000
            power_W = current_A * voltage_V
            gpu_util = df['gpu_util_pct'].mean()
            b = df['brightness'].mean() / 255
            cpu_load = df['cpu_util_pct'].mean() / 100
            base_power = P0 + P_s0 + alpha_s * b + alpha_c * cpu_load
            residual = power_W - base_power
            gpu_data.append((gpu_util, residual, power_W))

    if gpu_data:
        x_gpu = np.array([d[0] for d in gpu_data])
        y_gpu = np.array([d[1] for d in gpu_data])
        ax3.scatter(x_gpu, y_gpu, s=80, c=COLORS['quaternary'],
                    edgecolors='black', linewidths=1.0, zorder=5, label='Measured')
        x_fit = np.linspace(0, 100, 100)
        y_fit = alpha_gpu * (x_fit / 100)
        ax3.plot(x_fit, y_fit, color=COLORS['secondary'], lw=2.2,
                 label=rf'Fit: $\alpha_{{gpu}}$={alpha_gpu:.3f} W')
        ax3.fill_between(x_fit, y_fit - 0.05, y_fit + 0.05,
                         color=COLORS['secondary'], alpha=0.12)
    else:
        x_fit = np.linspace(0, 100, 100)
        y_fit = alpha_gpu * (x_fit / 100)
        ax3.plot(x_fit, y_fit, color=COLORS['quaternary'], lw=2.5,
                 label=rf'$\alpha_{{gpu}}$={alpha_gpu:.3f} W @ 100%')

    ax3.set_xlabel('GPU Utilization (%)', fontsize=11)
    ax3.set_ylabel('Power Contribution (W)', fontsize=11)
    ax3.set_title('(c) GPU Load Effect', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_xlim(-5, 105)
    ax3.set_ylim(0, alpha_gpu * 1.15)
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.grid(True, which='major', alpha=0.35)
    ax3.grid(True, which='minor', alpha=0.12)
    import seaborn as sns
    sns.despine(ax=ax3, offset=3)

    # ========== Panel (d): Discrete Components ==========
    ax4 = axes[1, 1]
    components = {
        'Baseline\n$P_0$': P0,
        'Screen fixed\n$P_{s,0}$': P_s0,
        'WiFi\n$\\alpha_w$': alpha_w,
        'Cellular\n$\\alpha_m$': alpha_m,
        'GPS\n$\\alpha_g$': alpha_g,
    }
    names = list(components.keys())
    values = list(components.values())
    palette = [COLORS['gray'], COLORS['primary'], COLORS['tertiary'],
               COLORS['quaternary'], COLORS['secondary']]

    bars = ax4.bar(names, values, color=palette, edgecolor='black', linewidth=1.0, width=0.6)
    for bar, val in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='medium')

    ax4.set_ylabel('Power (W)', fontsize=11)
    ax4.set_title('(d) Fixed Power Components', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, max(values) * 1.3)
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax4.grid(axis='y', which='major', alpha=0.35)
    ax4.grid(axis='y', which='minor', alpha=0.12)
    import seaborn as sns
    sns.despine(ax=ax4, bottom=False)
    ax4.tick_params(axis='x', rotation=0)

    fig.suptitle('Power Model Components: Fitted Parameters Comparison',
                 fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, 'power_components_comparison.png')


def plot_parameter_summary():
    """Parameter summary bar chart - Academic quality"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    params = {
        r'$P_0$ (Baseline)': P0,
        r'$P_{s,0}$ (Screen fixed)': P_s0,
        r'$\alpha_s$ (Brightness)': alpha_s,
        r'$\alpha_c$ (CPU)': alpha_c,
        r'$\alpha_{gpu}$ (GPU)': alpha_gpu,
        r'$\alpha_w$ (WiFi)': alpha_w,
        r'$\alpha_m$ (Cellular)': alpha_m,
        r'$\alpha_g$ (GPS)': alpha_g,
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(params.keys())
    values = list(params.values())

    palette = sns.color_palette("viridis", n_colors=len(names))

    bars = ax.barh(names, values, color=palette, edgecolor='black', linewidth=1.0, height=0.7)

    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f} W', va='center', fontsize=10, fontweight='medium')

    ax.set_xlabel('Power Contribution (W)', fontsize=12)
    ax.set_title('Fitted Power Model Parameters', fontsize=14, pad=10)
    ax.set_xlim(0, max(values) * 1.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='x', which='major', alpha=0.4)
    ax.grid(axis='x', which='minor', alpha=0.15)
    sns.despine(ax=ax, left=True, bottom=False)
    ax.tick_params(left=False)

    fig.tight_layout()
    _save_figure(fig, 'parameter_summary.png')


__all__ = [
    'plot_power_components_comparison',
    'plot_parameter_summary',
]

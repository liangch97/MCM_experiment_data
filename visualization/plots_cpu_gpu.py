import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.tri import Triangulation
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import griddata

from .config import COLORS, _save_figure, get_palette


def plot_cpu_gpu_power_surface(summaries):
    """CPU/GPU vs model power: separate 3D surface + 2D analysis figures."""
    df = pd.DataFrame([
        {
            'scene': s['scene_label'],
            'cpu': s['settings']['cpu_raw'],
            'gpu': s['settings']['gpu_raw'],
            'power': s['model_power_W'],
        }
        for s in summaries
    ])
    df = df.dropna(subset=['cpu', 'gpu', 'power'])
    if len(df) < 3:
        print("CPU/GPU data insufficient for power surface plot")
        return

    fig3d = plt.figure(figsize=(8.5, 6.5))
    ax3d = fig3d.add_subplot(111, projection='3d')

    cpu_grid = np.linspace(df['cpu'].min() - 5, df['cpu'].max() + 5, 50)
    gpu_grid = np.linspace(df['gpu'].min() - 5, df['gpu'].max() + 5, 50)
    CPU, GPU = np.meshgrid(cpu_grid, gpu_grid)
    POWER = griddata((df['cpu'].values, df['gpu'].values), df['power'].values,
                     (CPU, GPU), method='cubic', fill_value=np.nan)

    surf = ax3d.plot_surface(CPU, GPU, POWER, cmap='viridis', alpha=0.75,
                             linewidth=0.25, edgecolor='gray', antialiased=True)
    ax3d.scatter(df['cpu'], df['gpu'], df['power'], c=df['power'], cmap='viridis',
                 s=70, edgecolor='black', linewidth=0.7, depthshade=True, zorder=5)

    ax3d.set_xlabel('CPU (%)', fontsize=10, labelpad=10)
    ax3d.set_ylabel('GPU (%)', fontsize=10, labelpad=10)
    ax3d.set_zlabel('Power (W)', fontsize=10, labelpad=10)
    ax3d.set_title('3D Power Surface', fontsize=12, fontweight='bold', pad=12)
    ax3d.view_init(elev=25, azim=135)
    ax3d.tick_params(axis='both', labelsize=8, pad=5)

    cbar3d = fig3d.colorbar(surf, ax=ax3d, shrink=0.75, aspect=18, pad=0.08)
    cbar3d.set_label('Power (W)', fontsize=9)
    cbar3d.ax.tick_params(labelsize=8)

    fig3d.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
    _save_figure(fig3d, 'cpu_gpu_power_surface_3d.png')

    fig2d = plt.figure(figsize=(14, 5.8))
    gs = fig2d.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.30)

    ax1 = fig2d.add_subplot(gs[0, 0])
    tri = Triangulation(df['cpu'], df['gpu'])
    levels = np.linspace(df['power'].min() * 0.95, df['power'].max() * 1.05, 12)
    tcf = ax1.tricontourf(tri, df['power'], levels=levels, cmap='coolwarm', alpha=0.85)
    cs = ax1.tricontour(tri, df['power'], levels=6, colors='white', linewidths=0.8, alpha=0.85)
    ax1.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    ax1.scatter(df['cpu'], df['gpu'], c=df['power'], cmap='coolwarm',
                edgecolor='black', s=80, linewidths=0.8, zorder=5)

    cbar1 = fig2d.colorbar(tcf, ax=ax1, shrink=0.9, aspect=24, pad=0.03)
    cbar1.set_label('Power (W)', fontsize=10)
    cbar1.ax.tick_params(labelsize=9)
    ax1.set_xlabel('CPU Utilization (%)', fontsize=11)
    ax1.set_ylabel('GPU Utilization (%)', fontsize=11)
    ax1.set_title('(a) Power Contour Map', fontsize=12, fontweight='bold', pad=10)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.set_xlim(df['cpu'].min() - 10, df['cpu'].max() + 10)
    ax1.set_ylim(df['gpu'].min() - 10, df['gpu'].max() + 10)
    sns.despine(ax=ax1, offset=4)

    ax2 = fig2d.add_subplot(gs[0, 1])
    df['load_norm'] = np.sqrt(df['cpu']**2 + df['gpu']**2)
    ax2.scatter(df['load_norm'], df['power'], s=85, c=COLORS['tertiary'],
                edgecolor='black', linewidths=0.8, alpha=0.85, zorder=4)

    z = np.polyfit(df['load_norm'], df['power'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['load_norm'].min(), df['load_norm'].max(), 100)
    r2 = np.corrcoef(df['load_norm'], df['power'])[0, 1]**2
    ax2.plot(x_line, p(x_line), '--', color=COLORS['secondary'], lw=2.0,
             label=f'Linear fit (R²={r2:.3f})', zorder=3)

    y_pred = p(df['load_norm'])
    std_err = np.std(df['power'] - y_pred)
    ax2.fill_between(x_line, p(x_line) - 1.96 * std_err, p(x_line) + 1.96 * std_err,
                     alpha=0.15, color=COLORS['secondary'], zorder=2)

    ax2.set_xlabel(r'Load Magnitude $\sqrt{\mathrm{CPU}^2 + \mathrm{GPU}^2}$ (%)', fontsize=10)
    ax2.set_ylabel('Model Power (W)', fontsize=11)
    ax2.set_title('(b) Power vs Combined Load', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.3, linestyle='--')
    sns.despine(ax=ax2, offset=4)

    fig2d.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.18, wspace=0.32)
    _save_figure(fig2d, 'cpu_gpu_power_surface.png')


__all__ = [
    'plot_cpu_gpu_power_surface',
]

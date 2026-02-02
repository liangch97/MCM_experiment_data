import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from .config import (
    COLORS, BASE_PALETTE, DATA, OUTPUT_DIR, V_nom,
    predict_power, compute_scene_power, estimate_scene_model_power,
    get_palette, _save_figure
)


def plot_scene_error_waterfall(summaries):
    """Residual waterfall + distribution, saved as two separate figures."""
    df = pd.DataFrame([
        {
            'scene': s['scene_label'],
            'residual': s['model_power_W'] - s['measured_power_W'],
            'measured': s['measured_power_W'],
            'model': s['model_power_W'],
        }
        for s in summaries
    ])
    if df.empty:
        print("No scenes for residual waterfall")
        return
    df = df.sort_values('residual').reset_index(drop=True)
    df['cumulative'] = df['residual'].cumsum()
    n = len(df)
    x = np.arange(n)
    bar_colors = [COLORS['secondary'] if r > 0 else COLORS['tertiary'] for r in df['residual']]

    # 使用水平柱状图避免标签重叠
    fig1, ax1 = plt.subplots(figsize=(10, max(8, n * 0.4)))
    
    # 简化标签名为编号+简短描述
    short_labels = []
    for i, label in enumerate(df['scene'].tolist()):
        # 提取关键词
        if 'bright' in label.lower():
            short = f"B{i+1}"
        elif 'cpu' in label.lower():
            short = f"C{i+1}"
        elif 'gpu' in label.lower():
            short = f"G{i+1}"
        else:
            short = f"S{i+1}"
        short_labels.append(short)
    
    # 水平柱状图
    bars = ax1.barh(x, df['residual'].values, color=bar_colors, edgecolor='white', linewidth=0.8,
                    alpha=0.85, zorder=3, height=0.7)
    
    # 在柱子旁边标注场景名（缩短版）
    for i, (bar, label) in enumerate(zip(bars, df['scene'].tolist())):
        # 缩短场景名
        short_name = label.replace('brightness ', 'B').replace('cpu ', 'CPU')
        short_name = short_name.replace('gpu ', 'GPU').replace('compare', '').replace('pct', '%')
        short_name = short_name.replace('  ', ' ').strip()
        if len(short_name) > 15:
            short_name = short_name[:13] + '..'
        
        # 标注在柱子右侧或左侧
        val = df['residual'].iloc[i]
        if val >= 0:
            ax1.text(val + 0.02, i, short_name, va='center', ha='left', fontsize=8, alpha=0.85)
        else:
            ax1.text(val - 0.02, i, short_name, va='center', ha='right', fontsize=8, alpha=0.85)
    
    ax1.axvline(0, color='black', lw=1.1, linestyle='-', zorder=5)

    ax1.set_yticks(x)
    ax1.set_yticklabels(short_labels, fontsize=9)
    ax1.set_xlabel('Residual: Model − Measured (W)', fontsize=11)
    ax1.set_ylabel('Scene ID', fontsize=11)
    ax1.set_title('Scene Residual Waterfall', fontsize=13, fontweight='bold', pad=10)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.28, linestyle='--', zorder=0, axis='x')
    sns.despine(ax=ax1, offset=5)

    fig1.tight_layout()
    _save_figure(fig1, 'scene_residual_waterfall.png')

    fig2, ax2 = plt.subplots(figsize=(7.2, 5.5))
    sns.histplot(data=df, x='residual', kde=True, ax=ax2, color=COLORS['primary'],
                 edgecolor='white', linewidth=1.0, alpha=0.75, bins=min(10, max(5, n // 2)))

    mean_resid = df['residual'].mean()
    ax2.axvline(mean_resid, color=COLORS['secondary'], linestyle='--', lw=2.0,
                label=f'Mean: {mean_resid:.3f} W')
    ax2.axvline(0, color='black', linestyle='-', lw=1.1, alpha=0.85)

    stats_text = f'n = {n}\nMAE = {df["residual"].abs().mean():.3f} W'
    ax2.text(0.97, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))

    ax2.set_xlabel('Residual (W)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Residual Distribution', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.3, linestyle='--')
    sns.despine(ax=ax2, offset=5)

    fig2.tight_layout(pad=1.8)
    _save_figure(fig2, 'scene_residual_distribution.png')


def collect_scene_summaries():
    """Collect per-scene summaries from data/scene_* folders with basic derived metrics."""
    scene_dirs = sorted([p for p in DATA.glob('scene_*') if p.is_dir()])
    summaries = []
    for scene_path in scene_dirs:
        csv_file = scene_path / 'battery_monitor_log.csv'
        if not csv_file.exists():
            continue
        df = pd.read_csv(csv_file)
        if 'elapsed_sec' not in df or 'charge_mAh' not in df:
            continue

        df['model_power_W'] = df.apply(predict_power, axis=1)

        dQ_mAh = -df['charge_mAh'].diff()
        dt_h = df['elapsed_sec'].diff() / 3600.0
        power_measured = (dQ_mAh / 1000.0) * V_nom / dt_h
        power_measured = power_measured.replace([np.inf, -np.inf], np.nan)
        measured_power_W = power_measured.median(skipna=True)
        model_power_W = df['model_power_W'].mean()
        model_total_W = model_power_W

        settings = {
            'scene_name': scene_path.name,
            'scene_label': scene_path.name.replace('scene_', '').replace('_', ' '),
            'brightness_raw': df['brightness'].median() if 'brightness' in df else np.nan,
            'cpu_raw': df['cpu_util_pct'].median() if 'cpu_util_pct' in df else np.nan,
            'gpu_raw': df['gpu_util_pct'].median() if 'gpu_util_pct' in df else np.nan,
            'u_s': 1 if ('screen' in df.columns and (df['screen'] == 'on').mean() > 0.5) else 0,
            'u_w': 1 if ('wifi_state' in df.columns and (df['wifi_state'] == 'on').mean() > 0.5) else 0,
            'u_m': 1 if ('mobile_state' in df.columns and (df['mobile_state'] == 'on').mean() > 0.5) else 0,
            'u_g': 1 if ('gps' in df.columns and (df['gps'] == 'on').mean() > 0.5) else 0,
            'temp_C': df['temp_C'].median() if 'temp_C' in df else np.nan,
        }

        summary = {
            'scene_label': settings['scene_label'],
            'scene_name': settings['scene_name'],
            'measured_power_W': measured_power_W,
            'model_power_W': model_power_W,
            'model_total_W': model_total_W,
            'settings': settings,
            'components': {},
            'sensitivity': {},
        }
        summaries.append(summary)

    if not summaries:
        print("No scene_* data found")
    return summaries


def export_scene_summary_csv(summaries):
    rows = []
    for s in summaries:
        row = {
            'scene': s['scene_label'],
            'raw_name': s['scene_name'],
            'measured_power_W': s['measured_power_W'],
            'model_power_W': s['model_power_W'],
            'model_total_W': s['model_total_W'],
            'brightness': s['settings']['brightness_raw'],
            'cpu_util_pct': s['settings']['cpu_raw'],
            'gpu_util_pct': s['settings']['gpu_raw'],
            'screen_on': s['settings']['u_s'],
            'wifi_on': s['settings']['u_w'],
            'mobile_on': s['settings']['u_m'],
            'gps_on': s['settings']['u_g'],
            'temp_C': s['settings']['temp_C'],
        }
        for comp, val in s['components'].items():
            row[f"component::{comp}"] = val
        for k, v in s['sensitivity'].items():
            row[f"sensitivity::{k}"] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'scene_power_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.relative_to(DATA.parent)}")


def plot_scene_power_overview(summaries):
    df = pd.DataFrame([
        {
            'scene': s['scene_label'],
            'measured_power_W': s['measured_power_W'],
            'model_power_W': s['model_power_W'],
        }
        for s in summaries
    ])
    if df.empty:
        print("No scenes found for overview plot")
        return
    df = df.sort_values('measured_power_W', ascending=True).reset_index(drop=True)
    
    # 创建简短标签
    short_labels = []
    for i, label in enumerate(df['scene'].tolist()):
        short = label.replace('brightness ', 'B').replace('cpu ', 'C').replace('gpu ', 'G')
        short = short.replace('compare', '').replace('pct', '%').replace('  ', ' ').strip()
        if len(short) > 12:
            short = short[:10] + '..'
        short_labels.append(short)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.35)))

    ax1 = axes[0]
    bar_width = 0.35
    y_pos = np.arange(len(df))
    ax1.barh(y_pos - bar_width/2, df['measured_power_W'], bar_width,
             color=COLORS['primary'], edgecolor='black', linewidth=0.6, label='Measured')
    ax1.barh(y_pos + bar_width/2, df['model_power_W'], bar_width,
             color=COLORS['secondary'], edgecolor='black', linewidth=0.6, label='Model')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(short_labels, fontsize=9)
    ax1.set_xlabel('Power (W)', fontsize=11)
    ax1.set_title('(a) Scene power: measured vs model', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35, axis='x')
    sns.despine(ax=ax1, offset=4)

    ax2 = axes[1]
    ax2.scatter(df['model_power_W'], df['measured_power_W'], s=70, color=COLORS['tertiary'],
                edgecolors='black', linewidths=0.7, zorder=5, alpha=0.85)
    lim = max(df[['measured_power_W', 'model_power_W']].to_numpy().max(), 1.0) * 1.1
    ax2.plot([0, lim], [0, lim], ls='--', color='black', lw=1.2, label='Ideal y=x', zorder=3)
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Model Power (W)', fontsize=11)
    ax2.set_ylabel('Measured Power (W)', fontsize=11)
    ax2.set_title('(b) Model fidelity per scene', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.35)
    sns.despine(ax=ax2, offset=4)

    fig.tight_layout()
    _save_figure(fig, 'scene_power_overview.png')


def plot_scene_component_stacks(summaries):
    records = []
    for s in summaries:
        for comp, val in s['components'].items():
            records.append({'scene': s['scene_label'], 'component': comp, 'value': val,
                            'measured_power_W': s['measured_power_W']})
    df = pd.DataFrame(records)
    if df.empty:
        print("No component data for stacked bars")
        return
    order = df.groupby('scene')['measured_power_W'].mean().sort_values().index.tolist()
    comp_order = ['Base $P_0$', 'Screen fixed $P_{s,0}$', 'Screen brightness', 'CPU', 'GPU', 'WiFi', 'Cellular', 'GPS']
    color_map = {
        'Base $P_0$': COLORS['gray'],
        'Screen fixed $P_{s,0}$': COLORS['primary'],
        'Screen brightness': BASE_PALETTE[2],
        'CPU': COLORS['secondary'],
        'GPU': COLORS['quaternary'],
        'WiFi': BASE_PALETTE[9],
        'Cellular': BASE_PALETTE[6],
        'GPS': BASE_PALETTE[8],
    }

    pivot = df.pivot_table(index='scene', columns='component', values='value', aggfunc='sum').fillna(0)
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(11, max(6, 0.4 * len(order))))

    cumulative = np.zeros(len(pivot))
    for comp in comp_order:
        if comp not in pivot.columns:
            continue
        values = pivot[comp].values
        ax.barh(pivot.index, values, left=cumulative, color=color_map.get(comp, '#999999'),
                edgecolor='black', linewidth=0.7, label=comp)
        cumulative += values

    ax.set_xlabel('Model Power Contribution (W)')
    ax.set_ylabel('Scene')
    ax.set_title('Per-scene power breakdown (model components)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, ncol=2, frameon=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which='major', alpha=0.35)
    ax.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax, offset=4)

    fig.tight_layout()
    _save_figure(fig, 'scene_power_components.png')


def plot_scene_sensitivity_heatmap(summaries):
    rows = []
    for s in summaries:
        if 'sensitivity' not in s or not s['sensitivity']:
            continue
        row = {'scene': s['scene_label']}
        for k, v in s['sensitivity'].items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty or len(df.columns) <= 1:
        print("No sensitivity data for heatmap")
        return
    df = df.set_index('scene')
    df_abs = df.abs()
    if df_abs.empty or df_abs.size == 0:
        print("Empty sensitivity dataframe, skipping heatmap")
        return
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(df_abs))))
    sns.heatmap(df_abs, annot=True, fmt='.3f', cmap='mako', cbar_kws={'label': 'Absolute sensitivity |S|'}, ax=ax)
    ax.set_title('Parameter sensitivity across all scenes', fontsize=12, fontweight='bold')
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Scene')
    fig.tight_layout()
    _save_figure(fig, 'scene_sensitivity_heatmap.png')


__all__ = [
    'plot_scene_error_waterfall',
    'collect_scene_summaries',
    'export_scene_summary_csv',
    'plot_scene_power_overview',
    'plot_scene_component_stacks',
    'plot_scene_sensitivity_heatmap',
]

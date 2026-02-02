import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import welch
from scipy.ndimage import gaussian_filter

from .config import COLORS, get_palette, _save_figure, BASE_PALETTE, GENERIC_GRADIENT

# Default ECM parameters (synthetic but representative)
ECM_PARAMS = dict(
    Voc=3.95,     # Open-circuit voltage (V)
    R0=0.045,     # Ohmic resistance (Ohm)
    Rct=0.09,     # Charge-transfer resistance (Ohm)
    Cp=2400.0,    # Polarization capacitance (F)
    eta_min=0.88, # Minimum efficiency
    alpha_eta=0.020,  # Efficiency slope vs |I|
    beta_eta=0.004,   # Efficiency slope vs |T-25|
    T_ref=25.0,
)


def _polarization_step_response(params, step_current=2.5, t_end=600, dt=0.5):
    """Simulate Thevenin one-order step response for polarization voltage and terminal voltage."""
    Voc, R0, Rct, Cp = params['Voc'], params['R0'], params['Rct'], params['Cp']
    n = int(t_end / dt) + 1
    t = np.linspace(0, t_end, n)
    vp = np.zeros_like(t)
    vt = np.zeros_like(t)
    for i in range(1, n):
        dvp = -(vp[i-1] / (Rct * Cp)) * dt + (step_current / Cp) * dt
        vp[i] = vp[i-1] + dvp
        vt[i] = Voc - R0 * step_current - vp[i]
    vt[0] = Voc - R0 * step_current - vp[0]
    return t, vp, vt


def _multi_step_response(params, currents, durations, dt=0.5):
    """Simulate response to multiple step current inputs."""
    Voc, R0, Rct, Cp = params['Voc'], params['R0'], params['Rct'], params['Cp']
    tau = Rct * Cp
    
    total_time = sum(durations)
    n = int(total_time / dt) + 1
    t = np.linspace(0, total_time, n)
    vp = np.zeros_like(t)
    vt = np.zeros_like(t)
    I_trace = np.zeros_like(t)
    
    # Build current profile
    t_boundaries = np.cumsum([0] + list(durations))
    for i, time in enumerate(t):
        for j in range(len(currents)):
            if t_boundaries[j] <= time < t_boundaries[j+1]:
                I_trace[i] = currents[j]
                break
    
    # Simulate
    for i in range(1, n):
        dvp = -(vp[i-1] / tau) * dt + (I_trace[i-1] / Cp) * dt
        vp[i] = vp[i-1] + dvp
        vt[i] = Voc - R0 * I_trace[i] - vp[i]
    vt[0] = Voc - R0 * I_trace[0] - vp[0]
    
    return t, vp, vt, I_trace


def plot_ecm_mechanics():
    """Feasible region (power discriminant), efficiency map, and current root geometry."""
    p = ECM_PARAMS
    palette = get_palette(4)

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 0.9], height_ratios=[1, 1.05], wspace=0.22, hspace=0.26)
    ax_feasible = fig.add_subplot(gs[0, 0])
    ax_eff = fig.add_subplot(gs[0, 1])
    ax_root = fig.add_subplot(gs[1, :])

    # (a) Power discriminant / feasible region
    vp_grid = np.linspace(-0.3, 0.3, 120)
    p_req = np.linspace(0, 10, 120)
    VP, PREQ = np.meshgrid(vp_grid, p_req)
    P_max = (p['Voc'] - VP) ** 2 / (4 * p['R0'])
    
    # 显示可行区域
    feasible = PREQ <= P_max
    cf = ax_feasible.contourf(VP, PREQ, feasible.astype(float), levels=[-0.1, 0.5, 1.1],
                              colors=[COLORS['secondary'], COLORS['tertiary']], alpha=0.55)
    # 画P_max边界曲线
    vp_line = np.linspace(-0.3, 0.3, 200)
    p_max_line = (p['Voc'] - vp_line) ** 2 / (4 * p['R0'])
    ax_feasible.plot(vp_line, p_max_line, color='black', lw=2.0, ls='-', label=r'$P_{max}=(V_{oc}-v_p)^2/(4R_0)$')
    ax_feasible.legend(loc='upper right', fontsize=8)
    ax_feasible.axhline(0, color='black', lw=1.0, alpha=0.8)
    ax_feasible.axvline(0, color='black', lw=1.0, alpha=0.8)
    ax_feasible.set_xlim(-0.32, 0.32)
    ax_feasible.set_ylim(0, 10)
    ax_feasible.set_xlabel(r'Polarization voltage $v_p$ (V)', fontsize=11)
    ax_feasible.set_ylabel('Requested power P (W)', fontsize=11)
    ax_feasible.set_title('(a) Feasible power region', fontsize=12, fontweight='bold')
    ax_feasible.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_feasible.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_feasible.grid(True, which='major', alpha=0.35)
    ax_feasible.grid(True, which='minor', alpha=0.12)
    sns.despine(ax=ax_feasible, offset=4)

    # (b) Efficiency map vs |I| and ΔT
    I_axis = np.linspace(-4.5, 4.5, 200)
    dT_axis = np.linspace(-25, 35, 150)
    I, DT = np.meshgrid(I_axis, dT_axis)
    eta = 1 - p['alpha_eta'] * np.abs(I) - p['beta_eta'] * np.abs(DT)
    eta = np.maximum(p['eta_min'], eta)
    cmap = sns.color_palette('crest', as_cmap=True)
    cf2 = ax_eff.contourf(I, DT, eta, levels=np.linspace(p['eta_min'], 1.0, 18), cmap=cmap, alpha=0.95)
    cs2 = ax_eff.contour(I, DT, eta, levels=[p['eta_min'], 0.9, 0.95], colors='k', linewidths=1.0, linestyles='--')
    ax_eff.clabel(cs2, fmt='η=%.2f', fontsize=8)
    cbar = fig.colorbar(cf2, ax=ax_eff, pad=0.015, aspect=26)
    cbar.set_label('Efficiency η', fontsize=10)
    ax_eff.axhline(0, color='gray', ls=':', lw=1.0)
    ax_eff.axvline(0, color='gray', ls=':', lw=1.0)
    ax_eff.set_xlabel('Current I (A)', fontsize=11)
    ax_eff.set_ylabel(r'Temperature offset ΔT (°C)', fontsize=11)
    ax_eff.set_title(r'(b) $\eta$ map: $\eta=\max(\eta_{min},1-\alpha_{\eta}|I|-\beta_{\eta}|\Delta T|)$', fontsize=12, fontweight='bold')
    ax_eff.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_eff.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_eff.grid(True, which='major', alpha=0.32)
    ax_eff.grid(True, which='minor', alpha=0.10)
    sns.despine(ax=ax_eff, offset=4)

    # (c) Current roots vs requested power (geometry of quadratic)
    vp0 = 0.05
    P_axis = np.linspace(0.01, 8.0, 240)  # 只看正功率
    roots_primary = []
    roots_secondary = []
    feasible_mask = []
    for P in P_axis:
        a = p['R0']
        b = - (p['Voc'] - vp0)
        c = P
        disc = b**2 - 4*a*c
        if disc >= 0:
            i1 = ( -b + np.sqrt(disc) ) / (2 * a)
            i2 = ( -b - np.sqrt(disc) ) / (2 * a)
            roots_primary.append(i1)
            roots_secondary.append(i2)
            feasible_mask.append(True)
        else:
            roots_primary.append(np.nan)
            roots_secondary.append(np.nan)
            feasible_mask.append(False)
    roots_primary = np.array(roots_primary)
    roots_secondary = np.array(roots_secondary)
    feasible_mask = np.array(feasible_mask)

    ax_root.plot(P_axis, roots_primary, color=palette[0], lw=2.2, label='Root + (higher I)')
    ax_root.plot(P_axis, roots_secondary, color=palette[1], lw=2.2, label='Root - (lower I)')
    ax_root.fill_between(P_axis, roots_primary, roots_secondary, where=feasible_mask,
                         color=COLORS['tertiary'], alpha=0.15, label='Feasible I band')
    ax_root.axhline(0, color='black', lw=1.0)
    # 计算合理的Y轴范围
    valid_roots = np.concatenate([roots_primary[feasible_mask], roots_secondary[feasible_mask]])
    y_max = np.nanmax(valid_roots) * 1.1 if len(valid_roots) > 0 else 5
    y_min = min(0, np.nanmin(valid_roots) * 1.1) if len(valid_roots) > 0 else -1
    ax_root.set_xlim(0, P_axis.max())
    ax_root.set_ylim(y_min, y_max)
    ax_root.set_xlabel('Requested power P (W)', fontsize=12)
    ax_root.set_ylabel('Current root I (A)', fontsize=12)
    ax_root.set_title(r'(c) Quadratic solution: $P = I(V_{oc} - v_p - R_0 I)$ at $v_p=0.05$ V', fontsize=12, fontweight='bold')
    ax_root.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_root.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_root.grid(True, which='major', alpha=0.32)
    ax_root.grid(True, which='minor', alpha=0.12)
    ax_root.legend(loc='upper left', fontsize=10, ncol=3)
    sns.despine(ax=ax_root, offset=5)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.08)
    fig.suptitle('ECM Mechanics: Feasible Power, Efficiency Clamp, Quadratic Roots', fontsize=14, fontweight='bold')
    _save_figure(fig, 'ecm_mechanics.png')


def plot_ecm_step_response():
    """Step load response: polarization voltage and terminal voltage trajectories."""
    p = ECM_PARAMS
    t, vp, vt = _polarization_step_response(p, step_current=2.5, t_end=600, dt=0.5)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), sharex=True, gridspec_kw={'height_ratios': [1.2, 1]})
    ax1, ax2 = axes

    ax1.plot(t, vt, color=COLORS['primary'], lw=2.2, label='Terminal voltage $V_t$')
    ax1.axhline(p['Voc'], color='gray', ls='--', lw=1.0, label='$V_{oc}$')
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title('Step load response (I = 2.5 A)', fontsize=13, fontweight='bold')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.32)
    ax1.grid(True, which='minor', alpha=0.12)
    ax1.legend(loc='upper right', fontsize=10)
    sns.despine(ax=ax1, offset=4)

    ax2.plot(t, vp, color=COLORS['secondary'], lw=2.0, label='Polarization voltage $v_p$')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('$v_p$ (V)', fontsize=12)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.32)
    ax2.grid(True, which='minor', alpha=0.12)
    ax2.legend(loc='upper right', fontsize=10)
    sns.despine(ax=ax2, offset=4)

    fig.tight_layout()
    _save_figure(fig, 'ecm_step_response.png')


__all__ = [
    'plot_ecm_mechanics',
    'plot_ecm_step_response',
]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from .config import COLORS, _save_figure, get_palette

# OCV coefficients (Oxford pseudo-OCV fit)
OCV_COEF = [4.115648, -0.646723, 0.541718, 0.250783, -0.046021]


def _generate_pseudo_ocv_samples(n=120, noise_std=0.012):
    soc = np.linspace(0.02, 0.98, n)
    a = np.array(OCV_COEF)
    ocv_true = a[0] + a[1] * soc + a[2] * soc**2 + a[3] * np.log(soc) + a[4] * np.log(1 - soc)
    noise = np.random.default_rng(42).normal(0, noise_std, size=n)
    ocv_meas = ocv_true + noise
    return soc, ocv_true, ocv_meas


def plot_ocv_with_residuals():
    """OCV–SOC curve with residual diagnostics (high-density two-panel view)."""
    soc, ocv_true, ocv_meas = _generate_pseudo_ocv_samples()
    residual = ocv_meas - ocv_true
    rmse = np.sqrt(np.mean(residual**2))
    p95 = np.percentile(np.abs(residual), 95)

    palette = get_palette(3)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={'width_ratios': [1.3, 1]})
    ax1, ax2 = axes

    # (a) Main curve with fit and samples
    ax1.plot(soc, ocv_true, color=COLORS['primary'], lw=2.4, label='Model fit')
    ax1.scatter(soc[::4], ocv_meas[::4], s=26, color=palette[1], edgecolor='black', linewidth=0.5,
                alpha=0.75, label='Pseudo measurements')
    ax1.set_xlabel('SOC', fontsize=12)
    ax1.set_ylabel('OCV (V)', fontsize=12)
    ax1.set_title('OCV–SOC Fit (Oxford pseudo-OCV)', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(2.6, 4.4)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.grid(True, which='major', alpha=0.35)
    ax1.grid(True, which='minor', alpha=0.12)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    sns.despine(ax=ax1, offset=4)

    formula = (r"$\mathrm{OCV}(s)=a_0+a_1 s+a_2 s^2+a_3\ln s+a_4\ln(1-s)$" + '\n'
               + f"[a0..a4]=[{OCV_COEF[0]:.3f}, {OCV_COEF[1]:.3f}, {OCV_COEF[2]:.3f}, {OCV_COEF[3]:.3f}, {OCV_COEF[4]:.3f}]")
    ax1.text(0.02, 0.96, formula, transform=ax1.transAxes, fontsize=9.8,
             va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # (b) Residual diagnostics
    ax2.axhline(0, color='black', lw=1.0)
    ax2.scatter(soc, residual * 1000, s=10, color=palette[2], alpha=0.6, label='Residual (mV)')
    sns.kdeplot(residual * 1000, ax=ax2, color=COLORS['secondary'], lw=2.0, fill=True, alpha=0.18,
                label='KDE')
    ax2.axhline(p95 * 1000, color='gray', ls='--', lw=1.1, label=f'P95 = {p95*1000:.1f} mV')
    ax2.axhline(-p95 * 1000, color='gray', ls='--', lw=1.1)
    ax2.set_xlabel('SOC', fontsize=12)
    ax2.set_ylabel('Residual (mV)', fontsize=12)
    ax2.set_title(f'Residuals (RMSE={rmse*1000:.1f} mV, P95={p95*1000:.1f} mV)', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(True, which='major', alpha=0.3)
    ax2.grid(True, which='minor', alpha=0.1)
    ax2.legend(loc='upper right', fontsize=9)
    sns.despine(ax=ax2, offset=4)

    fig.tight_layout()
    _save_figure(fig, 'ocv_fit_residuals.png')


__all__ = [
    'plot_ocv_with_residuals',
    'OCV_COEF',
]

"""Fit OCV-SOC curve using Oxford drive cycle pseudo-OCV data.

The Oxford_Battery_Degradation_Dataset_1.mat file contains pseudo-OCV
discharge segments (OCVdc) for eight cells. We take the earliest cycle
(`cyc0000`) of each cell, convert cumulative charge `q` to SOC, and fit
the standard five-parameter form used in the paper:

    OCV(SOC) = a0 + a1 * s + a2 * s^2 + a3 * ln(s) + a4 * ln(1 - s)

We solve the linear least-squares problem directly (since parameters are
linear in the chosen basis) after clipping SOC away from 0/1 to avoid
log singularities. Per-cell coefficients are averaged for a consensus
curve, and a diagnostic plot is saved to figures/ocv_fit_oxford.png.
"""

from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


DATA_PATH = Path(
    r"d:\美赛\fitting_bundle\论文适用数据集\04_驾驶循环动态放电\Oxford_DriveCycle\Dataset\Oxford Battery Degradation Dataset 1\Oxford_Battery_Degradation_Dataset_1.mat"
)


def load_ocv_discharge(cell_struct):
    """Extract SOC and voltage arrays from the first cycle OCV discharge."""

    ocv = cell_struct.cyc0000.OCVdc
    q = np.asarray(ocv.q, dtype=float)  # cumulative charge, negative during discharge
    v = np.asarray(ocv.v, dtype=float)

    # Convert to SOC in [0, 1]; q is negative and reaches its minimum at full discharge.
    soc = 1.0 + q / abs(q.min())

    # Sort by SOC to make it monotonic for interpolation/fitting.
    order = np.argsort(soc)
    soc = soc[order]
    v = v[order]

    return soc, v


def fit_ocv_curve(soc, voltage, grid=None, eps=1e-4):
    """Fit the 5-parameter OCV model and return grid, fitted voltage, coef."""

    if grid is None:
        grid = np.linspace(0.01, 0.99, 300)

    # Avoid singularities at 0/1 for log terms.
    soc = np.clip(soc, eps, 1 - eps)

    # Interpolate to a regular SOC grid to reduce noise.
    v_interp = np.interp(grid, soc, voltage)

    # Build design matrix for linear least squares.
    X = np.stack(
        [
            np.ones_like(grid),
            grid,
            grid ** 2,
            np.log(grid),
            np.log(1 - grid),
        ],
        axis=1,
    )

    coef, *_ = np.linalg.lstsq(X, v_interp, rcond=None)
    v_fit = X @ coef

    return grid, v_fit, coef


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    mat = sio.loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)

    cell_names = [f"Cell{i}" for i in range(1, 9)]
    coefs = []

    plt.figure(figsize=(8, 5))

    for name in cell_names:
        cell = mat[name]
        soc, v = load_ocv_discharge(cell)
        grid, v_fit, coef = fit_ocv_curve(soc, v)
        coefs.append(coef)

        plt.plot(grid, v_fit, alpha=0.5, lw=1, label=f"{name} fit")

    coefs = np.vstack(coefs)
    coef_mean = coefs.mean(axis=0)

    # Plot averaged curve
    X_grid = np.stack(
        [
            np.ones_like(grid),
            grid,
            grid ** 2,
            np.log(grid),
            np.log(1 - grid),
        ],
        axis=1,
    )
    v_avg = X_grid @ coef_mean
    plt.plot(grid, v_avg, color="k", lw=2.5, label="Mean fit")

    plt.xlabel("SOC")
    plt.ylabel("OCV (V)")
    plt.title("Oxford pseudo-OCV fits (cyc0000 discharge)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)

    out_fig = Path("figures/ocv_fit_oxford.png")
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)

    header = ["a0", "a1", "a2", "a3", "a4"]
    print("Per-cell coefficients (a0 a1 a2 a3 a4):")
    for name, coef in zip(cell_names, coefs):
        print(name, " ".join(f"{c:.6f}" for c in coef))

    print("\nMean coefficients:")
    print(" ".join(f"{c:.6f}" for c in coef_mean))
    print(f"Figure saved to: {out_fig}")


if __name__ == "__main__":
    main()
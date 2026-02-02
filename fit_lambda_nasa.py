"""
从 NASA_BatteryAging 数据集拟合老化系数 λ, β
电池: B0005, B0006, B0007, B0018
策略: 提取每个循环的放电容量，拟合 Q(n)/Q_max = 1 - λ * n^β
输出: 单电池 λ, β 与全体平均；绘制归一化容量衰减曲线。
"""
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ROOT = Path(r"d:\美赛\fitting_bundle\论文适用数据集\01_电池老化与容量衰减\NASA_BatteryAging")
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

def extract_capacity_vs_cycle(mat_path: Path):
    """从 NASA .mat 文件提取放电容量 vs 循环数"""
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    # 找到电池变量
    bat_name = mat_path.stem
    if bat_name in data:
        bat = data[bat_name]
    else:
        for key in data:
            if not key.startswith('_'):
                bat = data[key]
                break
    
    cycles = bat.cycle
    capacities = []
    cycle_nums = []
    
    for i, cyc in enumerate(cycles):
        if hasattr(cyc, 'type') and cyc.type == 'discharge':
            if hasattr(cyc, 'data') and hasattr(cyc.data, 'Capacity'):
                cap = cyc.data.Capacity
                if hasattr(cap, '__len__'):
                    cap = cap[-1] if len(cap) > 0 else np.nan
                if not np.isnan(cap) and cap > 0:
                    capacities.append(cap)
                    cycle_nums.append(len(cycle_nums) + 1)
    
    return np.array(cycle_nums), np.array(capacities)


# 收集所有电池数据
all_results = {}
plt.figure(figsize=(10, 6))

for bat in BATTERIES:
    mat_path = ROOT / f"{bat}.mat"
    if not mat_path.exists():
        print(f"{bat}: file not found")
        continue
    
    try:
        cycles, caps = extract_capacity_vs_cycle(mat_path)
        if len(cycles) < 10:
            print(f"{bat}: insufficient data ({len(cycles)} cycles)")
            continue
        
        Q_max = caps[0]  # 初始容量
        y = caps / Q_max  # 归一化
        
        # 非线性模型: y = 1 - λ * n^β
        loss = 1 - y

        def model(n, lam, beta):
            return lam * np.power(n, beta)

        # 初始值: λ 取线性最小二乘, β=1
        lam_init, *_ = np.linalg.lstsq(cycles.reshape(-1, 1), loss, rcond=None)
        lam_init = float(lam_init[0]) if lam_init.size else 1e-4
        p0 = [lam_init, 1.0]
        bounds = ([0, 0.1], [1e-2, 3.0])  # β 下界 0.1，上界 3

        popt, _ = curve_fit(model, cycles, loss, p0=p0, bounds=bounds, maxfev=10000)
        lambda_fit, beta_fit = popt
        
        all_results[bat] = {
            'Q_max': Q_max,
            'lambda': lambda_fit,
            'beta': beta_fit,
            'cycles': len(cycles),
            'final_cap': caps[-1],
            'fade': (1 - caps[-1]/Q_max) * 100
        }
        
        print(
            f"{bat}: Q_max={Q_max:.3f}Ah, λ={lambda_fit:.2e}/cycle, β={beta_fit:.2f}, "
            f"{len(cycles)} cycles, final={caps[-1]:.3f}Ah ({all_results[bat]['fade']:.1f}% fade)"
        )
        
        # 绘图
        plt.plot(cycles, y, 'o-', markersize=2, label=f"{bat} (λ={lambda_fit:.2e}, β={beta_fit:.2f})")
        
    except Exception as e:
        print(f"{bat}: error - {e}")

# 计算平均 λ
if all_results:
    avg_lambda = np.mean([r['lambda'] for r in all_results.values()])
    avg_beta = np.mean([r['beta'] for r in all_results.values()])
    print(f"\n=== 拟合结果 ===")
    print(f"平均 λ = {avg_lambda:.4e} /cycle")
    print(f"平均 β = {avg_beta:.2f}")
    print(f"对应 500 cycles 衰减(非线性): {(avg_lambda * (500**avg_beta)) * 100:.1f}%")
    
    # 添加拟合线（使用平均 λ, β）
    n_pred = np.arange(0, 250)
    y_pred = 1 - avg_lambda * np.power(n_pred, avg_beta)
    plt.plot(n_pred, y_pred, 'k--', linewidth=2, label=f'Avg: λ={avg_lambda:.2e}, β={avg_beta:.2f}')

plt.xlabel('Cycle Number')
plt.ylabel('Q / Q_initial')
plt.title('Capacity Fade vs Cycles (NASA Battery Aging)')
plt.legend()
plt.grid(True)
plt.xlim(0, None)
plt.ylim(0.6, 1.05)
plt.savefig(r'd:\美赛\fitting_bundle\figures\aging_capacity_fit_nasa.png', dpi=150)
plt.close()
print("图已保存: figures/aging_capacity_fit_nasa.png")

"""
从 Toyota_CycleLife 数据集拟合老化系数 λ
数据来源: Severson et al. 2019 (Stanford/Toyota) 
策略: 提取电池循环寿命数据，拟合 Q(n)/Q_max = 1 - λ*n
"""
import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt

ROOT = Path(r"d:\美赛\fitting_bundle\论文适用数据集\01_电池老化与容量衰减\Toyota_CycleLife")

def load_toyota_batch(mat_path: Path):
    """从 Toyota batch .mat 文件 (v7.3/HDF5) 提取容量 vs 循环数"""
    results = []
    
    with h5py.File(str(mat_path), 'r') as f:
        batch = f['batch']
        
        # summary 是引用数组，每个引用指向一个电池的摘要数据
        summary_refs = np.array(batch['summary']).flatten()
        
        for i, ref in enumerate(summary_refs):
            try:
                summary = f[ref]
                qdis = np.array(summary['QDischarge']).flatten()
                
                # 过滤 NaN
                valid = ~np.isnan(qdis)
                qdis = qdis[valid]
                
                if len(qdis) > 50:
                    cycles = np.arange(1, len(qdis) + 1)
                    results.append((f"Cell_{i+1}", cycles, qdis))
            except Exception as e:
                pass
    
    return results


# 收集数据
all_results = {}
plt.figure(figsize=(12, 6))

mat_files = [
    "2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat",
]

sample_count = 0
max_samples = 20  # 只采样一部分电池用于可视化

for mat_name in mat_files:
    mat_path = ROOT / mat_name
    if not mat_path.exists():
        print(f"File not found: {mat_name}")
        continue
    
    print(f"Processing: {mat_name}")
    cells = load_toyota_batch(mat_path)
    print(f"  Found {len(cells)} cells with sufficient data")
    
    for name, cycles, caps in cells:
        if len(cycles) < 100:
            continue
        
        Q_max = caps[0]
        y = caps / Q_max
        
        # 拟合
        loss = 1 - y
        lambda_fit, *_ = np.linalg.lstsq(cycles.reshape(-1, 1), loss, rcond=None)
        lambda_fit = lambda_fit[0]
        
        all_results[f"{mat_name[:10]}_{name}"] = {
            'Q_max': Q_max,
            'lambda': lambda_fit,
            'cycles': len(cycles),
            'fade': (1 - caps[-1]/Q_max) * 100
        }
        
        if sample_count < max_samples:
            plt.plot(cycles, y, '-', alpha=0.5, linewidth=0.8)
            sample_count += 1

if all_results:
    lambdas = [r['lambda'] for r in all_results.values() if r['lambda'] > 0]
    avg_lambda = np.mean(lambdas)
    median_lambda = np.median(lambdas)
    
    print(f"\n=== Toyota/Stanford 拟合结果 ({len(all_results)} cells) ===")
    print(f"平均 λ = {avg_lambda:.4e} /cycle")
    print(f"中位数 λ = {median_lambda:.4e} /cycle")
    print(f"范围: [{min(lambdas):.2e}, {max(lambdas):.2e}]")
    print(f"对应 500 cycles 衰减 (中位数): {median_lambda * 500 * 100:.1f}%")
    
    # 添加平均拟合线
    n_pred = np.arange(0, 1200)
    y_pred = 1 - median_lambda * n_pred
    plt.plot(n_pred, y_pred, 'r--', linewidth=2, label=f'Median fit: λ={median_lambda:.2e}')
    
    plt.xlabel('Cycle Number')
    plt.ylabel('Q / Q_initial')
    plt.title('Capacity Fade vs Cycles (Toyota/Stanford Dataset)')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1200)
    plt.ylim(0.7, 1.05)
    plt.savefig(r'd:\美赛\fitting_bundle\figures\aging_capacity_fit_toyota.png', dpi=150)
    plt.close()
    print("图已保存: figures/aging_capacity_fit_toyota.png")
else:
    print("No data extracted")

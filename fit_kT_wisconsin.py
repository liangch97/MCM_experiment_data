"""
从 Wisconsin_MultiTemp 数据集拟合温度系数 k_T
温度范围：-20°C, -10°C, 0°C, 10°C, 25°C
策略：提取不同温度下 C/20 放电容量，用 Q(T)/Q_max = 1 - kT*(T-25)^2 拟合
"""
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt

ROOT = Path(r"d:\美赛\fitting_bundle\论文适用数据集\02_不同温度下电池性能\Wisconsin_MultiTemp")

TEMPS = [-20, -10, 0, 10, 25]  # 温度 °C

def find_mat(folder: Path, pattern: str):
    for f in folder.rglob("*.mat"):
        if pattern.lower() in f.name.lower():
            return f
    return None


def get_capacity_from_mat(mat_path: Path):
    """从 .mat 文件读取最大 Ah 放电容量"""
    try:
        data = loadmat(str(mat_path))
        # 尝试读取 'meas' 结构
        if 'meas' in data:
            meas = data['meas']
            if hasattr(meas, 'dtype') and meas.dtype.names and 'Ah' in meas.dtype.names:
                ah = meas['Ah'][0, 0].flatten()
                return abs(ah.min()) if ah.min() < 0 else ah.max()
            elif isinstance(meas, np.ndarray) and meas.shape == (1, 1):
                inner = meas[0, 0]
                if hasattr(inner, 'dtype') and inner.dtype.names and 'Ah' in inner.dtype.names:
                    ah = inner['Ah'].flatten()
                    return abs(ah.min()) if ah.min() < 0 else ah.max()
        # 遍历所有 key
        for key in data:
            if key.startswith('_'):
                continue
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.shape == (1, 1):
                inner = arr[0, 0]
                if hasattr(inner, 'dtype') and inner.dtype.names and 'Ah' in inner.dtype.names:
                    ah = inner['Ah'].flatten()
                    return abs(ah.min()) if ah.min() < 0 else ah.max()
            if hasattr(arr, 'dtype') and arr.dtype.names and 'Ah' in arr.dtype.names:
                ah = arr['Ah'].flatten()
                return abs(ah.min()) if ah.min() < 0 else ah.max()
        return None
    except Exception as e:
        print(f"Error reading {mat_path}: {e}")
        return None


# 收集每个温度下的容量（使用 Drive cycles 文件夹下的 Cycle_1）
temp_cap = []
for t in TEMPS:
    mat = None
    if t == 25:
        # 先尝试 C/20 OCV 文件
        folder = ROOT / "25degC" / "C20 OCV and 1C discharge tests_start_of_tests"
        mat = find_mat(folder, "C20")
    if mat is None:
        # 尝试 Drive cycles 下的 Cycle_1
        folder = ROOT / f"{t}degC" / "Drive cycles"
        if folder.exists():
            mat = find_mat(folder, "Cycle_1")
    if mat is None:
        # 尝试顶层目录
        folder = ROOT / f"{t}degC"
        mat = find_mat(folder, "Cycle_1")
    if mat:
        cap = get_capacity_from_mat(mat)
        if cap:
            temp_cap.append((t, cap))
            print(f"T={t}°C  capacity={cap:.4f} Ah  file={mat.name}")
        else:
            print(f"T={t}°C  cannot read capacity from {mat}")
    else:
        print(f"T={t}°C  no suitable file found")

if len(temp_cap) < 3:
    print("Not enough data points")
    exit(1)

temp_cap = sorted(temp_cap, key=lambda x: x[0])
temps = np.array([x[0] for x in temp_cap])
caps = np.array([x[1] for x in temp_cap])

# Q_max 取 25°C 的容量
Q_25 = caps[temps == 25][0] if 25 in temps else caps.max()
y = caps / Q_25  # 归一化

# 拟合 y = 1 - kT*(T-25)^2
dT2 = (temps - 25.0) ** 2
# y = 1 - kT*dT2 => (1-y) = kT*dT2
loss = 1 - y
kT, *_ = np.linalg.lstsq(dT2.reshape(-1, 1), loss, rcond=None)
kT = kT[0]

print(f"\n=== 拟合结果 ===")
print(f"Q_25 (参考容量) = {Q_25:.4f} Ah")
print(f"k_T = {kT:.6e} /°C²")
print(f"温度范围: {temps.min()}°C ~ {temps.max()}°C")

# 预测
T_pred = np.linspace(-25, 30, 100)
y_pred = 1 - kT * (T_pred - 25) ** 2

plt.figure(figsize=(8, 5))
plt.scatter(temps, y, s=80, c='red', label='Measured')
plt.plot(T_pred, y_pred, 'b-', label=f'Fit: 1 - {kT:.2e}·(T-25)²')
plt.xlabel('Temperature (°C)')
plt.ylabel('Q / Q_25°C')
plt.title('Temperature Effect on Capacity (Wisconsin_MultiTemp)')
plt.legend()
plt.grid(True)
plt.xlim(-25, 30)
plt.ylim(0.4, 1.1)
plt.savefig(r'd:\美赛\fitting_bundle\figures\temperature_capacity_fit.png', dpi=150)
plt.close()
print("图已保存: figures/temperature_capacity_fit.png")

# MCM Experiment Data

美国大学生数学建模竞赛 (MCM) 电池建模与功耗分析项目的实验数据与代码仓库。

## 项目概述

本项目包含智能手机电池放电建模的完整实验数据和分析代码，涵盖：

- **功耗建模**：基于多场景实验数据建立电池功耗模型
- **OCV-SOC 曲线拟合**：使用 Oxford 电池数据集拟合开路电压曲线
- **等效电路模型 (ECM)**：电池动态特性建模
- **剩余时间预测 (TTE)**：基于模型的电池续航预测

## 目录结构

```
.
├── experiment_data/          # 实验数据目录
│   ├── scene_baseline_off/   # 基线场景（屏幕关闭）
│   ├── scene_brightness_*/   # 不同屏幕亮度场景 (0-255)
│   ├── scene_cpu_*/          # 不同 CPU 负载场景 (5%-80%)
│   ├── scene_gpu_*/          # 不同 GPU 负载场景 (20%-80%)
│   ├── scene_wifi_compare/   # WiFi 开关对比
│   ├── scene_mobile_compare/ # 移动网络对比
│   ├── scene_gps_compare/    # GPS 开关对比
│   └── scene_long_discharge_combined/  # 长时间综合放电
│
├── visualization/            # 可视化模块
│   ├── plots_power_model.py  # 功耗模型图表
│   ├── plots_ecm.py          # 等效电路模型图表
│   ├── plots_ocv.py          # OCV 曲线图表
│   ├── plots_discharge.py    # 放电曲线图表
│   └── ...
│
├── figures/                  # 生成的图表输出目录
│
├── fit_ocv_oxford.py         # OCV-SOC 曲线拟合
├── fit_kT_wisconsin.py       # 温度容量模型拟合
├── fit_lambda_nasa.py        # 老化模型拟合 (NASA 数据)
├── fit_lambda_toyota.py      # 老化模型拟合 (Toyota 数据)
├── visualize_results.py      # 可视化主入口
├── battery_full_monitor.ps1  # 电池数据采集脚本
└── paper.tex                 # 论文 LaTeX 源文件
```

## 实验数据格式

每个场景目录下的 `battery_monitor_log.csv` 包含以下字段：

| 字段 | 说明 |
|------|------|
| `timestamp` | 时间戳 |
| `elapsed_sec` | 经过时间（秒） |
| `charge_mAh` | 剩余电量（mAh） |
| `level_pct` | 电量百分比 |
| `voltage_mV` | 电压（mV） |
| `temp_C` | 温度（°C） |
| `screen` | 屏幕状态 |
| `brightness` | 屏幕亮度 |
| `network_type` | 网络类型 |
| `wifi_state` | WiFi 状态 |
| `mobile_state` | 移动网络状态 |
| `gps` | GPS 状态 |
| `cpu_util_pct` | CPU 利用率（%） |
| `gpu_util_pct` | GPU 利用率（%） |

## 使用方法

### 1. 安装依赖

```bash
pip install numpy scipy matplotlib pandas
```

### 2. 运行可视化

```bash
python visualize_results.py
```

生成的图表将保存到 `figures/` 目录。

### 3. 运行模型拟合

```bash
python fit_ocv_oxford.py      # OCV-SOC 曲线拟合
python fit_kT_wisconsin.py    # 温度-容量模型
python fit_lambda_nasa.py     # 老化模型
```

## 核心模型

### OCV-SOC 曲线

$$OCV(SOC) = a_0 + a_1 \cdot s + a_2 \cdot s^2 + a_3 \cdot \ln(s) + a_4 \cdot \ln(1-s)$$

### 功耗模型

$$P_{total} = P_{base} + P_{screen}(B) + P_{CPU}(u_{cpu}) + P_{GPU}(u_{gpu}) + P_{network}$$

## License

MIT License

## 联系方式

如有问题，请通过 GitHub Issues 联系。

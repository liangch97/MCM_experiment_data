# MCM Experiment Data

Experiment data and code repository for battery modeling and power consumption analysis in the Mathematical Contest in Modeling (MCM).

## Overview

This project contains complete experiment data and analysis code for smartphone battery discharge modeling, including:

- **Power Consumption Modeling**: Battery power model based on multi-scenario experiment data
- **OCV-SOC Curve Fitting**: Open-circuit voltage curve fitting using Oxford battery dataset
- **Equivalent Circuit Model (ECM)**: Battery dynamic characteristics modeling
- **Time-to-Empty (TTE) Prediction**: Model-based battery life prediction

## Directory Structure

```
.
├── experiment_data/          # Experiment data directory
│   ├── scene_baseline_off/   # Baseline scenario (screen off)
│   ├── scene_brightness_*/   # Different screen brightness levels (0-255)
│   ├── scene_cpu_*/          # Different CPU load scenarios (5%-80%)
│   ├── scene_gpu_*/          # Different GPU load scenarios (20%-80%)
│   ├── scene_wifi_compare/   # WiFi on/off comparison
│   ├── scene_mobile_compare/ # Mobile network comparison
│   ├── scene_gps_compare/    # GPS on/off comparison
│   └── scene_long_discharge_combined/  # Long-term combined discharge
│
├── visualization/            # Visualization modules
│   ├── plots_power_model.py  # Power model plots
│   ├── plots_ecm.py          # ECM plots
│   ├── plots_ocv.py          # OCV curve plots
│   ├── plots_discharge.py    # Discharge curve plots
│   └── ...
│
├── figures/                  # Generated figures output directory
│
├── fit_ocv_oxford.py         # OCV-SOC curve fitting
├── fit_kT_wisconsin.py       # Temperature-capacity model fitting
├── fit_lambda_nasa.py        # Aging model fitting (NASA data)
├── fit_lambda_toyota.py      # Aging model fitting (Toyota data)
├── visualize_results.py      # Visualization entry point
├── battery_full_monitor.ps1  # Battery data collection script
└── paper.tex                 # Paper LaTeX source file
```

## Experiment Data Format

Each scenario directory contains a `battery_monitor_log.csv` with the following fields:

| Field | Description |
|-------|-------------|
| `timestamp` | Timestamp |
| `elapsed_sec` | Elapsed time (seconds) |
| `charge_mAh` | Remaining charge (mAh) |
| `level_pct` | Battery level percentage |
| `voltage_mV` | Voltage (mV) |
| `temp_C` | Temperature (°C) |
| `screen` | Screen state |
| `brightness` | Screen brightness |
| `network_type` | Network type |
| `wifi_state` | WiFi state |
| `mobile_state` | Mobile network state |
| `gps` | GPS state |
| `cpu_util_pct` | CPU utilization (%) |
| `gpu_util_pct` | GPU utilization (%) |

## Usage

### 1. Install Dependencies

```bash
pip install numpy scipy matplotlib pandas
```

### 2. Run Visualization

```bash
python visualize_results.py
```

Generated figures will be saved to the `figures/` directory.

### 3. Run Model Fitting

```bash
python fit_ocv_oxford.py      # OCV-SOC curve fitting
python fit_kT_wisconsin.py    # Temperature-capacity model
python fit_lambda_nasa.py     # Aging model
```

## Core Models

### OCV-SOC Curve

$$OCV(SOC) = a_0 + a_1 \cdot s + a_2 \cdot s^2 + a_3 \cdot \ln(s) + a_4 \cdot \ln(1-s)$$

### Power Consumption Model

$$P_{total} = P_{base} + P_{screen}(B) + P_{CPU}(u_{cpu}) + P_{GPU}(u_{gpu}) + P_{network}$$

## License

MIT License

## Contact

For questions, please contact via GitHub Issues.

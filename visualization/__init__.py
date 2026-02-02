from . import config
from .config import *
from .plots_power_model import plot_power_components_comparison, plot_parameter_summary
from .plots_discharge import plot_long_discharge, plot_long_discharge_multivariate, plot_long_discharge_ecm_states
from .plots_temperature_aging import (
    plot_temperature_capacity_curve,
    plot_aging_cycle_curve,
    plot_combined_capacity_model,
)
from .plots_ocv import plot_ocv_with_residuals
from .plots_ecm import plot_ecm_mechanics, plot_ecm_step_response
from .plots_scene_analysis import (
    plot_scene_error_waterfall,
    collect_scene_summaries,
    export_scene_summary_csv,
    plot_scene_power_overview,
    plot_scene_component_stacks,
    plot_scene_sensitivity_heatmap,
)
from .plots_cpu_gpu import plot_cpu_gpu_power_surface
from .plots_tte import (
    plot_tte_room_temp_comparison,
    plot_tte_temperature_comparison,
    plot_tte_event_and_hazard,
    plot_tte_step_sensitivity,
)
from .plots_complex_mechanism import (
    plot_ocv_nonlinearity_mechanism,
    plot_power_current_coupling_mechanism,
    plot_temperature_aging_capacity_surface,
    plot_polarization_dynamics_mechanism,
    plot_tte_event_detection_mechanism,
)

__all__ = [
    # Config exports
    *config.__all__,
    # Power model
    'plot_power_components_comparison', 'plot_parameter_summary',
    # Discharge
    'plot_long_discharge', 'plot_long_discharge_multivariate', 'plot_long_discharge_ecm_states',
    # ECM & OCV
    'plot_ecm_mechanics', 'plot_ecm_step_response', 'plot_ocv_with_residuals',
    # Temperature & aging
    'plot_temperature_capacity_curve', 'plot_aging_cycle_curve', 'plot_combined_capacity_model',
    # Scene analysis
    'plot_scene_error_waterfall', 'collect_scene_summaries', 'export_scene_summary_csv',
    'plot_scene_power_overview', 'plot_scene_component_stacks', 'plot_scene_sensitivity_heatmap',
    # CPU/GPU
    'plot_cpu_gpu_power_surface',
    # TTE
    'plot_tte_room_temp_comparison', 'plot_tte_temperature_comparison',
    'plot_tte_event_and_hazard', 'plot_tte_step_sensitivity',
    # Complex mechanism
    'plot_ocv_nonlinearity_mechanism', 'plot_power_current_coupling_mechanism',
    'plot_temperature_aging_capacity_surface', 'plot_polarization_dynamics_mechanism',
    'plot_tte_event_detection_mechanism',
]

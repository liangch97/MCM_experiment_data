"""Visualization entrypoint orchestrating categorized plot modules."""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免弹窗
import matplotlib.pyplot as plt

from visualization import (
    plot_parameter_summary,
    plot_power_components_comparison,
    plot_ecm_mechanics,
    plot_ecm_step_response,
    plot_ocv_with_residuals,
    plot_long_discharge,
    plot_long_discharge_multivariate,
    plot_long_discharge_ecm_states,
    plot_temperature_capacity_curve,
    plot_aging_cycle_curve,
    plot_combined_capacity_model,
    plot_tte_room_temp_comparison,
    plot_tte_event_and_hazard,
    plot_tte_step_sensitivity,
    plot_ocv_nonlinearity_mechanism,
    plot_power_current_coupling_mechanism,
    plot_temperature_aging_capacity_surface,
    plot_polarization_dynamics_mechanism,
    plot_tte_event_detection_mechanism,
    collect_scene_summaries,
    export_scene_summary_csv,
    plot_scene_power_overview,
    plot_scene_component_stacks,
    plot_scene_sensitivity_heatmap,
    plot_scene_error_waterfall,
    plot_cpu_gpu_power_surface,
)


def main():
    print("Generating high-quality academic figures...")

    # Core power model plots
    plot_parameter_summary()
    plot_power_components_comparison()

    # ECM mechanics and step response
    plot_ecm_mechanics()
    plot_ecm_step_response()

    # OCV fit diagnostics
    plot_ocv_with_residuals()

    # Discharge validation (including ECM enhanced view)
    plot_long_discharge()
    plot_long_discharge_multivariate()
    plot_long_discharge_ecm_states()

    # Temperature & aging analyses
    plot_temperature_capacity_curve()
    plot_aging_cycle_curve()
    plot_combined_capacity_model()

    # Scenario comparisons
    plot_tte_room_temp_comparison()
    plot_tte_event_and_hazard()
    plot_tte_step_sensitivity()

    # Complex mechanism evidence (ECM + TTE)
    plot_ocv_nonlinearity_mechanism()
    plot_power_current_coupling_mechanism()
    plot_temperature_aging_capacity_surface()
    plot_polarization_dynamics_mechanism()
    plot_tte_event_detection_mechanism()

    # Scene-level summaries and advanced visuals
    summaries = collect_scene_summaries()
    if summaries:
        export_scene_summary_csv(summaries)
        plot_scene_power_overview(summaries)
        plot_scene_component_stacks(summaries)
        plot_scene_sensitivity_heatmap(summaries)
        plot_scene_error_waterfall(summaries)
        plot_cpu_gpu_power_surface(summaries)
    else:
        print("No scene_* data found for comprehensive per-scene plots")

    print("\nAll figures have been generated!")


if __name__ == "__main__":
    main()

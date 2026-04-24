from __future__ import annotations

HEAD_FEATURE_COLUMNS = [
    "duration_s",
    "n_samples",
    "mean_sampling_rate_hz",
    "head_path_length_m",
    "head_net_displacement_m",
    "head_mean_speed_m_s",
    "head_median_speed_m_s",
    "head_max_speed_m_s",
    "head_std_speed_m_s",
    "head_rms_speed_m_s",
    "head_mean_acc_m_s2",
    "head_max_acc_m_s2",
    "head_rms_acc_m_s2",
    "head_mean_jerk_m_s3",
    "head_max_jerk_m_s3",
    "head_rms_jerk_m_s3",
    "head_stationary_ratio",
    "head_x_range_m",
    "head_y_range_m",
    "head_z_range_m",
    "head_sway_ml_std_m",
    "head_bobbing_vertical_std_m",
    "head_sway_ap_std_m",
    "head_total_angular_displacement_rad",
    "head_mean_angular_speed_rad_s",
    "head_median_angular_speed_rad_s",
    "head_max_angular_speed_rad_s",
    "head_std_angular_speed_rad_s",
    "head_rms_angular_speed_rad_s",
    "head_mean_angular_acc_rad_s2",
    "head_max_angular_acc_rad_s2",
    "head_mean_angular_jerk_rad_s3",
    "head_max_angular_jerk_rad_s3",
    "head_yaw_range_deg",
    "head_pitch_range_deg",
    "head_roll_range_deg",
    "head_mean_yaw_rate_deg_s",
    "head_mean_pitch_rate_deg_s",
    "head_scanpath_length_rad",
    "head_n_turns",
    "head_exploration_entropy",
    "head_exploration_entropy_norm",
    "head_mean_pitch_deg",
    "head_std_pitch_deg",
    "head_downward_pitch_ratio",
    "head_extreme_pitch_ratio",
]

PARTICIPANT_CONTEXT_COLUMNS = ["vr_system_ordinal"]

FEATURE_COLUMNS = HEAD_FEATURE_COLUMNS + PARTICIPANT_CONTEXT_COLUMNS

FEATURE_FAMILIES = {
    "acquisition_session_metadata": [
        "duration_s",
        "n_samples",
        "mean_sampling_rate_hz",
    ],
    "translation": [
        "head_path_length_m",
        "head_net_displacement_m",
        "head_mean_speed_m_s",
        "head_median_speed_m_s",
        "head_max_speed_m_s",
        "head_x_range_m",
        "head_y_range_m",
        "head_z_range_m",
        "head_stationary_ratio",
    ],
    "rotation": [
        "head_total_angular_displacement_rad",
        "head_mean_angular_speed_rad_s",
        "head_median_angular_speed_rad_s",
        "head_max_angular_speed_rad_s",
        "head_yaw_range_deg",
        "head_pitch_range_deg",
        "head_roll_range_deg",
        "head_mean_yaw_rate_deg_s",
        "head_mean_pitch_rate_deg_s",
        "head_scanpath_length_rad",
    ],
    "exploration": [
        "head_n_turns",
        "head_exploration_entropy",
        "head_exploration_entropy_norm",
        "head_scanpath_length_rad",
    ],
    "smoothness_stability": [
        "head_std_speed_m_s",
        "head_rms_speed_m_s",
        "head_mean_acc_m_s2",
        "head_max_acc_m_s2",
        "head_rms_acc_m_s2",
        "head_mean_jerk_m_s3",
        "head_max_jerk_m_s3",
        "head_rms_jerk_m_s3",
        "head_sway_ml_std_m",
        "head_bobbing_vertical_std_m",
        "head_sway_ap_std_m",
        "head_std_angular_speed_rad_s",
        "head_rms_angular_speed_rad_s",
        "head_mean_angular_acc_rad_s2",
        "head_max_angular_acc_rad_s2",
        "head_mean_angular_jerk_rad_s3",
        "head_max_angular_jerk_rad_s3",
    ],
    "posture": [
        "head_mean_pitch_deg",
        "head_std_pitch_deg",
        "head_downward_pitch_ratio",
        "head_extreme_pitch_ratio",
    ],
    "participant_context": [
        "vr_system_ordinal",
    ],
}


def missing_required_columns(columns: list[str] | set[str]) -> list[str]:
    present = set(columns)
    return [column for column in FEATURE_COLUMNS if column not in present]


def available_feature_families() -> list[str]:
    return list(FEATURE_FAMILIES)

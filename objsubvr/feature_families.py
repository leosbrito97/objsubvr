from __future__ import annotations

from collections import OrderedDict

from objsubvr.fast_tracking_ssq_dataset import FEATURE_COLUMNS


FEATURE_FAMILIES: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        (
            "acquisition_session_metadata",
            [
                "duration_s",
                "n_samples",
                "mean_sampling_rate_hz",
            ],
        ),
        (
            "translation",
            [
                "head_path_length_m",
                "head_net_displacement_m",
                "head_mean_speed_m_s",
                "head_median_speed_m_s",
                "head_max_speed_m_s",
                "head_rms_speed_m_s",
                "head_x_range_m",
                "head_y_range_m",
                "head_z_range_m",
            ],
        ),
        (
            "rotation",
            [
                "head_total_angular_displacement_rad",
                "head_mean_angular_speed_rad_s",
                "head_median_angular_speed_rad_s",
                "head_max_angular_speed_rad_s",
                "head_rms_angular_speed_rad_s",
                "head_yaw_range_deg",
                "head_pitch_range_deg",
                "head_roll_range_deg",
                "head_mean_yaw_rate_deg_s",
                "head_mean_pitch_rate_deg_s",
            ],
        ),
        (
            "exploration",
            [
                "head_scanpath_length_rad",
                "head_n_turns",
                "head_exploration_entropy",
                "head_exploration_entropy_norm",
            ],
        ),
        (
            "smoothness_stability",
            [
                "head_std_speed_m_s",
                "head_mean_acc_m_s2",
                "head_max_acc_m_s2",
                "head_rms_acc_m_s2",
                "head_mean_jerk_m_s3",
                "head_max_jerk_m_s3",
                "head_rms_jerk_m_s3",
                "head_stationary_ratio",
                "head_sway_ml_std_m",
                "head_bobbing_vertical_std_m",
                "head_sway_ap_std_m",
                "head_std_angular_speed_rad_s",
                "head_mean_angular_acc_rad_s2",
                "head_max_angular_acc_rad_s2",
                "head_mean_angular_jerk_rad_s3",
                "head_max_angular_jerk_rad_s3",
            ],
        ),
        (
            "posture",
            [
                "head_mean_pitch_deg",
                "head_std_pitch_deg",
                "head_downward_pitch_ratio",
                "head_extreme_pitch_ratio",
            ],
        ),
        (
            "participant_context",
            [
                "vr_system_ordinal",
            ],
        ),
    ]
)


def validate_feature_families(feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS) -> None:
    feature_set = set(feature_columns)
    assigned_features = [feature for features in FEATURE_FAMILIES.values() for feature in features]
    assigned_set = set(assigned_features)

    duplicated = sorted({feature for feature in assigned_features if assigned_features.count(feature) > 1})
    missing_from_dataset = sorted(assigned_set - feature_set)
    uncategorized = sorted(feature_set - assigned_set)

    if duplicated:
        raise ValueError(f"Features duplicadas em FEATURE_FAMILIES: {duplicated}")
    if missing_from_dataset:
        raise ValueError(f"Features categorizadas ausentes de FEATURE_COLUMNS: {missing_from_dataset}")
    if uncategorized:
        raise ValueError(f"Features sem familia em FEATURE_FAMILIES: {uncategorized}")


def features_without_family(
    removed_family: str | None,
    feature_columns: list[str] | tuple[str, ...] = FEATURE_COLUMNS,
) -> list[str]:
    validate_feature_families(feature_columns)
    if removed_family is None:
        return list(feature_columns)
    if removed_family not in FEATURE_FAMILIES:
        raise ValueError(f"Familia invalida: {removed_family}. Use uma de {list(FEATURE_FAMILIES)}.")

    removed = set(FEATURE_FAMILIES[removed_family])
    return [feature for feature in feature_columns if feature not in removed]

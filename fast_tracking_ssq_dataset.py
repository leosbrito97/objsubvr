from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rotation


DEFAULT_SCORES_PATH = Path("FAST-Dataset") / "FAST Scores.xlsx"
DEFAULT_TRACKING_DIR = Path("FAST-Dataset") / "Tracking"

TRACKING_USECOLS = [
    "Timestamp",
    "Head_position_x",
    "Head_position_y",
    "Head_position_z",
    "Head_quat_x",
    "Head_quat_y",
    "Head_quat_z",
    "Head_quat_w",
]

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

VR_SYSTEM_SOURCE_COL = "VRSYSTEM"
VR_SYSTEM_FEATURE_COL = "vr_system_ordinal"
PARTICIPANT_CONTEXT_COLUMNS = [VR_SYSTEM_FEATURE_COL]
FEATURE_COLUMNS = HEAD_FEATURE_COLUMNS + PARTICIPANT_CONTEXT_COLUMNS

TRACKING_FILE_RE = re.compile(r"^(FAB[^_]+)_Build([A-Za-z]+)_.+\.csv$")

QUESTIONNAIRE_TYPES = ("SSQ", "SPES", "TLX", "SUS")

QUESTIONNAIRE_COLUMN_PATTERNS = {
    "SSQ": lambda build: [f"{build}-SSQ{i}" for i in range(1, 17)],
    "SPES": lambda build: [f"{build}-SPES{i}" for i in range(1, 9)],
    "TLX": lambda build: [f"{build}-TLX{i}_1" for i in range(1, 7)],
    "SUS": lambda build: [f"{build}-SUS{i}" for i in range(1, 11)],
}


def ssq_target_columns(build: str) -> list[str]:
    build = build.upper().strip()
    return [f"{build}-SSQ{i}" for i in range(1, 17)]


def questionnaire_target_columns(build: str, questionnaire_type: str) -> list[str]:
    build = build.upper().strip()
    questionnaire_type = questionnaire_type.upper().strip()

    if questionnaire_type not in QUESTIONNAIRE_COLUMN_PATTERNS:
        raise ValueError(
            f"questionnaire_type invalido: {questionnaire_type}. "
            f"Use um de {QUESTIONNAIRE_TYPES}."
        )

    return QUESTIONNAIRE_COLUMN_PATTERNS[questionnaire_type](build)


def _participant_base_key(participant_id: str) -> str:
    participant_id = str(participant_id).strip()
    match = re.match(r"^(FAB\d+)[A-Za-z]$", participant_id)
    return match.group(1) if match else participant_id


def encode_vr_system(value: object) -> int:
    """Encode FAST Scores VRSYSTEM as 0=no VR system, 1=has VR system."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ValueError("VRSYSTEM ausente ou invalido.")

    code = int(numeric)
    if code == 1:
        return 0
    if code == 2:
        return 1
    raise ValueError(f"VRSYSTEM invalido: {value!r}. Esperado 1=nao possui, 2=possui.")


def vr_system_label(encoded_value: int) -> str:
    if int(encoded_value) == 0:
        return "no_vr_system"
    if int(encoded_value) == 1:
        return "has_vr_system"
    return "unknown"


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else math.nan


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else math.nan


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else math.nan


def _safe_rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan


def _finite_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def load_scores_table(scores_path: str | Path = DEFAULT_SCORES_PATH) -> pd.DataFrame:
    scores_df = pd.read_excel(scores_path).iloc[2:].reset_index(drop=True)
    scores_df.columns = [str(col).strip() for col in scores_df.columns]
    if "PID" not in scores_df.columns:
        raise ValueError(f"Coluna 'PID' nao encontrada em {scores_path}.")
    scores_df["PID"] = scores_df["PID"].astype(str).str.strip()
    scores_df = scores_df.loc[scores_df["PID"].ne("") & scores_df["PID"].ne("nan")].reset_index(drop=True)
    return scores_df


def _build_tracking_index(tracking_dir: str | Path, build: str) -> tuple[dict[str, Path], dict[str, Path]]:
    tracking_dir = Path(tracking_dir)
    build = build.upper().strip()

    exact_index: dict[str, Path] = {}
    base_index: dict[str, Path] = {}

    for path in sorted(tracking_dir.glob(f"*_Build{build}_*.csv")):
        match = TRACKING_FILE_RE.match(path.name)
        if not match:
            continue
        tracking_pid = match.group(1).strip()
        exact_index[tracking_pid] = path

        base_key = _participant_base_key(tracking_pid)
        if base_key in base_index and base_index[base_key] != path:
            raise ValueError(f"Mais de um arquivo de tracking encontrado para a chave base {base_key}.")
        base_index[base_key] = path

    if not exact_index:
        raise ValueError(f"Nenhum arquivo Build {build} encontrado em {tracking_dir}.")

    return exact_index, base_index


def _resolve_tracking_path(
    participant_id: str,
    exact_index: dict[str, Path],
    base_index: dict[str, Path],
) -> tuple[Path | None, str]:
    participant_id = str(participant_id).strip()
    if participant_id in exact_index:
        return exact_index[participant_id], "exact"

    base_key = _participant_base_key(participant_id)
    if base_key in base_index:
        return base_index[base_key], "base_key"

    return None, "missing"


def _value_range(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


def _positive_derivative(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if values.size < 2 or timestamps.size != values.size:
        return np.array([], dtype=np.float64)

    dt = np.diff(timestamps)
    valid = dt > 0
    if not np.any(valid):
        return np.array([], dtype=np.float64)

    diffs = np.diff(values)[valid]
    return np.abs(diffs / dt[valid])


def _turn_count_from_yaw(yaw_rad_unwrapped: np.ndarray, min_delta_deg: float = 0.05) -> int:
    if yaw_rad_unwrapped.size < 2:
        return 0

    yaw_delta_deg = np.rad2deg(np.diff(yaw_rad_unwrapped))
    yaw_delta_deg[np.abs(yaw_delta_deg) < min_delta_deg] = 0.0
    direction = np.sign(yaw_delta_deg)
    direction = direction[direction != 0]
    if direction.size < 2:
        return 0
    return int(np.sum(direction[1:] != direction[:-1]))


def _histogram_range(values: np.ndarray) -> list[float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if math.isclose(vmin, vmax):
        return [vmin - 1e-6, vmax + 1e-6]
    return [vmin, vmax]


def extract_head_features_from_tracking(
    tracking_path: str | Path,
    *,
    stationary_speed_threshold_m_s: float = 0.02,
    downward_pitch_threshold_deg: float = -10.0,
    extreme_pitch_threshold_deg: float = 30.0,
    n_turns_threshold_deg: float = 0.05,
    entropy_bins: tuple[int, int] = (24, 20),
) -> dict[str, float]:
    tracking_path = Path(tracking_path)

    tracking_df = pd.read_csv(tracking_path, usecols=TRACKING_USECOLS)
    tracking_df["Timestamp"] = pd.to_datetime(tracking_df["Timestamp"], errors="coerce")
    for col in TRACKING_USECOLS[1:]:
        tracking_df[col] = _finite_series(tracking_df[col])

    tracking_df = tracking_df.dropna(subset=TRACKING_USECOLS).sort_values("Timestamp").reset_index(drop=True)
    if len(tracking_df) < 4:
        raise ValueError(f"Arquivo {tracking_path} nao possui amostras suficientes apos limpeza.")

    timestamps = (
        tracking_df["Timestamp"] - tracking_df["Timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=np.float64)
    duration_s = float(timestamps[-1] - timestamps[0])
    if duration_s <= 0:
        raise ValueError(f"Arquivo {tracking_path} possui duracao nao positiva.")

    n_samples = int(len(tracking_df))
    mean_sampling_rate_hz = float((n_samples - 1) / duration_s)

    position = tracking_df[["Head_position_x", "Head_position_y", "Head_position_z"]].to_numpy(dtype=np.float64)
    quat = tracking_df[["Head_quat_x", "Head_quat_y", "Head_quat_z", "Head_quat_w"]].to_numpy(dtype=np.float64)

    pos_step = np.diff(position, axis=0)
    step_distance_all = np.linalg.norm(pos_step, axis=1)
    dt_all = np.diff(timestamps)
    valid_dt = dt_all > 0

    head_path_length_m = float(np.sum(step_distance_all))
    head_net_displacement_m = float(np.linalg.norm(position[-1] - position[0]))

    speed = step_distance_all[valid_dt] / dt_all[valid_dt]
    speed_times = timestamps[1:][valid_dt]
    acc = _positive_derivative(speed, speed_times)
    acc_times = speed_times[1:]
    jerk = _positive_derivative(acc, acc_times)

    rotation = Rotation.from_quat(quat)
    relative_rotation = rotation[:-1].inv() * rotation[1:]
    angular_step_all = relative_rotation.magnitude()
    angular_speed = angular_step_all[valid_dt] / dt_all[valid_dt]
    angular_speed_times = timestamps[1:][valid_dt]
    angular_acc = _positive_derivative(angular_speed, angular_speed_times)
    angular_acc_times = angular_speed_times[1:]
    angular_jerk = _positive_derivative(angular_acc, angular_acc_times)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected*")
        euler_deg = rotation.as_euler("yxz", degrees=True)

    yaw_deg = euler_deg[:, 0]
    pitch_deg = euler_deg[:, 1]
    roll_deg = euler_deg[:, 2]

    yaw_rad_unwrapped = np.unwrap(np.deg2rad(yaw_deg))
    pitch_rad_unwrapped = np.unwrap(np.deg2rad(pitch_deg))

    yaw_step_deg = np.abs(np.rad2deg(np.diff(yaw_rad_unwrapped)))
    pitch_step_deg = np.abs(np.rad2deg(np.diff(pitch_rad_unwrapped)))
    yaw_rate_deg_s = yaw_step_deg[valid_dt] / dt_all[valid_dt]
    pitch_rate_deg_s = pitch_step_deg[valid_dt] / dt_all[valid_dt]

    scan_step_rad = np.sqrt(np.diff(yaw_rad_unwrapped) ** 2 + np.diff(pitch_rad_unwrapped) ** 2)
    head_scanpath_length_rad = float(np.sum(scan_step_rad))

    histogram, _, _ = np.histogram2d(
        yaw_deg,
        pitch_deg,
        bins=entropy_bins,
        range=[_histogram_range(yaw_deg), _histogram_range(pitch_deg)],
    )
    histogram_prob = histogram.ravel() / histogram.sum()
    histogram_prob = histogram_prob[histogram_prob > 0]
    head_exploration_entropy = float(-np.sum(histogram_prob * np.log(histogram_prob)))
    head_exploration_entropy_norm = float(
        head_exploration_entropy / np.log(entropy_bins[0] * entropy_bins[1])
    )

    features = {
        "duration_s": duration_s,
        "n_samples": n_samples,
        "mean_sampling_rate_hz": mean_sampling_rate_hz,
        "head_path_length_m": head_path_length_m,
        "head_net_displacement_m": head_net_displacement_m,
        "head_mean_speed_m_s": _safe_mean(speed),
        "head_median_speed_m_s": _safe_median(speed),
        "head_max_speed_m_s": _safe_max(speed),
        "head_std_speed_m_s": _safe_std(speed),
        "head_rms_speed_m_s": _safe_rms(speed),
        "head_mean_acc_m_s2": _safe_mean(acc),
        "head_max_acc_m_s2": _safe_max(acc),
        "head_rms_acc_m_s2": _safe_rms(acc),
        "head_mean_jerk_m_s3": _safe_mean(jerk),
        "head_max_jerk_m_s3": _safe_max(jerk),
        "head_rms_jerk_m_s3": _safe_rms(jerk),
        "head_stationary_ratio": float(np.mean(speed < stationary_speed_threshold_m_s)),
        "head_x_range_m": _value_range(position[:, 0]),
        "head_y_range_m": _value_range(position[:, 1]),
        "head_z_range_m": _value_range(position[:, 2]),
        "head_sway_ml_std_m": float(np.std(position[:, 0])),
        "head_bobbing_vertical_std_m": float(np.std(position[:, 1])),
        "head_sway_ap_std_m": float(np.std(position[:, 2])),
        "head_total_angular_displacement_rad": float(np.sum(angular_step_all)),
        "head_mean_angular_speed_rad_s": _safe_mean(angular_speed),
        "head_median_angular_speed_rad_s": _safe_median(angular_speed),
        "head_max_angular_speed_rad_s": _safe_max(angular_speed),
        "head_std_angular_speed_rad_s": _safe_std(angular_speed),
        "head_rms_angular_speed_rad_s": _safe_rms(angular_speed),
        "head_mean_angular_acc_rad_s2": _safe_mean(angular_acc),
        "head_max_angular_acc_rad_s2": _safe_max(angular_acc),
        "head_mean_angular_jerk_rad_s3": _safe_mean(angular_jerk),
        "head_max_angular_jerk_rad_s3": _safe_max(angular_jerk),
        "head_yaw_range_deg": _value_range(yaw_deg),
        "head_pitch_range_deg": _value_range(pitch_deg),
        "head_roll_range_deg": _value_range(roll_deg),
        "head_mean_yaw_rate_deg_s": _safe_mean(yaw_rate_deg_s),
        "head_mean_pitch_rate_deg_s": _safe_mean(pitch_rate_deg_s),
        "head_scanpath_length_rad": head_scanpath_length_rad,
        "head_n_turns": _turn_count_from_yaw(yaw_rad_unwrapped, min_delta_deg=n_turns_threshold_deg),
        "head_exploration_entropy": head_exploration_entropy,
        "head_exploration_entropy_norm": head_exploration_entropy_norm,
        "head_mean_pitch_deg": float(np.mean(pitch_deg)),
        "head_std_pitch_deg": float(np.std(pitch_deg)),
        "head_downward_pitch_ratio": float(np.mean(pitch_deg <= downward_pitch_threshold_deg)),
        "head_extreme_pitch_ratio": float(np.mean(np.abs(pitch_deg) >= extreme_pitch_threshold_deg)),
    }

    return features


def build_headfeatures_questionnaire_dataset(
    build: str,
    questionnaire_type: str,
    *,
    scores_path: str | Path = DEFAULT_SCORES_PATH,
    tracking_dir: str | Path = DEFAULT_TRACKING_DIR,
    n_turns_threshold_deg: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    build = build.upper().strip()
    questionnaire_type = questionnaire_type.upper().strip()
    target_cols = questionnaire_target_columns(build, questionnaire_type)

    scores_df = load_scores_table(scores_path)
    required_cols = ["PID", VR_SYSTEM_SOURCE_COL] + target_cols
    missing_target_cols = [col for col in required_cols if col not in scores_df.columns]
    if missing_target_cols:
        raise ValueError(f"Colunas ausentes em {scores_path}: {missing_target_cols}")

    exact_index, base_index = _build_tracking_index(tracking_dir, build)

    dataset_rows: list[dict[str, float]] = []
    metadata_rows: list[dict[str, str | int]] = []
    skipped_participants: list[str] = []

    for row_idx, score_row in scores_df.iterrows():
        participant_id = str(score_row["PID"]).strip()
        tracking_path, match_mode = _resolve_tracking_path(participant_id, exact_index, base_index)

        if tracking_path is None:
            skipped_participants.append(participant_id)
            continue

        features = extract_head_features_from_tracking(
            tracking_path,
            n_turns_threshold_deg=n_turns_threshold_deg,
        )

        targets = pd.to_numeric(score_row[target_cols], errors="coerce")
        if targets.isna().any():
            raise ValueError(
                f"Participante {participant_id} possui targets ausentes para Build {build} "
                f"e questionario {questionnaire_type}: {targets[targets.isna()].index.tolist()}"
            )
        vr_system_ordinal = encode_vr_system(score_row[VR_SYSTEM_SOURCE_COL])

        dataset_rows.append(
            {
                **features,
                VR_SYSTEM_FEATURE_COL: vr_system_ordinal,
                **{col: int(targets[col]) for col in target_cols},
            }
        )
        metadata_rows.append(
            {
                "row_index": len(dataset_rows) - 1,
                "build": build,
                "questionnaire_type": questionnaire_type,
                "score_pid": participant_id,
                "matched_tracking_pid": tracking_path.name.split("_Build", 1)[0],
                "id_match_mode": match_mode,
                "tracking_file": tracking_path.name,
                "source_score_row": int(row_idx),
                "vr_system_raw": int(pd.to_numeric(score_row[VR_SYSTEM_SOURCE_COL], errors="coerce")),
                VR_SYSTEM_FEATURE_COL: vr_system_ordinal,
                "vr_system_label": vr_system_label(vr_system_ordinal),
            }
        )

    dataset_df = pd.DataFrame(dataset_rows, columns=FEATURE_COLUMNS + target_cols)
    metadata_df = pd.DataFrame(metadata_rows)
    return dataset_df, metadata_df, skipped_participants


def build_headfeatures_ssq_dataset(
    build: str,
    *,
    scores_path: str | Path = DEFAULT_SCORES_PATH,
    tracking_dir: str | Path = DEFAULT_TRACKING_DIR,
    n_turns_threshold_deg: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return build_headfeatures_questionnaire_dataset(
        build=build,
        questionnaire_type="SSQ",
        scores_path=scores_path,
        tracking_dir=tracking_dir,
        n_turns_threshold_deg=n_turns_threshold_deg,
    )

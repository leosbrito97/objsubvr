# Feature-Extraction Parameters

The table below reports the parameter values used to compute the head-tracking features. These values are defined in `fast_tracking_ssq_dataset.py`.

| Parameter | Value | Purpose |
|:--|:--|:--|
| Stationary-speed threshold `tau_s` | `0.02 m/s` | Defines `head_stationary_ratio` as the proportion of samples with translational speed below this threshold. |
| Yaw turn threshold `tau_yaw` | `0.05 deg` | Defines `head_n_turns` by ignoring yaw changes smaller than this value before counting direction changes. |
| Downward-pitch threshold `tau_down` | `-10.0 deg` | Defines `head_downward_pitch_ratio` as the proportion of samples with pitch `<= -10.0 deg`. |
| Extreme-pitch threshold `tau_extreme` | `30.0 deg` | Defines `head_extreme_pitch_ratio` as the proportion of samples with `abs(pitch) >= 30.0 deg`. |
| Yaw-pitch entropy grid `K` | `(24, 20)` bins | Defines the 2D yaw-pitch histogram used for `head_exploration_entropy` and `head_exploration_entropy_norm`. |
| Quaternion-to-Euler convention | `Rotation.from_quat([x, y, z, w])`, then `as_euler("yxz", degrees=True)` | Extracts yaw, pitch, and roll angles from the headset quaternion. |
| Angle unwrapping | `np.unwrap(np.deg2rad(yaw_deg))` and `np.unwrap(np.deg2rad(pitch_deg))` | Removes angular discontinuities before computing yaw/pitch rates and yaw-pitch scanpath length. |
| Filtering/smoothing | No smoothing or low-pass filter applied | Tracking rows are converted to finite numeric values, missing rows are dropped, rows are sorted by timestamp, and non-positive time intervals are ignored for derivative features. |

Note: `tau_yaw = 0.05 deg` is not a yaw-rate threshold in `deg/s`. It is a minimum yaw-angle delta used before turn-event counting in `head_n_turns`.

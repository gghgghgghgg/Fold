from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalRepairConfig:
    velocity_weight: float = 7.0
    translation_velocity_weight: float | None = 20.0
    acceleration_weight: float = 16.0
    outlier_z: float = 8.0
    outlier_weight: float = 0.05
    endpoint_weight: float = 20.0


def _difference_matrix(length: int, order: int) -> np.ndarray:
    return np.diff(np.eye(length, dtype=np.float64), n=order, axis=0)


def _frame_weights(values: np.ndarray, config: TemporalRepairConfig) -> np.ndarray:
    second_difference = np.diff(values, n=2, axis=0)
    frame_score = np.linalg.norm(second_difference, axis=1)
    median = np.median(frame_score)
    scale = 1.4826 * np.median(np.abs(frame_score - median)) + 1e-8
    outlier_frames = np.flatnonzero(frame_score > median + config.outlier_z * scale) + 1

    weights = np.ones(values.shape[0], dtype=np.float64)
    weights[outlier_frames] = config.outlier_weight
    weights[0] = config.endpoint_weight
    weights[-1] = config.endpoint_weight
    return weights


def repair_trajectory(
    values: np.ndarray,
    config: TemporalRepairConfig = TemporalRepairConfig(),
    angular_indices: tuple[int, ...] = (),
    translation_indices: tuple[int, ...] = (),
    lower_limits: np.ndarray | None = None,
    upper_limits: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError(f"Expected a trajectory shaped (T, D) with T >= 3, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Trajectory contains non-finite values")

    working = values.copy()
    if angular_indices:
        working[:, angular_indices] = np.unwrap(working[:, angular_indices], axis=0)

    weights = _frame_weights(working, config)
    first_difference = _difference_matrix(len(working), 1)
    second_difference = _difference_matrix(len(working), 2)
    system = np.diag(weights)
    system += config.velocity_weight * first_difference.T @ first_difference
    system += config.acceleration_weight * second_difference.T @ second_difference
    repaired = np.linalg.solve(system, weights[:, None] * working)

    if translation_indices and config.translation_velocity_weight is not None:
        translation_system = np.diag(weights)
        translation_system += (
            config.translation_velocity_weight
            * first_difference.T
            @ first_difference
        )
        translation_system += (
            config.acceleration_weight
            * second_difference.T
            @ second_difference
        )
        repaired[:, translation_indices] = np.linalg.solve(
            translation_system,
            weights[:, None] * working[:, translation_indices],
        )

    if lower_limits is not None or upper_limits is not None:
        repaired = np.clip(repaired, lower_limits, upper_limits)
    return repaired.astype(np.float32)


def trajectory_metrics(
    values: np.ndarray,
    translation_indices: tuple[int, ...] = (0, 1, 2),
    angular_indices: tuple[int, ...] = (),
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    differences = np.diff(values, axis=0)
    if angular_indices:
        differences[:, angular_indices] = (
            differences[:, angular_indices] + np.pi
        ) % (2 * np.pi) - np.pi

    translation = np.linalg.norm(differences[:, translation_indices], axis=1)
    non_translation = np.delete(differences, translation_indices, axis=1)
    joint_max = np.max(np.abs(non_translation), axis=1)
    joint_l2 = np.linalg.norm(non_translation, axis=1)
    acceleration = np.linalg.norm(np.diff(differences, axis=0), axis=1)
    return {
        "translation_max": float(translation.max()),
        "translation_p95": float(np.percentile(translation, 95)),
        "joint_max": float(joint_max.max()),
        "joint_p95": float(np.percentile(joint_l2, 95)),
        "acceleration_max": float(acceleration.max()),
    }

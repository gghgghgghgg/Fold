import numpy as np

from fold.retargeting.temporal import (
    TemporalRepairConfig,
    repair_trajectory,
    trajectory_metrics,
)


def test_repair_trajectory_reduces_single_frame_spike():
    time = np.linspace(0.0, 1.0, 80)
    trajectory = np.stack([time, np.sin(time), np.cos(time)], axis=1)
    trajectory[40, 1] += 2.0

    repaired = repair_trajectory(trajectory)

    assert np.max(np.abs(np.diff(repaired[:, 1]))) < 0.1
    assert np.max(np.abs(repaired[:, 0] - trajectory[:, 0])) < 0.03


def test_repair_trajectory_unwraps_angles():
    angles = np.linspace(2.8, 3.5, 60)
    wrapped = (angles + np.pi) % (2 * np.pi) - np.pi
    trajectory = np.stack([np.zeros_like(angles), wrapped], axis=1)

    repaired = repair_trajectory(trajectory, angular_indices=(1,))

    assert np.max(np.abs(np.diff(repaired[:, 1]))) < 0.03


def test_translation_velocity_weight_only_affects_selected_columns():
    trajectory = np.zeros((12, 2), dtype=np.float64)
    trajectory[6:, :] = 1.0
    config = TemporalRepairConfig(
        velocity_weight=1.0,
        translation_velocity_weight=20.0,
        acceleration_weight=0.0,
    )

    repaired = repair_trajectory(
        trajectory,
        config=config,
        translation_indices=(0,),
    )

    assert np.max(np.abs(np.diff(repaired[:, 0]))) < np.max(
        np.abs(np.diff(repaired[:, 1]))
    )


def test_trajectory_metrics_reports_translation_and_joint_jumps():
    trajectory = np.zeros((10, 5), dtype=np.float32)
    trajectory[5:, 0] = 0.08
    trajectory[7:, 4] = 0.3

    metrics = trajectory_metrics(trajectory)

    assert np.isclose(metrics["translation_max"], 0.08)
    assert np.isclose(metrics["joint_max"], 0.3)


def test_repair_respects_joint_limits():
    trajectory = np.zeros((20, 2), dtype=np.float32)
    trajectory[10, 1] = 3.0
    config = TemporalRepairConfig(velocity_weight=1.0, acceleration_weight=2.0)

    repaired = repair_trajectory(
        trajectory,
        config=config,
        lower_limits=np.array([-1.0, -0.5]),
        upper_limits=np.array([1.0, 0.5]),
    )

    assert repaired[:, 1].max() <= 0.5
    assert repaired[:, 1].min() >= -0.5

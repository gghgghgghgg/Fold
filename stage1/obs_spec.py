from __future__ import annotations

from typing import Any


ROBOT_FIELD_LAYOUT = (
    ("dof_target_pos", "qpos_target_dim"),
    ("dof_pos", "qpos_dim"),
    ("dof_vel", "qvel_dim"),
    ("kpt_pos", "kpt_dim"),
    ("wrist_pose", "wrist_dim"),
)


def _add_field(
    fields: list[dict[str, Any]],
    *,
    group: str,
    source: str,
    field: str,
    start: int,
    dim: int,
) -> int:
    if dim <= 0:
        return start
    end = start + dim
    fields.append(
        {
            "name": f"{group}_{field}",
            "group": group,
            "source": source,
            "field": field,
            "start": int(start),
            "end": int(end),
            "dim": int(dim),
        }
    )
    return end


def build_stage1_obs_spec(env) -> dict[str, Any]:
    """Recover semantic slices from FoLD's flat policy observation."""

    fields: list[dict[str, Any]] = []
    cursor = 0

    for side in ("left", "right"):
        robot = env.robots[side]
        for field_name, dim_key in ROBOT_FIELD_LAYOUT:
            cursor = _add_field(
                fields,
                group=f"{side}_hand",
                source=side,
                field=field_name,
                start=cursor,
                dim=int(robot.obs_dim_info[dim_key]),
            )

    for obj_name in env.object_names:
        obj = env.objects[obj_name]
        cursor = _add_field(
            fields,
            group="object",
            source=obj_name,
            field="parts_pos",
            start=cursor,
            dim=int(obj.n_links * 3),
        )
        cursor = _add_field(
            fields,
            group="object",
            source=obj_name,
            field="parts_quat",
            start=cursor,
            dim=int(obj.n_links * 4),
        )
        cursor = _add_field(
            fields,
            group="object",
            source=obj_name,
            field="dof_pos",
            start=cursor,
            dim=int(obj.num_joints),
        )
        cursor = _add_field(
            fields,
            group="object",
            source=obj_name,
            field="state_diff",
            start=cursor,
            dim=8,
        )
        cursor = _add_field(
            fields,
            group="object",
            source=obj_name,
            field="root_ang_vel",
            start=cursor,
            dim=3,
        )
        cursor = _add_field(
            fields,
            group="object",
            source=obj_name,
            field="root_lin_vel",
            start=cursor,
            dim=3,
        )

    if getattr(env, "observe_tip_dist", False):
        left_tip_dim = int(env.kpt_dists_left.shape[1] * env.kpt_dists_left.shape[2])
        right_tip_dim = int(env.kpt_dists_right.shape[1] * env.kpt_dists_right.shape[2])
        cursor = _add_field(
            fields,
            group="extras",
            source="environment",
            field="left_tip_dist",
            start=cursor,
            dim=left_tip_dim,
        )
        cursor = _add_field(
            fields,
            group="extras",
            source="environment",
            field="right_tip_dist",
            start=cursor,
            dim=right_tip_dim,
        )

    if getattr(env, "observe_contact_force", False):
        contact_force_dim = int(env.contact_forces.shape[1] * env.contact_forces.shape[2])
        cursor = _add_field(
            fields,
            group="extras",
            source="environment",
            field="contact_force",
            start=cursor,
            dim=contact_force_dim,
        )

    cursor = _add_field(
        fields,
        group="extras",
        source="environment",
        field="episode_progress",
        start=cursor,
        dim=1,
    )

    obs_dim = int(env.obs_dim)
    if cursor != obs_dim:
        raise ValueError(
            f"Stage1 observation spec mismatch: built {cursor} dims but environment reports {obs_dim}."
        )

    groups: dict[str, list[str]] = {}
    for field in fields:
        groups.setdefault(field["group"], []).append(field["name"])

    return {
        "obs_dim": obs_dim,
        "fields": fields,
        "groups": groups,
    }

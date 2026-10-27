from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_DIR = ROOT / "global_anchor_force_field"
if str(GLOBAL_DIR) not in sys.path:
    sys.path.insert(0, str(GLOBAL_DIR))

from one_frame_global_anchor_fit import (
    DynState,
    aa_to_rotmat,
    desired_qddot,
    simple_dynamics_terms,
    solve_normal_anchor_qp,
    target_qdot_between,
)
from rollout_global_anchor_force import (
    _tangent_basis,
    build_friction_pyramid_wrench_matrix,
    build_normal_wrench_matrix,
    load_cached_anchors,
    select_topk_anchors_by_score,
    solve_friction_pyramid_qp,
    subset_edges,
    transform_cached_anchors,
)


def load_urdf_link_masses(path: str | Path) -> tuple[float, float]:
    root = ET.parse(path).getroot()
    masses = {}
    for link in root.findall("link"):
        mass = link.find("./inertial/mass")
        if mass is not None:
            masses[link.attrib["name"]] = float(mass.attrib["value"])
    missing = [name for name in ("bottom", "top") if name not in masses]
    if missing:
        raise ValueError(f"URDF {path} is missing inertial masses for {missing}")
    return masses["bottom"], masses["top"]


def solve_gt_frame(
    target_q: np.ndarray,
    frame: int,
    dt: float,
    anchor_model,
    mass_terms: tuple[np.ndarray, np.ndarray, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray | float | int]:
    curr = DynState(
        q=target_q[frame].copy(),
        qdot=target_qdot_between(target_q, max(frame - 1, 0), frame, dt),
        qddot=np.zeros(7, dtype=np.float64),
    )
    target_next = DynState(
        q=target_q[frame + 1].copy(),
        qdot=target_qdot_between(target_q, frame, frame + 1, dt),
        qddot=np.zeros(7, dtype=np.float64),
    )
    M, h, tau_known = mass_terms
    tau = M @ desired_qddot(curr, target_next, dt) + h - tau_known

    p_rel, normal_dirs, part_ids = transform_cached_anchors(anchor_model, curr.q)
    A_normal = build_normal_wrench_matrix(p_rel, normal_dirs, part_ids, curr.q)
    if int(args.active_top_k) > 0 and int(args.active_top_k) < A_normal.shape[1] and args.topk_method == "normal_prefit":
        prefit_lambda, _prefit_bias, _prefit_achieved = solve_normal_anchor_qp(
            A_normal,
            tau,
            anchor_model.edges,
            anchor_model.lambda_upper,
            rot_w=float(args.rot_w),
            hinge_w=float(args.hinge_w),
            reg=float(args.reg),
            smooth_w=float(args.smooth_w),
            bias_reg=1e6,
            bias_bound=0.0,
        )
        selected = np.argsort(prefit_lambda)[-int(args.active_top_k) :][::-1].astype(np.int64)
    else:
        selected = select_topk_anchors_by_score(
            A_normal,
            tau,
            int(args.active_top_k),
            rot_w=float(args.rot_w),
            hinge_w=float(args.hinge_w),
        )
    A, t1, t2 = build_friction_pyramid_wrench_matrix(
        p_rel[selected], normal_dirs[selected], part_ids[selected], curr.q
    )
    coeff_flat, achieved = solve_friction_pyramid_qp(
        A,
        tau,
        subset_edges(anchor_model.edges, selected),
        anchor_model.lambda_upper[selected],
        mu=float(args.mu),
        rot_w=float(args.rot_w),
        hinge_w=float(args.hinge_w),
        reg=float(args.reg),
        smooth_w=float(args.smooth_w),
        tangent_reg_scale=float(args.tangent_reg_scale),
    )
    coeff_selected = coeff_flat.reshape(-1, 5)
    coeff = np.zeros((len(anchor_model.points), 5), dtype=np.float64)
    coeff[selected] = coeff_selected

    force_selected = (
        coeff_selected[:, 0:1] * normal_dirs[selected]
        + (coeff_selected[:, 1:2] - coeff_selected[:, 2:3]) * t1
        + (coeff_selected[:, 3:4] - coeff_selected[:, 4:5]) * t2
    )
    force_world = np.zeros((len(anchor_model.points), 3), dtype=np.float64)
    force_world[selected] = force_selected

    R = aa_to_rotmat(curr.q[3:6])
    force_base = (R.T @ force_world.T).T
    p_base = (R.T @ p_rel.T).T
    normal_base = (R.T @ normal_dirs.T).T
    generalized_base = achieved.copy()
    generalized_base[:3] = R.T @ achieved[:3]
    generalized_base[3:6] = R.T @ achieved[3:6]
    residual = tau - achieved
    return {
        "tau": tau,
        "achieved_world": achieved,
        "achieved_base": generalized_base,
        "residual": residual,
        "rel_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(tau), 1e-9)),
        "coeff": coeff,
        "anchor_force_world": force_world,
        "anchor_force_base": force_base,
        "anchor_position_base": p_base,
        "anchor_normal_base": normal_base,
        "active_count": int(np.sum(coeff[:, 0] > max(float(np.max(coeff[:, 0])) * 0.05, 1e-8))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate global-anchor force commands on exact GT states.")
    parser.add_argument("--anchors", default="global_anchor_force_field/data/box_global_anchors_200.npz")
    parser.add_argument("--urdf", help="FoLD object URDF supplying bottom/top inertial masses.")
    parser.add_argument("--target-q", default=r"D:\a_arctic_data_force\s01_box_use_01\s01_box_use_01_target_q_real.npy")
    parser.add_argument("--out", default="force_command_estimator/data/box_gt_global_force_commands.npz")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=20, help="Number of GT transitions; <=0 uses the rest of the sequence.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0.1)
    parser.add_argument("--active-top-k", type=int, default=60, help="0 uses all anchors.")
    parser.add_argument("--topk-method", choices=["normal_prefit", "score"], default="normal_prefit")
    parser.add_argument("--rot-w", type=float, default=200.0)
    parser.add_argument("--hinge-w", type=float, default=200.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--smooth-w", type=float, default=1e-2)
    parser.add_argument("--tangent-reg-scale", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    target_q = np.load(args.target_q, allow_pickle=False)
    start = int(args.start_frame)
    available = len(target_q) - 1 - start
    count = available if int(args.num_frames) <= 0 else min(int(args.num_frames), available)
    if start < 0 or count <= 0:
        raise ValueError(f"invalid range: start={start}, count={count}, target length={len(target_q)}")

    anchor_data = dict(np.load(args.anchors, allow_pickle=True))
    anchor_model = load_cached_anchors(anchor_data)
    if args.urdf:
        base_mass, lid_mass = load_urdf_link_masses(args.urdf)
    else:
        base_mass, lid_mass = 0.964, 0.132
    mass_terms = simple_dynamics_terms(base_mass=base_mass, lid_mass=lid_mass)
    rows = []
    solve_times = []
    for local_t, frame in enumerate(range(start, start + count)):
        tic = time.perf_counter()
        row = solve_gt_frame(target_q, frame, float(args.dt), anchor_model, mass_terms, args)
        solve_times.append(time.perf_counter() - tic)
        rows.append(row)
        if args.print_every > 0 and (local_t == 0 or (local_t + 1) % int(args.print_every) == 0):
            print(
                f"frame={frame:04d} rel={row['rel_residual']:.4f} "
                f"active={row['active_count']} solve={solve_times[-1]:.4f}s"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame_index = np.arange(start, start + count, dtype=np.int32)
    np.savez_compressed(
        out,
        frame_index=frame_index,
        force_cmd_world=np.asarray([r["achieved_world"][:3] for r in rows], dtype=np.float32),
        force_cmd_base=np.asarray([r["achieved_base"][:3] for r in rows], dtype=np.float32),
        wrench_cmd_world=np.asarray([r["achieved_world"][:6] for r in rows], dtype=np.float32),
        wrench_cmd_base=np.asarray([r["achieved_base"][:6] for r in rows], dtype=np.float32),
        generalized_cmd_world=np.asarray([r["achieved_world"] for r in rows], dtype=np.float32),
        generalized_cmd_base=np.asarray([r["achieved_base"] for r in rows], dtype=np.float32),
        hinge_tau_cmd=np.asarray([r["achieved_world"][6] for r in rows], dtype=np.float32),
        tau_target_world=np.asarray([r["tau"] for r in rows], dtype=np.float32),
        residual_world=np.asarray([r["residual"] for r in rows], dtype=np.float32),
        rel_residual=np.asarray([r["rel_residual"] for r in rows], dtype=np.float32),
        contact_coeffs=np.asarray([r["coeff"] for r in rows], dtype=np.float32),
        anchor_force_world=np.asarray([r["anchor_force_world"] for r in rows], dtype=np.float32),
        anchor_force_base=np.asarray([r["anchor_force_base"] for r in rows], dtype=np.float32),
        anchor_position_base=np.asarray([r["anchor_position_base"] for r in rows], dtype=np.float32),
        anchor_normal_base=np.asarray([r["anchor_normal_base"] for r in rows], dtype=np.float32),
        active_counts=np.asarray([r["active_count"] for r in rows], dtype=np.int32),
        valid_mask=np.ones(count, dtype=bool),
        solve_time=np.asarray(solve_times, dtype=np.float32),
        part_ids=anchor_model.part_ids.astype(np.int32),
        points_obj=anchor_model.points.astype(np.float32),
        force_dirs_obj=anchor_model.force_dirs.astype(np.float32),
        dt=np.asarray(args.dt, dtype=np.float32),
        mu=np.asarray(args.mu, dtype=np.float32),
        active_top_k=np.asarray(args.active_top_k, dtype=np.int32),
        topk_method=np.asarray(args.topk_method),
        base_mass=np.asarray(base_mass, dtype=np.float32),
        lid_mass=np.asarray(lid_mass, dtype=np.float32),
        urdf=np.asarray(args.urdf or ""),
        anchors=np.asarray(args.anchors),
        target_q=np.asarray(args.target_q),
        command_alignment=np.asarray("command[i] maps GT[frame_index[i]] to GT[frame_index[i]+1]"),
    )
    print("\n--- GT Command Summary ---")
    print(f"Saved             : {out}")
    print(f"Masses            : base={base_mass:.6f} kg lid={lid_mass:.6f} kg")
    print(f"Frames            : {count} ({start}..{start + count - 1})")
    print(f"Mean solve time   : {np.mean(solve_times):.4f} s/frame")
    print(f"Mean rel residual : {np.mean([r['rel_residual'] for r in rows]):.4f}")
    print(f"Mean active       : {np.mean([r['active_count'] for r in rows]):.1f}")


if __name__ == "__main__":
    main()

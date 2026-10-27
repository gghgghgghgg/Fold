"""Closed-loop Genesis rollout using full-surface global normal anchors.

This tests the online/realtime-force-field idea: at each frame, compute a target
7D generalized force from the current simulator state and the next GT state, fit
that generalized force with global normal-only anchors, then step Genesis.

No free bias is used by default. The only forces come from nonnegative inward
surface-normal anchor coefficients.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osqp
import scipy.sparse as sp

from one_frame_global_anchor_fit import (
    DynState,
    aa_to_rotmat,
    build_normal_wrench_matrix,
    hinge_rot,
    pose_errors,
    quat_wxyz_to_rotvec,
    rotvec_to_quat_wxyz,
    simple_dynamics_terms,
    solve_normal_anchor_qp,
    target_qdot_between,
)
from generate_global_anchors import _parse_obj


@dataclass
class CachedAnchors:
    points: np.ndarray
    force_dirs: np.ndarray
    part_ids: np.ndarray
    base_center: np.ndarray
    edges: np.ndarray
    lambda_upper: np.ndarray


def _to_numpy_f64(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64, copy=False)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float64)


def load_cached_anchors(data: dict[str, np.ndarray]) -> CachedAnchors:
    mesh_scale = float(np.asarray(data.get("mesh_scale", 0.001)))
    mesh_paths = [str(x) for x in data["mesh_paths"]]
    centers = []
    for mesh_path in mesh_paths:
        mesh = _parse_obj(Path(mesh_path), mesh_scale)
        centers.append(mesh.vertices.mean(axis=0))
    return CachedAnchors(
        points=np.asarray(data["points_obj"], dtype=np.float64),
        force_dirs=np.asarray(data["force_dirs_obj"], dtype=np.float64),
        part_ids=np.asarray(data["part_ids"], dtype=np.int64),
        base_center=np.asarray(centers[0], dtype=np.float64),
        edges=np.asarray(data["neighbor_edges"], dtype=np.int64),
        lambda_upper=np.asarray(data["lambda_upper"], dtype=np.float64),
    )


def transform_cached_anchors(model: CachedAnchors, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_base = aa_to_rotmat(q[3:6])
    R_lid = hinge_rot(float(q[6]))
    p_rel = np.zeros_like(model.points)
    f_world = np.zeros_like(model.force_dirs)
    for i, (p, f, part) in enumerate(zip(model.points, model.force_dirs, model.part_ids)):
        if int(part) == 0:
            p_base = p - model.base_center
            f_base = f
        else:
            p_base = R_lid @ p - model.base_center
            f_base = R_lid @ f
        p_rel[i] = R_base @ p_base
        f_world[i] = R_base @ f_base
    n = np.linalg.norm(f_world, axis=1)
    ok = n > 1e-12
    f_world[ok] /= n[ok, None]
    return p_rel, f_world, model.part_ids



def _tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(normals, dtype=np.float64)
    t1 = np.zeros_like(normals)
    refs = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (len(normals), 1))
    near_z = np.abs(np.sum(normals * refs, axis=1)) > 0.9
    refs[near_z] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = np.cross(refs, normals)
    t1_norm = np.linalg.norm(t1, axis=1)
    ok = t1_norm > 1e-12
    t1[ok] /= t1_norm[ok, None]
    t2 = np.cross(normals, t1)
    t2_norm = np.linalg.norm(t2, axis=1)
    ok2 = t2_norm > 1e-12
    t2[ok2] /= t2_norm[ok2, None]
    return t1, t2


def _generalized_wrench_column(
    p: np.ndarray,
    f: np.ndarray,
    part: int,
    q: np.ndarray,
    hinge_axis_local=(0.0, 0.0, -1.0),
    hinge_pos_local=(0.0, 0.0, 0.0),
) -> np.ndarray:
    col = np.zeros(7, dtype=np.float64)
    col[:3] = f
    col[3:6] = np.cross(p, f)
    if int(part) == 1:
        R_base = aa_to_rotmat(q[3:6])
        hinge_axis_world = R_base @ np.asarray(hinge_axis_local, dtype=np.float64)
        hinge_axis_world /= max(float(np.linalg.norm(hinge_axis_world)), 1e-12)
        hinge_world_rel = R_base @ np.asarray(hinge_pos_local, dtype=np.float64)
        col[6] = float(np.dot(np.cross(p - hinge_world_rel, f), hinge_axis_world))
    return col


def build_friction_pyramid_wrench_matrix(
    p_rel: np.ndarray,
    normal_dirs: np.ndarray,
    part_ids: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return A for variables [fn,t1+,t1-,t2+,t2-] per anchor."""
    t1, t2 = _tangent_basis(normal_dirs)
    n = len(p_rel)
    A = np.zeros((7, 5 * n), dtype=np.float64)
    for i in range(n):
        dirs = (normal_dirs[i], t1[i], -t1[i], t2[i], -t2[i])
        for j, f in enumerate(dirs):
            A[:, 5 * i + j] = _generalized_wrench_column(p_rel[i], f, int(part_ids[i]), q)
    return A, t1, t2


def select_topk_anchors_by_score(A_normal: np.ndarray, tau: np.ndarray, top_k: int, rot_w: float, hinge_w: float) -> np.ndarray:
    n = A_normal.shape[1]
    if top_k <= 0 or top_k >= n:
        return np.arange(n, dtype=np.int64)
    W = np.diag([1.0, 1.0, 1.0, float(rot_w), float(rot_w), float(rot_w), float(hinge_w)])
    score = (W @ A_normal).T @ (W @ tau)
    # Positive correlation means a nonnegative normal force helps the target tau.
    score = np.maximum(score, 0.0)
    if np.count_nonzero(score) < top_k:
        score = np.abs((W @ A_normal).T @ (W @ tau))
    idx = np.argpartition(score, -int(top_k))[-int(top_k):]
    return np.asarray(idx[np.argsort(score[idx])[::-1]], dtype=np.int64)


def subset_edges(edges: np.ndarray, selected: np.ndarray) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int64)
    remap = {int(old): i for i, old in enumerate(selected)}
    out = []
    for a, b in np.asarray(edges, dtype=np.int64):
        ia = remap.get(int(a))
        ib = remap.get(int(b))
        if ia is not None and ib is not None:
            out.append((ia, ib))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def solve_friction_pyramid_qp(
    A: np.ndarray,
    tau: np.ndarray,
    edges: np.ndarray,
    lambda_upper: np.ndarray,
    mu: float,
    rot_w: float,
    hinge_w: float,
    reg: float,
    smooth_w: float,
    tangent_reg_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(lambda_upper)
    dim = 5 * n
    W = np.diag([1.0, 1.0, 1.0, rot_w, rot_w, rot_w, hinge_w])
    Aw = W @ A
    tw = W @ tau
    P = 2.0 * (Aw.T @ Aw)
    q = -2.0 * Aw.T @ tw
    diag = np.full(dim, float(reg), dtype=np.float64)
    diag[1::5] *= float(tangent_reg_scale)
    diag[2::5] *= float(tangent_reg_scale)
    diag[3::5] *= float(tangent_reg_scale)
    diag[4::5] *= float(tangent_reg_scale)
    P += np.diag(diag)

    if smooth_w > 0.0 and len(edges) > 0:
        for a, b in edges.astype(int):
            ia, ib = 5 * int(a), 5 * int(b)
            P[ia, ia] += 2.0 * smooth_w
            P[ib, ib] += 2.0 * smooth_w
            P[ia, ib] -= 2.0 * smooth_w
            P[ib, ia] -= 2.0 * smooth_w

    rows = []
    lower = []
    upper = []
    # Variable bounds.
    rows.append(sp.eye(dim, format="csc"))
    lb = np.zeros(dim, dtype=np.float64)
    ub = np.full(dim, np.inf, dtype=np.float64)
    ub[0::5] = lambda_upper.astype(np.float64)
    # Individual tangential caps help numerics while the coupled row enforces the pyramid.
    tangent_cap = np.repeat(lambda_upper.astype(np.float64) * max(float(mu), 0.0), 4)
    ub[np.ravel(np.column_stack([np.arange(1, dim, 5), np.arange(2, dim, 5), np.arange(3, dim, 5), np.arange(4, dim, 5)]))] = tangent_cap
    lower.append(lb)
    upper.append(ub)

    # Friction pyramid: t1+ + t1- + t2+ + t2- <= mu * fn.
    row_i = []
    col_i = []
    data_i = []
    for i in range(n):
        row = i
        base = 5 * i
        row_i.extend([row] * 5)
        col_i.extend([base, base + 1, base + 2, base + 3, base + 4])
        data_i.extend([-float(mu), 1.0, 1.0, 1.0, 1.0])
    rows.append(sp.coo_matrix((data_i, (row_i, col_i)), shape=(n, dim)).tocsc())
    lower.append(np.full(n, -np.inf, dtype=np.float64))
    upper.append(np.zeros(n, dtype=np.float64))

    Acon = sp.vstack(rows, format="csc")
    l = np.concatenate(lower)
    u = np.concatenate(upper)
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(P), q, Acon, l, u, verbose=False, max_iter=10000, eps_abs=1e-5, eps_rel=1e-5)
    res = prob.solve()
    if res.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP friction failed: {res.info.status}")
    coeff = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, np.inf)
    return coeff, A @ coeff


def get_sim_state(box) -> DynState:
    pos = _to_numpy_f64(box.get_pos()).reshape(3)
    quat = _to_numpy_f64(box.get_quat()).reshape(4)
    rotvec = quat_wxyz_to_rotvec(quat)
    hinge = float(_to_numpy_f64(box.get_dofs_position([6])).reshape(-1)[0])
    qdot = _to_numpy_f64(box.get_dofs_velocity(list(range(7)))).reshape(7)
    return DynState(q=np.concatenate([pos, rotvec, [hinge]]), qdot=qdot, qddot=np.zeros(7))


def generalized_to_dof_force(total: np.ndarray, q: np.ndarray, max_abs: float) -> np.ndarray:
    total_f = total[:3]
    total_tau_local = aa_to_rotmat(q[3:6]).T @ total[3:6]
    dof_force = np.concatenate([total_f, total_tau_local, [total[6]]]).astype(np.float64)
    if max_abs > 0.0:
        dof_force = np.clip(dof_force, -float(max_abs), float(max_abs))
    return np.nan_to_num(dof_force, nan=0.0).astype(np.float32)


def desired_qddot_from_genesis_state(curr: DynState, target_next: DynState, dt: float) -> np.ndarray:
    """Match closed_loop_force_rollout_v3.py PD convention.

    Genesis returns floating-base angular velocity in the local/body frame. The
    SO(3) rotation error used here is in the world frame, so the damping term
    must compare against world-frame angular velocity.
    """
    kp_trans, kd_trans = 100.0, 30.0
    kp_rot, kd_rot = 900.0, 60.0
    kp_hinge, kd_hinge = 10000.0, 200.0
    diff = np.zeros(7, dtype=np.float64)
    diff[:3] = target_next.q[:3] - curr.q[:3]
    from one_frame_global_anchor_fit import rotvec_diff
    diff[3:6] = rotvec_diff(curr.q[3:6], target_next.q[3:6])
    diff[6] = target_next.q[6] - curr.q[6]
    kp = np.array([kp_trans] * 3 + [kp_rot] * 3 + [kp_hinge], dtype=np.float64)
    kd = np.array([kd_trans] * 3 + [kd_rot] * 3 + [kd_hinge], dtype=np.float64)
    curr_qdot_world = curr.qdot.copy()
    curr_qdot_world[3:6] = aa_to_rotmat(curr.q[3:6]) @ curr.qdot[3:6]
    target_qdot = diff / max(float(dt), 1e-9)
    qddot = kp * diff + kd * (target_qdot - curr_qdot_world)
    max_acc = np.array([200.0, 200.0, 200.0, 2000.0, 2000.0, 2000.0, 2000.0], dtype=np.float64)
    return np.clip(qddot, -max_acc, max_acc)


def run_rollout(args: argparse.Namespace) -> None:
    import genesis as gs

    anchor_data = dict(np.load(args.anchors, allow_pickle=True))
    anchor_model = load_cached_anchors(anchor_data)
    target_q_all = np.load(args.target_q, allow_pickle=False)
    start = int(args.start_frame)
    if args.num_frames <= 0:
        end = len(target_q_all) - 1
    else:
        end = min(len(target_q_all) - 1, start + int(args.num_frames))
    if start < 0 or start >= end:
        raise ValueError(f"invalid rollout range start={start}, end={end}, target_len={len(target_q_all)}")
    target_q = target_q_all[start : end + 1].copy()

    gs.init(seed=42, backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=float(args.dt), substeps=int(args.substeps)))
    box = scene.add_entity(material=gs.materials.Rigid(), morph=gs.morphs.MJCF(file=args.xml))
    scene.build()
    scene.step()

    box.set_pos(np.asarray(target_q[0, :3], dtype=np.float64))
    box.set_quat(rotvec_to_quat_wxyz(target_q[0, 3:6]))
    box.set_dofs_position(np.asarray([target_q[0, 6]], dtype=np.float64), [6])
    curr = get_sim_state(box)

    M, h, tau_known = simple_dynamics_terms()
    log_states = []
    lambda_seq = []
    contact_coeff_seq = []
    tau_seq = []
    achieved_seq = []
    residual_seq = []
    solve_times = []
    pos_errs = []
    rot_errs = []
    hinge_errs = []
    rel_resids = []
    active_counts = []

    t0_all = time.perf_counter()
    for t in range(end - start):
        tgt_next = DynState(
            q=target_q[t + 1].copy(),
            qdot=target_qdot_between(target_q, t, t + 1, float(args.dt)),
            qddot=np.zeros(7),
        )
        tau = M @ desired_qddot_from_genesis_state(curr, tgt_next, float(args.dt)) + h - tau_known
        qsolve0 = time.perf_counter()
        p_rel, f_dirs, part_ids = transform_cached_anchors(anchor_model, curr.q)
        if args.contact_model == "normal":
            A = build_normal_wrench_matrix(p_rel, f_dirs, part_ids, curr.q)
            lam, _bias, achieved = solve_normal_anchor_qp(
                A,
                tau,
                anchor_model.edges,
                anchor_model.lambda_upper,
                rot_w=float(args.rot_w),
                hinge_w=float(args.hinge_w),
                reg=float(args.reg),
                smooth_w=float(args.smooth_w),
                bias_reg=float(args.bias_reg),
                bias_bound=0.0,
            )
            contact_coeff = lam[:, None]
        elif args.contact_model == "friction_pyramid":
            A_normal = build_normal_wrench_matrix(p_rel, f_dirs, part_ids, curr.q)
            selected = select_topk_anchors_by_score(
                A_normal,
                tau,
                int(args.active_top_k),
                rot_w=float(args.rot_w),
                hinge_w=float(args.hinge_w),
            )
            A, _t1, _t2 = build_friction_pyramid_wrench_matrix(
                p_rel[selected], f_dirs[selected], part_ids[selected], curr.q
            )
            contact_coeff_flat, achieved = solve_friction_pyramid_qp(
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
            contact_coeff = np.zeros((len(anchor_model.points), 5), dtype=np.float64)
            contact_coeff[selected] = contact_coeff_flat.reshape(-1, 5)
            lam = contact_coeff[:, 0]
        else:
            raise ValueError(f"unknown contact model: {args.contact_model}")
        solve_times.append(time.perf_counter() - qsolve0)
        total = achieved  # bias is forced to zero.
        residual = tau - total
        rel_resids.append(float(np.linalg.norm(residual) / max(np.linalg.norm(tau), 1e-9)))
        active_counts.append(int(np.sum(lam > max(float(np.max(lam)) * 0.05, 1e-8))))

        dof_force = generalized_to_dof_force(total, curr.q, float(args.max_force_abs))
        box.control_dofs_force(dof_force, np.arange(7))
        scene.step()
        curr = get_sim_state(box)
        log_states.append(curr.q.copy())

        pe, re, he = pose_errors(curr.q, tgt_next.q)
        pos_errs.append(pe)
        rot_errs.append(re)
        hinge_errs.append(he)
        lambda_seq.append(lam.astype(np.float32))
        contact_coeff_seq.append(contact_coeff.astype(np.float32))
        tau_seq.append(tau.astype(np.float32))
        achieved_seq.append(achieved.astype(np.float32))
        residual_seq.append(residual.astype(np.float32))

        if args.print_every > 0 and ((t + 1) % int(args.print_every) == 0 or t == 0):
            print(
                f"frame {start+t+1:04d}/{end}: "
                f"pos={pe:.2f}mm rot={re:.2f}deg hinge={he:.3f}deg "
                f"rel={rel_resids[-1]:.3f} active={active_counts[-1]} "
                f"solve={solve_times[-1]:.4f}s"
            )

        if args.stop_pos_mm > 0 and pe > float(args.stop_pos_mm):
            print(f"Early stop: pos error {pe:.2f} mm > {args.stop_pos_mm:.2f} mm")
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pred_q=np.asarray(log_states, dtype=np.float64),
        lambda_values=np.asarray(lambda_seq, dtype=np.float32),
        contact_coeffs=np.asarray(contact_coeff_seq, dtype=np.float32),
        tau=np.asarray(tau_seq, dtype=np.float32),
        achieved=np.asarray(achieved_seq, dtype=np.float32),
        residual=np.asarray(residual_seq, dtype=np.float32),
        pos_err_mm=np.asarray(pos_errs, dtype=np.float32),
        rot_err_deg=np.asarray(rot_errs, dtype=np.float32),
        hinge_err_deg=np.asarray(hinge_errs, dtype=np.float32),
        rel_resid=np.asarray(rel_resids, dtype=np.float32),
        solve_time=np.asarray(solve_times, dtype=np.float32),
        active_counts=np.asarray(active_counts, dtype=np.int32),
        start_frame=np.asarray(start, dtype=np.int32),
        source_end_frame=np.asarray(start + len(log_states), dtype=np.int32),
        anchors=str(args.anchors),
        target_q=str(args.target_q),
        xml=str(args.xml),
        contact_model=str(args.contact_model),
        mu=np.asarray(float(args.mu), dtype=np.float32),
        active_top_k=np.asarray(int(args.active_top_k), dtype=np.int32),
    )

    pred_npy = out.with_suffix(".npy")
    np.save(pred_npy, np.asarray(log_states, dtype=np.float64))

    elapsed = time.perf_counter() - t0_all
    print("\n--- Global Anchor Rollout Summary ---")
    print(f"Frames            : {len(log_states)} / requested {end - start}")
    print(f"Anchors           : {len(anchor_model.points)}")
    print("Bias              : disabled")
    print(f"Contact Model     : {args.contact_model}" + (f" (mu={float(args.mu):.3f}, top_k={int(args.active_top_k) if int(args.active_top_k) > 0 else 'all'})" if args.contact_model == "friction_pyramid" else ""))
    if solve_times:
        print(f"Mean Solve Time   : {np.mean(solve_times):.4f} s/frame (Max: {np.max(solve_times):.4f})")
    if pos_errs:
        print(f"Mean Pos Error    : {np.mean(pos_errs):.2f} mm (Max: {np.max(pos_errs):.2f})")
        print(f"Mean Rot Error    : {np.mean(rot_errs):.2f} deg (Max: {np.max(rot_errs):.2f})")
        print(f"Mean Hinge Error  : {np.mean(hinge_errs):.3f} deg (Max: {np.max(hinge_errs):.3f})")
        print(f"Mean Rel Resid    : {np.mean(rel_resids):.4f}")
        print(f"Mean Active       : {np.mean(active_counts):.1f}")
    print(f"Elapsed           : {elapsed:.1f} s")
    print(f"Saved NPZ         : {out}")
    print(f"Saved pred NPY    : {pred_npy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop rollout with global anchors, optional friction pyramid, and no bias.")
    parser.add_argument("--anchors", default="global_anchor_force_field/data/box_global_anchors_200.npz")
    parser.add_argument("--target-q", default=r"D:\a_arctic_data_force\s01_box_use_01\s01_box_use_01_target_q_real.npy")
    parser.add_argument("--xml", default=r"D:\a_arctic_data_force\s01_box_use_01\s01_box_use_01_genesis_aligned.xml")
    parser.add_argument("--out", default="global_anchor_force_field/results/rollout/global_anchor_200_box_full.npz")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=80, help="<=0 means full sequence.")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--rot-w", type=float, default=200.0)
    parser.add_argument("--hinge-w", type=float, default=200.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--smooth-w", type=float, default=1e-2)
    parser.add_argument("--bias-reg", type=float, default=10.0)
    parser.add_argument("--contact-model", choices=["normal", "friction_pyramid"], default="normal")
    parser.add_argument("--mu", type=float, default=0.1, help="Friction pyramid coefficient for tangential force <= mu * normal force.")
    parser.add_argument("--tangent-reg-scale", type=float, default=1.0, help="Multiplier on tangential variable L2 regularization.")
    parser.add_argument("--active-top-k", type=int, default=0, help="For friction_pyramid, select only top-k anchors by normal-tau score. 0 uses all anchors.")
    parser.add_argument("--max-force-abs", type=float, default=200.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--stop-pos-mm", type=float, default=0.0)
    args = parser.parse_args()
    run_rollout(args)


if __name__ == "__main__":
    main()

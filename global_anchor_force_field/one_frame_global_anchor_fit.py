"""One-frame global-anchor normal-force fit for a hinged box.

This is a small standalone test that mirrors the one-frame target construction
from hinge_inverse_dynamics/closed_loop_force_rollout_v3.py, but uses full-surface
normal-only global anchors instead of demo contact anchors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import osqp
import scipy.sparse as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from generate_global_anchors import _face_geometry, _parse_obj


@dataclass
class DynState:
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray


def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = np.asarray(aa, dtype=np.float64) / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    tr = float(np.trace(R))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return (theta / (2.0 * np.sin(theta))) * w


def rotvec_diff(curr_r: np.ndarray, tgt_r: np.ndarray) -> np.ndarray:
    return so3_log(aa_to_rotmat(tgt_r) @ aa_to_rotmat(curr_r).T)


def desired_qddot(curr: DynState, target_next: DynState, dt: float) -> np.ndarray:
    kp_trans, kd_trans = 100.0, 30.0
    kp_rot, kd_rot = 900.0, 60.0
    kp_hinge, kd_hinge = 10000.0, 200.0
    diff = np.zeros(7, dtype=np.float64)
    diff[:3] = target_next.q[:3] - curr.q[:3]
    diff[3:6] = rotvec_diff(curr.q[3:6], target_next.q[3:6])
    diff[6] = target_next.q[6] - curr.q[6]
    kp = np.array([kp_trans] * 3 + [kp_rot] * 3 + [kp_hinge], dtype=np.float64)
    kd = np.array([kd_trans] * 3 + [kd_rot] * 3 + [kd_hinge], dtype=np.float64)
    target_qdot = diff / max(dt, 1e-9)
    qddot = kp * diff + kd * (target_qdot - curr.qdot)
    max_acc = np.array([200.0, 200.0, 200.0, 2000.0, 2000.0, 2000.0, 2000.0], dtype=np.float64)
    return np.clip(qddot, -max_acc, max_acc)


def target_qdot_between(target_q: np.ndarray, a: int, b: int, dt: float) -> np.ndarray:
    a = int(np.clip(a, 0, len(target_q) - 1))
    b = int(np.clip(b, 0, len(target_q) - 1))
    qdot = np.zeros(7, dtype=np.float64)
    qdot[:3] = (target_q[b, :3] - target_q[a, :3]) / max(dt, 1e-9)
    qdot[3:6] = rotvec_diff(target_q[a, 3:6], target_q[b, 3:6]) / max(dt, 1e-9)
    qdot[6] = (target_q[b, 6] - target_q[a, 6]) / max(dt, 1e-9)
    return qdot


def simple_dynamics_terms(base_mass: float = 0.964, lid_mass: float = 0.132) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_mass = float(base_mass) + float(lid_mass)
    M = np.diag([total_mass, total_mass, total_mass, 0.0051, 0.0051, 0.0051, 0.02]).astype(np.float64)
    h = np.zeros(7, dtype=np.float64)
    h[2] = total_mass * 9.81
    tau_known = np.zeros(7, dtype=np.float64)
    return M, h, tau_known


def load_part_centers(mesh_paths: list[str], scale: float) -> np.ndarray:
    centers = []
    for path in mesh_paths:
        mesh = _parse_obj(Path(str(path)), scale)
        centers.append(mesh.vertices.mean(axis=0))
    return np.asarray(centers, dtype=np.float64)


def hinge_rot(angle: float, axis=(0.0, 0.0, -1.0)) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return aa_to_rotmat(axis * float(angle))


def transform_anchors_to_base_frame(data: dict[str, np.ndarray], q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(data["points_obj"], dtype=np.float64)
    force_dirs = np.asarray(data["force_dirs_obj"], dtype=np.float64)
    part_ids = np.asarray(data["part_ids"], dtype=np.int64)
    mesh_scale = float(np.asarray(data.get("mesh_scale", 0.001)))
    centers = load_part_centers([str(x) for x in data["mesh_paths"]], mesh_scale)
    base_center = centers[0]
    R_base = aa_to_rotmat(q[3:6])
    R_lid = hinge_rot(float(q[6]))

    p_rel = np.zeros_like(points)
    f_world = np.zeros_like(force_dirs)
    for i, (p, f, part) in enumerate(zip(points, force_dirs, part_ids)):
        if int(part) == 0:
            p_base = p - base_center
            f_base = f
        else:
            # Lid body is articulated relative to the base origin. Express both part centers
            # relative to the base center so base/lid do not share smoothing but can both
            # contribute wrench around the base COM.
            p_base = R_lid @ p - base_center
            f_base = R_lid @ f
        p_rel[i] = R_base @ p_base
        f_world[i] = R_base @ f_base
    f_norm = np.linalg.norm(f_world, axis=1)
    ok = f_norm > 1e-12
    f_world[ok] /= f_norm[ok, None]
    return p_rel, f_world, part_ids


def build_normal_wrench_matrix(
    p_rel: np.ndarray,
    f_dirs: np.ndarray,
    part_ids: np.ndarray,
    q: np.ndarray,
    hinge_axis_local=(0.0, 0.0, -1.0),
    hinge_pos_local=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Map normal anchor magnitudes to 7D generalized force.

    Rows 0:3 are base force, rows 3:6 are base torque around base COM,
    row 6 is hinge torque. Only lid anchors contribute to hinge torque.
    """
    A = np.zeros((7, len(p_rel)), dtype=np.float64)
    R_base = aa_to_rotmat(q[3:6])
    hinge_axis_world = R_base @ np.asarray(hinge_axis_local, dtype=np.float64)
    hinge_axis_world = hinge_axis_world / max(float(np.linalg.norm(hinge_axis_world)), 1e-12)
    hinge_world_rel = R_base @ np.asarray(hinge_pos_local, dtype=np.float64)
    for i, (p, f, part) in enumerate(zip(p_rel, f_dirs, part_ids)):
        A[:6, i] = np.concatenate([f, np.cross(p, f)])
        if int(part) == 1:
            A[6, i] = float(np.dot(np.cross(p - hinge_world_rel, f), hinge_axis_world))
    return A


def solve_normal_anchor_qp(
    A: np.ndarray,
    tau: np.ndarray,
    edges: np.ndarray,
    lambda_upper: np.ndarray,
    rot_w: float,
    hinge_w: float,
    reg: float,
    smooth_w: float,
    bias_reg: float,
    bias_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = A.shape[1]
    dof = A.shape[0]
    A_full = np.concatenate([A, np.eye(dof)], axis=1)
    if dof == 7:
        W = np.diag([1.0, 1.0, 1.0, rot_w, rot_w, rot_w, hinge_w])
    else:
        W = np.diag([1.0, 1.0, 1.0, rot_w, rot_w, rot_w])
    Aw = W @ A_full
    tw = W @ tau
    P = 2.0 * (Aw.T @ Aw)
    q = -2.0 * Aw.T @ tw
    diag = np.concatenate([np.full(n, reg, dtype=np.float64), np.full(dof, bias_reg, dtype=np.float64)])
    P += np.diag(diag)

    if smooth_w > 0.0 and len(edges) > 0:
        for a, b in edges.astype(int):
            P[a, a] += 2.0 * smooth_w
            P[b, b] += 2.0 * smooth_w
            P[a, b] -= 2.0 * smooth_w
            P[b, a] -= 2.0 * smooth_w

    lb = np.concatenate([np.zeros(n, dtype=np.float64), np.full(dof, -float(bias_bound), dtype=np.float64)])
    ub = np.concatenate([lambda_upper.astype(np.float64), np.full(dof, float(bias_bound), dtype=np.float64)])
    prob = osqp.OSQP()
    prob.setup(sp.csc_matrix(P), q, sp.eye(n + dof, format="csc"), lb, ub, verbose=False, max_iter=5000)
    res = prob.solve()
    if res.info.status_val not in (1, 2):
        raise RuntimeError(f"OSQP failed: {res.info.status}")
    x = np.asarray(res.x, dtype=np.float64)
    lam = np.clip(x[:n], 0.0, lambda_upper)
    bias = np.clip(x[n:], -float(bias_bound), float(bias_bound))
    return lam, bias, A @ lam




def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = np.asarray(rotvec, dtype=np.float64) / theta
    half = 0.5 * theta
    return np.array([np.cos(half), *(axis * np.sin(half))], dtype=np.float64)


def quat_wxyz_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    w = float(np.clip(w, -1.0, 1.0))
    theta = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-9:
        return np.zeros(3, dtype=np.float64)
    return np.array([x, y, z], dtype=np.float64) / s * theta


def genesis_one_step(xml: str, q_curr: np.ndarray, dof_force: np.ndarray, dt: float, substeps: int) -> np.ndarray:
    import genesis as gs

    gs.init(seed=42, backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=float(dt), substeps=int(substeps)))
    box = scene.add_entity(material=gs.materials.Rigid(), morph=gs.morphs.MJCF(file=xml))
    scene.build()
    scene.step()
    box.set_pos(np.asarray(q_curr[:3], dtype=np.float64))
    box.set_quat(rotvec_to_quat_wxyz(q_curr[3:6]))
    box.set_dofs_position(np.asarray([q_curr[6]], dtype=np.float64), [6])
    box.control_dofs_force(np.asarray(dof_force, dtype=np.float32), np.arange(7))
    scene.step()
    pos = np.asarray(box.get_pos(), dtype=np.float64).reshape(3)
    quat = np.asarray(box.get_quat(), dtype=np.float64).reshape(4)
    hinge = float(np.asarray(box.get_dofs_position([6]), dtype=np.float64).reshape(-1)[0])
    return np.concatenate([pos, quat_wxyz_to_rotvec(quat), [hinge]])


def pose_errors(q_pred: np.ndarray, q_tgt: np.ndarray) -> tuple[float, float, float]:
    pos_mm = float(np.linalg.norm(q_pred[:3] - q_tgt[:3]) * 1000.0)
    rot_deg = float(np.linalg.norm(rotvec_diff(q_pred[3:6], q_tgt[3:6])) * 180.0 / np.pi)
    hinge_deg = float(abs(q_pred[6] - q_tgt[6]) * 180.0 / np.pi)
    return pos_mm, rot_deg, hinge_deg

def visualize_solution(data: dict[str, np.ndarray], lam: np.ndarray, out: Path) -> None:
    points = np.asarray(data["points_obj"], dtype=np.float64)
    patch_ids = np.asarray(data["patch_ids"], dtype=np.int64)
    if np.max(lam) > 1e-12:
        color_val = lam / np.max(lam)
    else:
        color_val = lam
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color_val, cmap="magma", s=28, depthshade=False)
    active = np.flatnonzero(lam > max(float(np.max(lam)) * 0.05, 1e-6))
    if len(active) > 0:
        ax.scatter(points[active, 0], points[active, 1], points[active, 2], facecolors="none", edgecolors="cyan", s=80)
    ax.set_title("One-frame global-anchor normal force fit\ncolor = normalized lambda, cyan = active anchors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - 0.55 * span, c + 0.55 * span)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="lambda / max(lambda)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", default="global_anchor_force_field/data/box_global_anchors_1000.npz")
    parser.add_argument("--target-q", default=r"D:\a_arctic_data_force\s01_box_use_01\s01_box_use_01_target_q_real.npy")
    parser.add_argument("--frame", type=int, default=60)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--rot-w", type=float, default=200.0)
    parser.add_argument("--hinge-w", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--smooth-w", type=float, default=1e-2)
    parser.add_argument("--bias-reg", type=float, default=10.0)
    parser.add_argument("--bias-bound", type=float, default=0.0, help="0 disables free 6D bias.")
    parser.add_argument("--out", default="global_anchor_force_field/results/one_frame/one_frame_box_fit_frame60.npz")
    parser.add_argument("--plot", default="global_anchor_force_field/results/visualizations/one_frame_box_fit_frame60.png")
    parser.add_argument("--xml", default=r"D:\a_arctic_data_force\s01_box_use_01\s01_box_use_01_genesis_aligned.xml")
    parser.add_argument("--genesis-step", action="store_true")
    parser.add_argument("--substeps", type=int, default=40)
    args = parser.parse_args()

    data = dict(np.load(args.anchors, allow_pickle=True))
    target_q = np.load(args.target_q, allow_pickle=False)
    frame = int(args.frame)
    if frame < 0 or frame >= len(target_q) - 1:
        raise ValueError(f"frame must be in [0, {len(target_q)-2}], got {frame}")

    curr = DynState(q=target_q[frame].copy(), qdot=target_qdot_between(target_q, max(frame - 1, 0), frame, args.dt), qddot=np.zeros(7))
    tgt_next = DynState(q=target_q[frame + 1].copy(), qdot=target_qdot_between(target_q, frame, frame + 1, args.dt), qddot=np.zeros(7))
    M, h, tau_known = simple_dynamics_terms()
    tau = M @ desired_qddot(curr, tgt_next, args.dt) + h - tau_known

    p_rel, f_dirs, part_ids = transform_anchors_to_base_frame(data, curr.q)
    A = build_normal_wrench_matrix(p_rel, f_dirs, part_ids, curr.q)
    lam, bias, achieved = solve_normal_anchor_qp(
        A,
        tau,
        np.asarray(data["neighbor_edges"], dtype=np.int64),
        np.asarray(data["lambda_upper"], dtype=np.float64),
        rot_w=float(args.rot_w),
        hinge_w=float(args.hinge_w),
        reg=float(args.reg),
        smooth_w=float(args.smooth_w),
        bias_reg=float(args.bias_reg),
        bias_bound=float(args.bias_bound),
    )
    total = achieved + bias
    residual = tau - total

    genesis_q = np.full(7, np.nan, dtype=np.float64)
    genesis_err = np.full(3, np.nan, dtype=np.float64)
    if args.genesis_step:
        total_f = total[:3]
        total_tau_world = total[3:6]
        total_tau_local = aa_to_rotmat(curr.q[3:6]).T @ total_tau_world
        dof_force = np.concatenate([total_f, total_tau_local, [total[6]]])
        genesis_q = genesis_one_step(args.xml, curr.q, dof_force, args.dt, args.substeps)
        genesis_err = np.asarray(pose_errors(genesis_q, tgt_next.q), dtype=np.float64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frame=np.asarray(frame, dtype=np.int32),
        tau=tau.astype(np.float64),
        achieved_wrench=achieved.astype(np.float64),
        bias_wrench=bias.astype(np.float64),
        residual=residual.astype(np.float64),
        lambda_values=lam.astype(np.float32),
        active_mask=(lam > max(float(np.max(lam)) * 0.05, 1e-6)),
        p_rels=p_rel.astype(np.float32),
        force_dirs_world=f_dirs.astype(np.float32),
        genesis_q=genesis_q.astype(np.float64),
        genesis_err=genesis_err.astype(np.float64),
        anchors=str(args.anchors),
        target_q=str(args.target_q),
    )
    visualize_solution(data, lam, Path(args.plot))

    tau_norm = max(float(np.linalg.norm(tau)), 1e-9)
    print(f"Saved: {out}")
    print(f"Saved plot: {args.plot}")
    print(f"frame={frame} anchors={len(lam)} active>{max(float(np.max(lam))*0.05, 1e-6):.3g}: {int(np.sum(lam > max(float(np.max(lam))*0.05, 1e-6)))}")
    print(f"tau7       : {np.array2string(tau, precision=4)}")
    print(f"achieved   : {np.array2string(achieved, precision=4)}")
    print(f"bias       : {np.array2string(bias, precision=4)}")
    print(f"residual   : {np.array2string(residual, precision=4)}")
    print(f"rel_resid  : {np.linalg.norm(residual) / tau_norm:.4f}")
    print(f"lambda mean/max/sum: {float(np.mean(lam)):.4f} / {float(np.max(lam)):.4f} / {float(np.sum(lam)):.4f}")
    if args.genesis_step:
        print(f"Genesis one-step error: pos={genesis_err[0]:.3f} mm rot={genesis_err[1]:.3f} deg hinge={genesis_err[2]:.3f} deg")


if __name__ == "__main__":
    main()

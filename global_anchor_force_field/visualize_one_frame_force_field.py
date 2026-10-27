"""Render one-frame global-anchor force-field strength on the object mesh.

Default output uses a real mesh renderer (trimesh + pyrender), not matplotlib
scatter/Poly3D sampling. The force magnitude is spread only inside the same
articulated part and smooth surface patch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from generate_global_anchors import _face_geometry, _normal_patches, _parse_obj
from visualize_global_anchors import _nearest_neighbor_sigma


@dataclass
class RenderPart:
    name: str
    part_id: int
    vertices: np.ndarray
    faces: np.ndarray
    centroids: np.ndarray
    patches: np.ndarray


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    nz = values[values > 1e-12]
    if len(nz) == 0:
        return np.zeros_like(values)
    vmax = float(np.percentile(nz, 95.0))
    vmax = max(vmax, float(np.max(nz)) * 0.25, 1e-12)
    return np.clip(values / vmax, 0.0, 1.0)


def _load_render_parts(data: dict[str, np.ndarray]) -> list[RenderPart]:
    mesh_paths = [str(x) for x in data["mesh_paths"]]
    part_names = [str(x) for x in data.get("part_names", np.arange(len(mesh_paths)))]
    scale = float(np.asarray(data.get("mesh_scale", 0.001)))
    sharp_angle_deg = float(np.asarray(data.get("sharp_angle_deg", 35.0)))
    parts: list[RenderPart] = []
    patch_offset = 0
    for part_id, mesh_path in enumerate(mesh_paths):
        mesh = _parse_obj(Path(mesh_path), scale)
        normals, centroids, _areas = _face_geometry(mesh.vertices, mesh.faces)
        patches = _normal_patches(mesh.faces, normals, sharp_angle_deg) + patch_offset
        patch_offset += int(np.max(patches)) + 1
        name = part_names[part_id] if part_id < len(part_names) else f"part_{part_id}"
        parts.append(
            RenderPart(
                name=name,
                part_id=part_id,
                vertices=mesh.vertices.astype(np.float64),
                faces=mesh.faces.astype(np.int64),
                centroids=centroids.astype(np.float64),
                patches=patches.astype(np.int64),
            )
        )
    return parts


def _face_force_weights(data: dict[str, np.ndarray], parts: list[RenderPart], lam: np.ndarray, sigma: float) -> list[np.ndarray]:
    points = np.asarray(data["points_obj"], dtype=np.float64)
    part_ids = np.asarray(data["part_ids"], dtype=np.int64)
    source_faces = np.asarray(data["source_face_ids"], dtype=np.int64)
    weights_by_part = [np.zeros(len(part.faces), dtype=np.float64) for part in parts]

    # Map each active anchor to the full-mesh smooth patch of its source face. This
    # avoids crossing sharp edges or base/lid part boundaries even when the anchor
    # patch ids were compacted after sampling.
    for part in parts:
        anchor_idx = np.flatnonzero((part_ids == part.part_id) & (lam > 1e-10))
        if len(anchor_idx) == 0:
            continue
        patch_to_anchors: dict[int, list[int]] = {}
        for anchor in anchor_idx:
            face_id = int(source_faces[anchor])
            if face_id < 0 or face_id >= len(part.patches):
                continue
            patch = int(part.patches[face_id])
            patch_to_anchors.setdefault(patch, []).append(int(anchor))

        for patch, anchors in patch_to_anchors.items():
            face_idx = np.flatnonzero(part.patches == patch)
            if len(face_idx) == 0:
                continue
            anchors_np = np.asarray(anchors, dtype=np.int64)
            d = np.linalg.norm(part.centroids[face_idx, None, :] - points[anchors_np][None, :, :], axis=2)
            k = np.exp(-(d * d) / (2.0 * sigma * sigma))
            weights_by_part[part.part_id][face_idx] = k @ lam[anchors_np]
    return weights_by_part


def print_patch_force_diagnostics(data: dict[str, np.ndarray], result: dict[str, np.ndarray], top_k: int = 12) -> None:
    lam = np.asarray(result["lambda_values"], dtype=np.float64).reshape(-1)
    points = np.asarray(data["points_obj"], dtype=np.float64)
    dirs = np.asarray(data["force_dirs_obj"], dtype=np.float64)
    patch_ids = np.asarray(data["patch_ids"], dtype=np.int64)
    part_ids = np.asarray(data["part_ids"], dtype=np.int64)
    part_names = [str(x) for x in data.get("part_names", [])]
    rows = []
    for patch in np.unique(patch_ids):
        idx = np.flatnonzero(patch_ids == int(patch))
        if len(idx) == 0:
            continue
        lsum = float(np.sum(lam[idx]))
        if lsum <= 1e-8:
            continue
        f = np.sum(dirs[idx] * lam[idx, None], axis=0)
        c = np.average(points[idx], axis=0, weights=lam[idx] + 1e-12)
        avg_dir = f / max(float(np.linalg.norm(f)), 1e-12)
        part = int(np.bincount(part_ids[idx]).argmax())
        part_name = part_names[part] if part < len(part_names) else str(part)
        rows.append((lsum, int(patch), part_name, c, avg_dir, f))
    rows.sort(reverse=True, key=lambda x: x[0])
    print("--- Patch force diagnostics (top lambda sums) ---")
    print("sum_lambda patch part center_xyz avg_force_dir net_force_xyz")
    for lsum, patch, part_name, c, avg_dir, f in rows[: int(top_k)]:
        print(
            f"{lsum:9.4f} {patch:5d} {part_name:>8s} "
            f"[{c[0]: .3f},{c[1]: .3f},{c[2]: .3f}] "
            f"[{avg_dir[0]: .2f},{avg_dir[1]: .2f},{avg_dir[2]: .2f}] "
            f"[{f[0]: .3f},{f[1]: .3f},{f[2]: .3f}]"
        )



def _face_colors(norm_weights: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    colors = np.zeros((len(norm_weights), 4), dtype=np.uint8)
    base = np.array([155, 158, 164, 255], dtype=np.uint8)
    colors[:] = base
    active = norm_weights > 0.004
    if np.any(active):
        mapped = (cmap(norm_weights[active]) * 255.0).astype(np.uint8)
        mapped[:, 3] = 255
        colors[active] = mapped
    return colors


def _camera_pose(target: np.ndarray, radius: float, elev_deg: float, azim_deg: float) -> np.ndarray:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    direction = np.array([
        np.cos(elev) * np.cos(azim),
        np.cos(elev) * np.sin(azim),
        np.sin(elev),
    ], dtype=np.float64)
    eye = target + radius * direction
    z_axis = eye - target
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = eye
    return pose


def _render_mesh_frame(
    parts: list[RenderPart],
    weights_by_part: list[np.ndarray],
    width: int,
    height: int,
    elev: float,
    azim: float,
    bg: tuple[float, float, float],
    cmap_name: str,
    use_lights: bool,
) -> np.ndarray:
    import pyrender

    all_vertices = np.concatenate([p.vertices for p in parts], axis=0)
    mins, maxs = all_vertices.min(axis=0), all_vertices.max(axis=0)
    target = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1e-4)
    radius = span * 2.2

    ambient = [0.95, 0.95, 0.95] if not use_lights else [0.28, 0.28, 0.28]
    scene = pyrender.Scene(bg_color=[*bg, 1.0], ambient_light=ambient)
    for part, weights in zip(parts, weights_by_part):
        mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
        mesh.visual.face_colors = _face_colors(weights, cmap_name=cmap_name)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    cam_pose = _camera_pose(target, radius=radius, elev_deg=elev, azim_deg=azim)
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.2), pose=cam_pose)
    if use_lights:
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.5), pose=cam_pose)
        fill_pose = _camera_pose(target, radius=radius, elev_deg=max(elev - 35.0, -60.0), azim_deg=azim + 80.0)
        scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5), pose=fill_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=int(width), viewport_height=int(height))
    try:
        flags = pyrender.RenderFlags.FLAT if not use_lights else pyrender.RenderFlags.NONE
        color, _depth = renderer.render(scene, flags=flags)
    finally:
        renderer.delete()
    return color


def _add_colorbar(img: np.ndarray, cmap_name: str = "jet", width: int = 46) -> np.ndarray:
    h = img.shape[0]
    cmap = plt.get_cmap(cmap_name)
    grad = np.linspace(1.0, 0.0, h, dtype=np.float64)
    bar = (cmap(grad)[:, :3] * 255.0).astype(np.uint8)
    bar = np.repeat(bar[:, None, :], width, axis=1)
    pad = np.full((h, 12, 3), 255, dtype=np.uint8)
    return np.concatenate([img, pad, bar], axis=1)


def render_force_field_mesh(
    data: dict[str, np.ndarray],
    result: dict[str, np.ndarray],
    out_png: Path,
    out_gif: Path | None,
    sigma: float,
    rotate_gif: bool,
    gif_frames: int,
    gif_fps: int,
    width: int,
    height: int,
    elev: float,
    azim: float,
    colorbar: bool,
    cmap_name: str,
    use_lights: bool,
) -> None:
    points = np.asarray(data["points_obj"], dtype=np.float64)
    lam = np.asarray(result["lambda_values"], dtype=np.float64).reshape(-1)
    if sigma <= 0.0:
        sigma = _nearest_neighbor_sigma(points, np.asarray(data["patch_ids"], dtype=np.int64))

    parts = _load_render_parts(data)
    raw_weights = _face_force_weights(data, parts, lam, sigma)
    all_weights = np.concatenate(raw_weights) if raw_weights else np.zeros(0)
    all_norm = _normalized(all_weights)
    weights_by_part = []
    offset = 0
    for w in raw_weights:
        weights_by_part.append(all_norm[offset: offset + len(w)])
        offset += len(w)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    frame = _render_mesh_frame(parts, weights_by_part, width, height, elev, azim, bg=(1.0, 1.0, 1.0), cmap_name=cmap_name, use_lights=use_lights)
    if colorbar:
        frame = _add_colorbar(frame, cmap_name=cmap_name)
    imageio.imwrite(out_png, frame)
    print(f"Saved mesh strength PNG: {out_png}")

    if rotate_gif:
        if out_gif is None:
            out_gif = out_png.with_name(out_png.stem + "_rotate.gif")
        images = []
        for i in range(max(4, int(gif_frames))):
            a = float(azim) + 360.0 * i / max(4, int(gif_frames))
            img = _render_mesh_frame(parts, weights_by_part, width, height, elev, a, bg=(1.0, 1.0, 1.0), cmap_name=cmap_name, use_lights=use_lights)
            if colorbar:
                img = _add_colorbar(img, cmap_name=cmap_name)
            images.append(img)
        imageio.mimsave(out_gif, images, fps=int(gif_fps), loop=0)
        print(f"Saved mesh strength GIF: {out_gif}")

    active = int(np.sum(lam > max(float(np.max(lam)) * 0.05, 1e-8))) if len(lam) else 0
    print(f"anchors={len(lam)} active={active} lambda mean/max/sum={float(np.mean(lam)):.4f}/{float(np.max(lam)):.4f}/{float(np.sum(lam)):.4f}")
    print(f"rbf sigma={sigma:.6f} m")
    print_patch_force_diagnostics(data, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one-frame global-anchor force-strength map on a real mesh.")
    parser.add_argument("--anchors", default="global_anchor_force_field/data/box_global_anchors_200.npz")
    parser.add_argument("--result", default="global_anchor_force_field/results/one_frame/one_frame_box_fit_frame60_200a_7d_hingew200_genesis.npz")
    parser.add_argument("--out-prefix", default="global_anchor_force_field/results/visualizations/one_frame_box_fit_frame60_200a_force_field_mesh")
    parser.add_argument("--sigma", type=float, default=0.0, help="RBF sigma in meters. <=0 uses median same-patch NN distance * 2.5.")
    parser.add_argument("--rotate-gif", action="store_true")
    parser.add_argument("--gif-frames", type=int, default=48)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--azim", type=float, default=-52.0)
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--with-light", action="store_true", help="Enable directional lights. Default is flat/ambient rendering.")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap; default jet maps low blue and high red.")
    args = parser.parse_args()

    data = dict(np.load(args.anchors, allow_pickle=True))
    result = dict(np.load(args.result, allow_pickle=True))
    prefix = Path(args.out_prefix)
    render_force_field_mesh(
        data=data,
        result=result,
        out_png=prefix.with_name(prefix.name + "_strength.png"),
        out_gif=prefix.with_name(prefix.name + "_strength_rotate.gif"),
        sigma=float(args.sigma),
        rotate_gif=bool(args.rotate_gif),
        gif_frames=int(args.gif_frames),
        gif_fps=int(args.gif_fps),
        width=int(args.width),
        height=int(args.height),
        elev=float(args.elev),
        azim=float(args.azim),
        colorbar=not bool(args.no_colorbar),
        cmap_name=str(args.cmap),
        use_lights=bool(args.with_light),
    )


if __name__ == "__main__":
    main()

"""Visualize full-surface anchors, smooth patches, and same-patch RBF ranges."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from generate_global_anchors import _face_geometry, _normal_patches, _parse_obj


class MeshViz:
    def __init__(self, triangles: np.ndarray, centroids: np.ndarray, patches: np.ndarray, part_ids: np.ndarray, face_ids: np.ndarray):
        self.triangles = triangles
        self.centroids = centroids
        self.patches = patches
        self.part_ids = part_ids
        self.face_ids = face_ids


def _set_equal_3d(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    span = max(span, 1e-6)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - 0.55 * span, c + 0.55 * span)


def _nearest_neighbor_sigma(points: np.ndarray, patch_ids: np.ndarray) -> float:
    nearest = []
    for patch in np.unique(patch_ids):
        idx = np.flatnonzero(patch_ids == patch)
        if len(idx) < 2:
            continue
        p = points[idx]
        dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        dist[dist <= 1e-12] = np.inf
        nearest.extend(np.min(dist, axis=1).tolist())
    if not nearest:
        return 0.02
    return float(np.median(nearest) * 2.5)


def _load_mesh_viz(data: dict[str, np.ndarray], max_faces: int, sharp_angle_deg: float | None = None, seed: int = 13) -> MeshViz | None:
    if max_faces <= 0 or "mesh_paths" not in data:
        return None
    mesh_paths = [str(x) for x in data["mesh_paths"]]
    scale = float(np.asarray(data.get("mesh_scale", 0.001)))
    if sharp_angle_deg is None:
        sharp_angle_deg = float(np.asarray(data.get("sharp_angle_deg", 35.0)))
    rng = np.random.default_rng(seed)
    tris = []
    cents = []
    patches_all = []
    part_ids_all = []
    face_ids_all = []
    per_mesh = max(1, max_faces // max(1, len(mesh_paths)))
    patch_offset = 0
    for part_id, mesh_path in enumerate(mesh_paths):
        path = Path(mesh_path)
        if not path.exists():
            continue
        mesh = _parse_obj(path, scale)
        normals, centroids, _areas = _face_geometry(mesh.vertices, mesh.faces)
        patches = _normal_patches(mesh.faces, normals, sharp_angle_deg) + patch_offset
        face_ids = np.arange(len(mesh.faces), dtype=np.int64)
        sel = face_ids
        if len(sel) > per_mesh:
            sel = rng.choice(sel, size=per_mesh, replace=False)
        tris.append(mesh.vertices[mesh.faces[sel]])
        cents.append(centroids[sel])
        patches_all.append(patches[sel])
        part_ids_all.append(np.full(len(sel), part_id, dtype=np.int64))
        face_ids_all.append(sel)
        patch_offset += int(np.max(patches)) + 1
    if not tris:
        return None
    return MeshViz(
        triangles=np.concatenate(tris, axis=0),
        centroids=np.concatenate(cents, axis=0),
        patches=np.concatenate(patches_all, axis=0),
        part_ids=np.concatenate(part_ids_all, axis=0),
        face_ids=np.concatenate(face_ids_all, axis=0),
    )


def _add_mesh_gray(ax, mesh: MeshViz | None, alpha: float = 0.16) -> None:
    if mesh is None:
        return
    coll = Poly3DCollection(mesh.triangles, alpha=alpha, linewidths=0.04)
    coll.set_facecolor((0.72, 0.76, 0.80, alpha))
    coll.set_edgecolor((0.45, 0.48, 0.52, 0.10))
    ax.add_collection3d(coll)


def _add_mesh_weighted(ax, mesh: MeshViz, weights: np.ndarray) -> Poly3DCollection:
    cmap = plt.get_cmap("viridis")
    colors = np.zeros((len(weights), 4), dtype=np.float64)
    inactive = weights <= 0.0
    colors[inactive] = (0.72, 0.74, 0.76, 0.16)
    colors[~inactive] = cmap(np.clip(weights[~inactive], 0.0, 1.0))
    colors[~inactive, 3] = 0.88
    coll = Poly3DCollection(mesh.triangles, linewidths=0.025)
    coll.set_facecolor(colors)
    coll.set_edgecolor((0.30, 0.30, 0.30, 0.05))
    ax.add_collection3d(coll)
    return coll


def _choose_examples(points: np.ndarray, normals: np.ndarray, patch_ids: np.ndarray, count: int) -> np.ndarray:
    """Pick anchors on patches with distinct normals and enough neighbors."""
    patch_scores = []
    for patch in np.unique(patch_ids):
        idx = np.flatnonzero(patch_ids == patch)
        if len(idx) < 4:
            continue
        mean_n = normals[idx].mean(axis=0)
        norm = np.linalg.norm(mean_n)
        if norm > 1e-8:
            mean_n = mean_n / norm
        patch_scores.append((len(idx), int(patch), mean_n))
    patch_scores.sort(reverse=True, key=lambda x: x[0])

    chosen = []
    chosen_normals = []
    for _size, patch, mean_n in patch_scores:
        if chosen_normals and max(float(abs(np.dot(mean_n, n))) for n in chosen_normals) > 0.82:
            continue
        idx = np.flatnonzero(patch_ids == patch)
        center = points[idx].mean(axis=0)
        anchor = idx[np.argmin(np.linalg.norm(points[idx] - center, axis=1))]
        chosen.append(int(anchor))
        chosen_normals.append(mean_n)
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        for _size, patch, _mean_n in patch_scores:
            idx = np.flatnonzero(patch_ids == patch)
            center = points[idx].mean(axis=0)
            anchor = int(idx[np.argmin(np.linalg.norm(points[idx] - center, axis=1))])
            if anchor not in chosen:
                chosen.append(anchor)
            if len(chosen) >= count:
                break
    return np.asarray(chosen[:count], dtype=np.int64)


def plot_overview(data: dict[str, np.ndarray], out: Path, mesh: MeshViz | None) -> None:
    points = data["points_obj"]
    force_dirs = data["force_dirs_obj"]
    patch_ids = data["patch_ids"]
    edges = data["neighbor_edges"]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh_gray(ax, mesh)
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=patch_ids,
        cmap="tab20",
        s=18 if len(points) > 500 else 28,
        depthshade=False,
    )
    if len(edges) and len(points) <= 400:
        for a, b in edges:
            pa, pb = points[a], points[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color="0.78", linewidth=0.45, alpha=0.45)

    step = max(1, len(points) // 70)
    q = np.arange(0, len(points), step)
    ax.quiver(
        points[q, 0],
        points[q, 1],
        points[q, 2],
        force_dirs[q, 0],
        force_dirs[q, 1],
        force_dirs[q, 2],
        length=0.018,
        color="black",
        linewidth=0.65,
        normalize=True,
    )
    _set_equal_3d(ax, points)
    ax.view_init(elev=24, azim=-52)
    ax.set_title("Global anchors on high-res mesh\ncolor = sharp-edge patch, black arrows = normal push directions")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.08)
    cbar.set_label("patch id")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _mesh_patch_for_anchor(data: dict[str, np.ndarray], mesh: MeshViz, anchor: int) -> int | None:
    part = int(data["part_ids"][anchor])
    face = int(data["source_face_ids"][anchor])
    mask = (mesh.part_ids == part) & (mesh.face_ids == face)
    if np.any(mask):
        return int(mesh.patches[np.flatnonzero(mask)[0]])
    # If the exact source face was not sampled for visualization, use nearest sampled face on same part.
    part_mask = mesh.part_ids == part
    if not np.any(part_mask):
        return None
    idx = np.flatnonzero(part_mask)
    p = data["points_obj"][anchor]
    nearest = idx[np.argmin(np.linalg.norm(mesh.centroids[idx] - p, axis=1))]
    return int(mesh.patches[nearest])


def plot_rbf_examples(data: dict[str, np.ndarray], out: Path, sigma: float | None, examples: int, mesh: MeshViz | None) -> None:
    points = data["points_obj"]
    normals = data["normals_obj"]
    patch_ids = data["patch_ids"]
    if sigma is None or sigma <= 0.0:
        sigma = _nearest_neighbor_sigma(points, patch_ids)
    example_idx = _choose_examples(points, normals, patch_ids, examples)

    fig = plt.figure(figsize=(5.2 * len(example_idx), 5.0))
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    for col, anchor in enumerate(example_idx, start=1):
        ax = fig.add_subplot(1, len(example_idx), col, projection="3d")
        if mesh is not None:
            mesh_patch = _mesh_patch_for_anchor(data, mesh, int(anchor))
            weights = np.zeros(len(mesh.triangles), dtype=np.float64)
            if mesh_patch is not None:
                same = mesh.patches == mesh_patch
                dist = np.linalg.norm(mesh.centroids - points[anchor], axis=1)
                weights[same] = np.exp(-(dist[same] ** 2) / (2.0 * sigma**2))
            _add_mesh_weighted(ax, mesh, weights)
        else:
            same = patch_ids == patch_ids[anchor]
            dist = np.linalg.norm(points - points[anchor], axis=1)
            weight = np.zeros(len(points), dtype=np.float64)
            weight[same] = np.exp(-(dist[same] ** 2) / (2.0 * sigma**2))
            ax.scatter(points[~same, 0], points[~same, 1], points[~same, 2], color="0.70", s=8, alpha=0.35)
            ax.scatter(points[same, 0], points[same, 1], points[same, 2], c=weight[same], cmap="viridis", vmin=0.0, vmax=1.0, s=22, depthshade=False)

        ax.scatter(
            [points[anchor, 0]],
            [points[anchor, 1]],
            [points[anchor, 2]],
            color="red",
            marker="*",
            s=180,
            depthshade=False,
        )
        _set_equal_3d(ax, points)
        ax.view_init(elev=24, azim=-52)
        ax.set_title(f"anchor {int(anchor)}: patch {int(patch_ids[anchor])}\ncolored surface = same-patch RBF range")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    fig.colorbar(mappable, ax=fig.axes, shrink=0.7, pad=0.02, label="RBF influence on mesh surface")
    fig.suptitle(f"RBF influence is rendered on mesh faces, sigma={sigma:.4f} m; sharp edges block influence", y=1.02)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", default="global_anchor_force_field/data/box_global_anchors_200.npz")
    parser.add_argument("--out-dir", default="global_anchor_force_field/results/visualizations")
    parser.add_argument("--sigma", type=float, default=0.0, help="RBF sigma in meters. <=0 uses median NN distance * 2.5.")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--mesh-face-sample", type=int, default=20000, help="Number of high-res mesh faces to show. <=0 hides mesh.")
    args = parser.parse_args()

    data = dict(np.load(args.anchors, allow_pickle=True))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.anchors).stem
    mesh = _load_mesh_viz(data, args.mesh_face_sample)

    overview = out_dir / f"{stem}_overview.png"
    examples = out_dir / f"{stem}_rbf_ranges.png"
    plot_overview(data, overview, mesh)
    plot_rbf_examples(data, examples, args.sigma, max(1, int(args.examples)), mesh)
    print(f"Saved: {overview}")
    print(f"Saved: {examples}")


if __name__ == "__main__":
    main()

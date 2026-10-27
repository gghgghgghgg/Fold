"""Generate full-surface normal-force anchors for an articulated object.

This first version is intentionally self-contained for server transfer:
it only depends on NumPy and reads Wavefront OBJ meshes directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MeshPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray


def _parse_obj(path: Path, scale: float) -> MeshPart:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                vals = line.strip().split()
                vertices.append([float(vals[1]), float(vals[2]), float(vals[3])])
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                idx = []
                for token in tokens:
                    # OBJ indices are 1-based. Handles v, v/vt, v/vt/vn, v//vn.
                    idx.append(int(token.split("/")[0]) - 1)
                if len(idx) < 3:
                    continue
                # Fan-triangulate polygons.
                for j in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[j], idx[j + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ has no vertices/faces: {path}")
    return MeshPart(
        name=path.stem,
        vertices=np.asarray(vertices, dtype=np.float64) * float(scale),
        faces=np.asarray(faces, dtype=np.int64),
    )


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    raw_normals = np.cross(e1, e2)
    dbl_area = np.linalg.norm(raw_normals, axis=1)
    valid = dbl_area > 1e-14
    normals = np.zeros_like(raw_normals)
    normals[valid] = raw_normals[valid] / dbl_area[valid, None]
    centroids = tri.mean(axis=1)

    # Make normals consistently outward for closed-ish meshes.
    mesh_center = vertices.mean(axis=0)
    inward = np.einsum("ij,ij->i", normals, centroids - mesh_center) < 0.0
    normals[inward] *= -1.0
    areas = 0.5 * dbl_area
    return normals, centroids, areas


def _face_adjacency(faces: np.ndarray) -> list[list[int]]:
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edge_to_faces.setdefault(key, []).append(fi)

    neigh = [[] for _ in range(len(faces))]
    for linked in edge_to_faces.values():
        if len(linked) < 2:
            continue
        for i in linked:
            for j in linked:
                if i != j:
                    neigh[i].append(j)
    return [sorted(set(x)) for x in neigh]


def _normal_patches(faces: np.ndarray, normals: np.ndarray, angle_deg: float) -> np.ndarray:
    """Connected components that do not cross sharp normal changes."""
    cos_thr = float(np.cos(np.deg2rad(angle_deg)))
    neigh = _face_adjacency(faces)
    patch = -np.ones(len(faces), dtype=np.int64)
    patch_id = 0
    for seed in range(len(faces)):
        if patch[seed] >= 0:
            continue
        stack = [seed]
        patch[seed] = patch_id
        while stack:
            cur = stack.pop()
            for nb in neigh[cur]:
                if patch[nb] >= 0:
                    continue
                if float(np.dot(normals[cur], normals[nb])) >= cos_thr:
                    patch[nb] = patch_id
                    stack.append(nb)
        patch_id += 1
    return patch


def _sample_points_on_faces(
    rng: np.random.Generator,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    patches: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = areas / np.sum(areas)
    face_idx = rng.choice(len(faces), size=count, replace=True, p=probs)
    tri = vertices[faces[face_idx]]
    u = rng.random(count)
    v = rng.random(count)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])
    return points, normals[face_idx], patches[face_idx], face_idx


def _allocate_counts(areas: np.ndarray, total: int) -> np.ndarray:
    raw = areas / np.sum(areas) * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(np.sum(counts))
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    return counts


def _build_neighbor_edges(points: np.ndarray, patch_ids: np.ndarray, k: int, radius: float) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for patch in np.unique(patch_ids):
        idx = np.flatnonzero(patch_ids == patch)
        if len(idx) <= 1:
            continue
        p = points[idx]
        diff = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        for local_i, global_i in enumerate(idx):
            order = np.argsort(dist[local_i])
            added = 0
            for local_j in order:
                if local_i == local_j:
                    continue
                if radius > 0.0 and dist[local_i, local_j] > radius:
                    continue
                a, b = sorted((int(global_i), int(idx[local_j])))
                edges.add((a, b))
                added += 1
                if added >= k:
                    break
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(sorted(edges), dtype=np.int64)


def generate_global_anchors(
    mesh_paths: list[Path],
    part_names: list[str],
    total_anchors: int,
    scale: float,
    sharp_angle_deg: float,
    neighbor_k: int,
    neighbor_radius: float,
    normal_force_cap: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = [_parse_obj(path, scale) for path in mesh_paths]
    if part_names:
        if len(part_names) != len(parts):
            raise ValueError("--part-names length must match --meshes length")
        for part, name in zip(parts, part_names):
            part.name = name

    part_areas = []
    face_cache = []
    for part in parts:
        normals, _centroids, areas = _face_geometry(part.vertices, part.faces)
        patches = _normal_patches(part.faces, normals, sharp_angle_deg)
        part_areas.append(float(np.sum(areas)))
        face_cache.append((normals, areas, patches))

    counts = _allocate_counts(np.asarray(part_areas, dtype=np.float64), total_anchors)

    all_points = []
    all_normals = []
    all_force_dirs = []
    all_patch_ids = []
    all_part_ids = []
    all_face_ids = []
    patch_offset = 0
    for part_id, (part, count) in enumerate(zip(parts, counts)):
        normals, areas, patches = face_cache[part_id]
        pts, nrm, patch, face_idx = _sample_points_on_faces(
            rng, part.vertices, part.faces, normals, areas, patches, int(count)
        )
        all_points.append(pts)
        all_normals.append(nrm)
        all_force_dirs.append(-nrm)
        all_patch_ids.append(patch + patch_offset)
        all_part_ids.append(np.full(count, part_id, dtype=np.int64))
        all_face_ids.append(face_idx)
        patch_offset += int(np.max(patches)) + 1

    points = np.concatenate(all_points, axis=0)
    normals = np.concatenate(all_normals, axis=0)
    force_dirs = np.concatenate(all_force_dirs, axis=0)
    patch_ids = np.concatenate(all_patch_ids, axis=0).astype(np.int64)
    part_ids = np.concatenate(all_part_ids, axis=0).astype(np.int64)
    face_ids = np.concatenate(all_face_ids, axis=0).astype(np.int64)
    unique_patches = np.unique(patch_ids)
    patch_remap = {int(old): new for new, old in enumerate(unique_patches)}
    patch_ids = np.asarray([patch_remap[int(x)] for x in patch_ids], dtype=np.int64)
    neighbor_edges = _build_neighbor_edges(points, patch_ids, neighbor_k, neighbor_radius)

    lower = np.zeros(len(points), dtype=np.float64)
    upper = np.full(len(points), float(normal_force_cap), dtype=np.float64)
    if normal_force_cap <= 0.0:
        upper[:] = np.inf

    return {
        "points_obj": points.astype(np.float32),
        "normals_obj": normals.astype(np.float32),
        "force_dirs_obj": force_dirs.astype(np.float32),
        "patch_ids": patch_ids.astype(np.int32),
        "part_ids": part_ids.astype(np.int32),
        "source_face_ids": face_ids.astype(np.int32),
        "neighbor_edges": neighbor_edges.astype(np.int32),
        "lambda_lower": lower.astype(np.float32),
        "lambda_upper": upper.astype(np.float32),
        "part_names": np.asarray([p.name for p in parts]),
        "mesh_paths": np.asarray([str(p) for p in mesh_paths]),
        "mesh_scale": np.asarray(scale, dtype=np.float32),
        "sharp_angle_deg": np.asarray(sharp_angle_deg, dtype=np.float32),
        "neighbor_k": np.asarray(neighbor_k, dtype=np.int32),
        "neighbor_radius": np.asarray(neighbor_radius, dtype=np.float32),
        "normal_force_cap": np.asarray(normal_force_cap, dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full-surface normal-force anchors.")
    parser.add_argument("--meshes", nargs="+", required=True, help="OBJ meshes, one per object part.")
    parser.add_argument("--part-names", nargs="*", default=None, help="Optional names matching --meshes.")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--num-anchors", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.001, help="OBJ-to-meter scale.")
    parser.add_argument("--sharp-angle-deg", type=float, default=35.0)
    parser.add_argument("--neighbor-k", type=int, default=6)
    parser.add_argument("--neighbor-radius", type=float, default=0.0, help="0 disables radius cutoff.")
    parser.add_argument("--normal-force-cap", type=float, default=100.0, help="<=0 means no upper cap.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    meshes = [Path(x) for x in args.meshes]
    for mesh in meshes:
        if not mesh.exists():
            raise FileNotFoundError(mesh)

    data = generate_global_anchors(
        mesh_paths=meshes,
        part_names=list(args.part_names or []),
        total_anchors=int(args.num_anchors),
        scale=float(args.scale),
        sharp_angle_deg=float(args.sharp_angle_deg),
        neighbor_k=int(args.neighbor_k),
        neighbor_radius=float(args.neighbor_radius),
        normal_force_cap=float(args.normal_force_cap),
        seed=int(args.seed),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f"Saved anchors: {out}")
    print(f"anchors={len(data['points_obj'])} patches={len(np.unique(data['patch_ids']))} edges={len(data['neighbor_edges'])}")
    for part_id, name in enumerate(data["part_names"]):
        print(f"part[{part_id}] {name}: {int(np.sum(data['part_ids'] == part_id))} anchors")
    print("normal force bounds: lambda_i in [0, cap]" if args.normal_force_cap > 0 else "normal force bounds: lambda_i >= 0")


if __name__ == "__main__":
    main()

"""Smoke test for box full-surface anchors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate_global_anchors import generate_global_anchors


def _check(data: dict[str, np.ndarray], expected_count: int) -> None:
    points = data["points_obj"]
    normals = data["normals_obj"]
    force_dirs = data["force_dirs_obj"]
    patch_ids = data["patch_ids"]
    edges = data["neighbor_edges"]

    assert points.shape == (expected_count, 3), points.shape
    assert normals.shape == (expected_count, 3), normals.shape
    assert force_dirs.shape == (expected_count, 3), force_dirs.shape
    assert patch_ids.shape == (expected_count,), patch_ids.shape
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-4)
    assert np.allclose(force_dirs, -normals, atol=1e-6)
    assert np.all(data["lambda_lower"] == 0.0)
    assert np.all(data["lambda_upper"] > 0.0)
    assert len(np.unique(patch_ids)) >= 6, "box should have multiple sharp-edge patches"
    assert edges.ndim == 2 and edges.shape[1] == 2
    part_ids = data["part_ids"]
    for patch in np.unique(patch_ids):
        assert len(np.unique(part_ids[patch_ids == patch])) == 1, "patch crossed an articulated part boundary"
    if len(edges) > 0:
        assert np.all(patch_ids[edges[:, 0]] == patch_ids[edges[:, 1]]), "neighbor edge crossed a patch boundary"
        assert np.all(part_ids[edges[:, 0]] == part_ids[edges[:, 1]]), "neighbor edge crossed an articulated part boundary"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--box-dir",
        default=r"D:\retarget\unpack\arctic_data\data\meta\object_vtemplates\box",
        help="Directory containing bottom.obj and top.obj.",
    )
    parser.add_argument("--out", default="global_anchor_force_field/data/box_global_anchors_200.npz")
    parser.add_argument("--num-anchors", type=int, default=200)
    args = parser.parse_args()

    box_dir = Path(args.box_dir)
    meshes = [box_dir / "bottom.obj", box_dir / "top.obj"]
    data = generate_global_anchors(
        mesh_paths=meshes,
        part_names=["base", "lid"],
        total_anchors=args.num_anchors,
        scale=0.001,
        sharp_angle_deg=35.0,
        neighbor_k=6,
        neighbor_radius=0.0,
        normal_force_cap=100.0,
        seed=7,
    )
    _check(data, args.num_anchors)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f"OK: generated {args.num_anchors} anchors at {out}")
    print(f"patches={len(np.unique(data['patch_ids']))}, edges={len(data['neighbor_edges'])}")
    for part_id, name in enumerate(data["part_names"]):
        print(f"{name}: {int(np.sum(data['part_ids'] == part_id))} anchors")


if __name__ == "__main__":
    main()

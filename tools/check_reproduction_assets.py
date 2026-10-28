#!/usr/bin/env python3
"""Check the local assets needed by the FoLD Box clip reproduction example."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch


def resolve_mesh(urdf_path: Path, filename: str, assets_root: Path) -> Path:
    if filename.startswith("package://"):
        return assets_root / "inspire_hand" / filename[len("package://") :]
    path = Path(filename)
    return path if path.is_absolute() else urdf_path.parent / path


def inspect_urdf(path: Path, assets_root: Path, missing: set[Path]) -> None:
    if not path.is_file():
        missing.add(path)
        return
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        print(f"Invalid URDF {path}: {exc}", file=sys.stderr)
        missing.add(path)
        return
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename:
            mesh_path = resolve_mesh(path, filename, assets_root)
            if not mesh_path.is_file():
                missing.add(mesh_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--arctic-root", type=Path, required=True,
                        help="ARCTIC data root containing data/meta/object_vtemplates")
    parser.add_argument("--object", default="box")
    parser.add_argument("--subject", default="s01")
    parser.add_argument("--task", default="use_01")
    parser.add_argument("--retarget", default="para")
    parser.add_argument("--clip-end", type=int, default=230,
                        help="Exclusive end frame; both sequence and retarget must cover it")
    parser.add_argument("--regenerate-retarget", action="store_true",
                        help="Also require the hand retarget_config.yaml")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    assets = root / "fold/assets"
    obj = args.object
    stem = f"{obj}_{args.task}"
    processed = assets / f"arctic/processed/{args.subject}/{stem}.npy"
    retarget = assets / f"retargeted/inspire_hand/{args.subject}/{stem}_vector_{args.retarget}.pt"
    object_urdf = assets / f"arctic/{obj}/decomp/{obj}_decomp.urdf"
    template_root = args.arctic_root.resolve() / "data/meta/object_vtemplates" / obj
    templates = [template_root / f"{part}.obj" for part in ("bottom", "top")]
    hand_urdfs = [assets / "inspire_hand/left_xyz_copy.urdf",
                  assets / "inspire_hand/right_xyz_copy.urdf"]
    missing: set[Path] = set()

    for path in [processed, retarget, object_urdf, *templates, *hand_urdfs]:
        if not path.is_file():
            missing.add(path)
    if args.regenerate_retarget:
        config = assets / "inspire_hand/retarget_config.yaml"
        if not config.is_file():
            missing.add(config)

    if object_urdf.is_file():
        inspect_urdf(object_urdf, assets, missing)
    for hand_urdf in hand_urdfs:
        inspect_urdf(hand_urdf, assets, missing)

    frame_errors: list[str] = []
    if processed.is_file():
        try:
            data = np.load(processed, allow_pickle=True).item()
            frames = len(data["params"]["obj_trans"])
            if frames < args.clip_end:
                frame_errors.append(f"processed sequence has {frames} frames; need at least {args.clip_end}")
        except Exception as exc:  # report malformed or incompatible user asset
            frame_errors.append(f"cannot read processed sequence {processed}: {exc}")
    if retarget.is_file():
        try:
            data = torch.load(retarget, map_location="cpu", weights_only=False)
            frames = len(data["demo_data"]["obj_pos"])
            if frames < args.clip_end:
                frame_errors.append(f"retarget has {frames} frames; need at least {args.clip_end}")
        except Exception as exc:  # report malformed or incompatible user asset
            frame_errors.append(f"cannot read retarget trajectory {retarget}: {exc}")

    if missing or frame_errors:
        print("Asset check failed.")
        for path in sorted(missing):
            print(f"MISSING: {path}")
        for error in frame_errors:
            print(f"INVALID: {error}")
        return 1

    print(f"Asset check passed for {obj}/{args.subject}/{args.task}, frames [0,{args.clip_end}).")
    print(f"processed: {processed}")
    print(f"retarget:  {retarget}")
    print(f"object:    {object_urdf}")
    print(f"templates: {templates[0]}, {templates[1]}")
    print(f"hands:     {hand_urdfs[0]}, {hand_urdfs[1]}")
    if args.regenerate_retarget:
        print(f"retarget config: {assets / 'inspire_hand/retarget_config.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

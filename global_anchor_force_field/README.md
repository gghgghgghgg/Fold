# Global Anchor Force Field

This folder is a self-contained prototype for online/global object-surface force guidance.

The first version generates global anchors over the full object surface instead of using a
fixed human-demo contact patch.

## Representation

Each anchor stores:

- `points_obj`: anchor position in the object-local frame.
- `normals_obj`: outward surface normal in the object-local frame.
- `force_dirs_obj`: allowed normal push direction, defined as `-normals_obj`.
- `patch_ids`: smooth-surface patch id. Sharp edges are boundaries.
- `neighbor_edges`: anchor graph edges used for smoothness. Edges never cross patch boundaries.
- `lambda_lower`, `lambda_upper`: normal-force coefficient bounds. The first test uses `lambda_i in [0, 100]`.

The key rule is that continuity is only defined inside a smooth patch. For a box, anchors on
different faces are close in Euclidean distance near an edge, but their RBF/smoothness influence
must not cross the edge. Base/lid part boundaries are also hard boundaries and must not transmit anchor influence.

## Box Smoke Test

Run from the repository root:

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\test_box_global_anchors.py
```

Expected output:

```text
OK: generated 200 anchors at global_anchor_force_field/data/box_global_anchors_200.npz
patches=..., edges=...
base: ... anchors
lid: ... anchors
```

## Direct Generation

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\generate_global_anchors.py `
  --meshes D:\retarget\unpack\arctic_data\data\meta\object_vtemplates\box\bottom.obj `
           D:\retarget\unpack\arctic_data\data\meta\object_vtemplates\box\top.obj `
  --part-names base lid `
  --num-anchors 200 `
  --scale 0.001 `
  --out global_anchor_force_field\data\box_global_anchors_200.npz
```

## Visualization

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\visualize_global_anchors.py
```

This writes:

- `results/visualizations/box_global_anchors_200_overview.png`: anchors colored by sharp-edge patch, with normal push directions.
- `results/visualizations/box_global_anchors_200_rbf_ranges.png`: several example anchors and their same-patch RBF influence ranges.


## One-Frame 7D Fit and Force Field

The one-frame test fits full 7D generalized force `[Fx,Fy,Fz,Tx,Ty,Tz,tau_hinge]` using global normal-only anchors. For the box frame-60 test, 200 anchors with `--hinge-w 200` reached Genesis one-step error `0.647 mm / 0.653 deg / 0.007 deg` without bias.

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\one_frame_global_anchor_fit.py `
  --anchors global_anchor_force_field\data\box_global_anchors_200.npz `
  --frame 60 `
  --bias-bound 0 `
  --smooth-w 0.01 `
  --hinge-w 200 `
  --genesis-step `
  --out global_anchor_force_field\results\one_frame\one_frame_box_fit_frame60_200a_7d_hingew200_genesis.npz `
  --plot global_anchor_force_field\results\visualizations\one_frame_box_fit_frame60_200a_7d_hingew200_genesis.png
```

Visualize the estimated force field with a real mesh renderer:

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\visualize_one_frame_force_field.py
```

This writes:

- `results/visualizations/one_frame_box_fit_frame60_200a_force_field_mesh_strength.png`: clean same-patch RBF-spread force-strength map rendered on the full OBJ mesh, with no anchor markers. The default `jet` colormap maps low strength to blue and high strength to red.

For a rotating strength-only GIF:

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\visualize_one_frame_force_field.py --rotate-gif --gif-frames 48 --gif-fps 12 --width 900 --height 700
```

This writes `results/visualizations/one_frame_box_fit_frame60_200a_force_field_mesh_strength_rotate.gif`. Use `--cmap turbo` or `--cmap coolwarm` if a different blue-red style is preferred.

## Next Step

The next module should solve online normal-force coefficients:

```text
min ||A lambda - tau_des||^2
  + spatial_smooth * sum_(i,j in same patch) (lambda_i - lambda_j)^2
  + temporal_smooth * ||lambda - lambda_prev||^2
s.t. lambda_i >= 0
```

where `A` maps normal anchor forces to object-level wrench.




## Closed-Loop Global Anchor Rollout

Normal-only rollout, no bias:

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\rollout_global_anchor_force.py --num-frames 20 --print-every 5 --out global_anchor_force_field\results\rollout\global_anchor_200_box_20f.npz
```

Friction-pyramid rollout, still no bias. Each anchor uses variables `[fn,t1+,t1-,t2+,t2-]` and enforces `t1+ + t1- + t2+ + t2- <= mu * fn`:

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\rollout_global_anchor_force.py `
  --contact-model friction_pyramid `
  --mu 0.1 `
  --num-frames 20 `
  --print-every 5 `
  --out global_anchor_force_field\results\rollout\global_anchor_200_box_20f_friction_mu01.npz
```

For the first unstable segment:

```powershell
D:\anaconda\envs\phyretarget\python.exe global_anchor_force_field\rollout_global_anchor_force.py `
  --contact-model friction_pyramid `
  --mu 0.1 `
  --start-frame 80 `
  --num-frames 80 `
  --print-every 10 `
  --out global_anchor_force_field\results\rollout\global_anchor_200_box_seg80_80f_friction_mu01.npz
```

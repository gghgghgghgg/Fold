# Example sequence data

The local release-candidate workspace has three complete `s01/use_01` examples staged under the same `fold/assets/` layout expected by the training code. These are ARCTIC-derived files. They remain Git-ignored and have not been committed or pushed while written redistribution permission is pending.

| Example | Processed sequence | Inspire retarget | Frames |
|---|---|---|---:|
| Box | `fold/assets/arctic/processed/s01/box_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt` | 889 |
| Ketchup | `fold/assets/arctic/processed/s01/ketchup_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/ketchup_use_01_vector_para.pt` | 697 |
| Notebook | `fold/assets/arctic/processed/s01/notebook_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/notebook_use_01_vector_para.pt` | 669 |

The processed `.npy` files contain FoLD-compatible `params` and `world_coord` dictionaries. Each paired Inspire `para` retarget file covers the same full frame count. The Box entry is the worked example used in [`training/`](../training/README.md); Ketchup and Notebook are additional examples for data layout and task variation.

SHA256 for the four files newly copied into the candidate workspace:

```text
6dfb7ffff186755cb4bc9c3be2b2572b6b3b4039cc3d5e220bb44ab79f5a2df7  ketchup_use_01.npy
6b3d0e6eedeaaa23465f4bcd8d31e2ff934c2e2003c031a4f8e5ced7983db968  notebook_use_01.npy
3ba8d4906e9f011cdb3aec2d85682fe406226310697edc5fd69659f3dd251115  ketchup_use_01_vector_para.pt
6aa91a612ad7de835fb016b27f49a6dcd9a8352356ffcac2b4483228540e54a2  notebook_use_01_vector_para.pt
```

ARCTIC's current license prohibits providing all or part of its Data & Software to third parties without prior written permission. Do not add these local files to Git or upload them until permission covering this FoLD release is confirmed. The official ARCTIC project and licensing instructions are linked from the [external data and asset guide](EXTERNAL_ASSETS.md).

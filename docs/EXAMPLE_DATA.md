# Example sequence data

At the reviewed candidate revision, the following six ARCTIC-derived files are tracked by Git: three processed sequences and their three full-sequence Inspire retarget trajectories. A commit containing these paths includes them in its Git objects. This manifest describes the local candidate tree; it does not independently verify which commit is present on GitHub. Check the source terms and obtain any required permission before redistributing the files.

| Example | Processed sequence | Inspire retarget | Frames |
|---|---|---|---:|
| Box | `fold/assets/arctic/processed/s01/box_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt` | 889 |
| Ketchup | `fold/assets/arctic/processed/s01/ketchup_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/ketchup_use_01_vector_para.pt` | 697 |
| Notebook | `fold/assets/arctic/processed/s01/notebook_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/notebook_use_01_vector_para.pt` | 669 |

The processed `.npy` files contain FoLD-compatible `params` and `world_coord` dictionaries. Each paired Inspire `para` retarget file covers the same full frame count. The Box entry is the worked example used in [`training/`](../training/README.md); Ketchup and Notebook are additional examples for data layout and task variation.

SHA256 for all six tracked example files:

```text
228fb49a4d8657120b239e54b3eaec5d1becf68d60438a7d417e2ffe2ec8f38e  box_use_01.npy
6dfb7ffff186755cb4bc9c3be2b2572b6b3b4039cc3d5e220bb44ab79f5a2df7  ketchup_use_01.npy
6b3d0e6eedeaaa23465f4bcd8d31e2ff934c2e2003c031a4f8e5ced7983db968  notebook_use_01.npy
5d753dd05e05abe542cc775e13e289faeaa25c5ac3510c8def3ffb596806fcaf  box_use_01_vector_para.pt
3ba8d4906e9f011cdb3aec2d85682fe406226310697edc5fd69659f3dd251115  ketchup_use_01_vector_para.pt
6aa91a612ad7de835fb016b27f49a6dcd9a8352356ffcac2b4483228540e54a2  notebook_use_01_vector_para.pt
```

The files are currently tracked in this candidate tree; tracking is not evidence that redistribution permission has been granted. The official ARCTIC project and licensing instructions are linked from the [external data and asset guide](EXTERNAL_ASSETS.md). Repository maintainers must verify the applicable terms before creating or distributing a release that contains these files.

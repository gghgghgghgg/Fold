# Example sequence data

At the reviewed candidate revision, the following six ARCTIC-derived files are tracked by Git: three processed sequences and their three full-sequence Inspire retarget trajectories. A commit containing these paths includes them in its Git objects. This manifest describes the local candidate tree; it does not independently verify which commit is present on GitHub. Check the source terms and obtain any required permission before redistributing the files.

| Example | Processed sequence | Inspire retarget | Frames |
|---|---|---|---:|
| Box | `fold/assets/arctic/processed/s01/box_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt` | 889 |
| Ketchup | `fold/assets/arctic/processed/s01/ketchup_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/ketchup_use_01_vector_para.pt` | 697 |
| Notebook | `fold/assets/arctic/processed/s01/notebook_use_01.npy` | `fold/assets/retargeted/inspire_hand/s01/notebook_use_01_vector_para.pt` | 669 |

The processed `.npy` files contain FoLD-compatible `params` and `world_coord` dictionaries. Each paired Inspire `para` retarget file covers the same full frame count. The Box entry is the worked example used in [`training/`](../training/README.md); Ketchup and Notebook are additional examples for data layout and task variation.



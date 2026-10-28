# External Data and Asset Downloads

FoLD does not rehost third-party simulator models or datasets. The public [DexMachina repository](https://github.com/MandiZhao/dexmachina) says its authors released dexterous-hand and ARCTIC assets used in their preprint, and its tree contains those files. That establishes what DexMachina publishes; it does not by itself establish that FoLD has permission to mirror or redistribute each third-party asset. Obtain assets from their original source or the upstream research repository only under the terms that apply to you. Do not add raw/processed ARCTIC sequences, retargeted trajectories, generated force arrays, checkpoints, logs, or real-world arm-scene recordings to FoLD.

## Required for the paper-matched Inspire + ARCTIC pipeline

| Input | Obtain it from | Notes |
|---|---|---|
| ARCTIC sequences, object templates, and object models | [ARCTIC project site](https://arctic.is.tue.mpg.de/) and [official data instructions](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md); the [DexMachina asset tree](https://github.com/MandiZhao/dexmachina/tree/main/dexmachina/assets/arctic) also contains static object models | ARCTIC's [Data & Software License](https://github.com/zc-alexfan/arctic/blob/master/LICENSE) covers models and 3D meshes and prohibits making them available to third parties without prior written permission. Do not mirror ARCTIC assets into FoLD unless the rightsholder has granted permission covering this release. |
| Inspire hand description and meshes | [Inspire Robot](https://www.inspire-robots.com/), [DexSuite dex-urdf](https://github.com/dexsuite/dex-urdf), and the [DexMachina asset tree](https://github.com/MandiZhao/dexmachina/tree/main/dexmachina/assets/inspire_hand) | DexSuite lists the Inspire model as CC BY-NC-SA 4.0. FoLD's local URDF variants include simulator edits; confirm the exact files' provenance, provide attribution, and meet the license's noncommercial/share-alike conditions before redistributing them. |
| MANO hand model, when using MANO retargeting | [MANO project site](https://mano.is.tuebingen.mpg.de/) | Registration and acceptance of the model license are required. Download it yourself; do not redistribute the model files. |

The core examples require static object and hand models plus separately obtained ARCTIC sequences. DexMachina's release announcement explicitly says its authors released hand and ARCTIC assets for their preprint, but that does not establish that FoLD may rehost those assets. The ARCTIC license specifically restricts third-party distribution; obtain written permission before including its meshes or derived assets in FoLD. Follow each model source's notices and terms.

## Optional hand models

The code contains configurations for additional hands. These are optional and are not required for the paper-matched Inspire pipeline. Obtain a model from its original source and check the license for the exact model files before use:

| FoLD asset family | Upstream acquisition/source link | Caution |
|---|---|---|
| Allegro | [DexSuite dex-urdf](https://github.com/dexsuite/dex-urdf), which points to the [Allegro ROS description](https://github.com/WonikRobotics/allegro_hand_ros) | FoLD's local URDFs include compatibility edits. Retain upstream notices and verify the specific mesh files' license. |
| SCHUNK SVH | [DexSuite dex-urdf](https://github.com/dexsuite/dex-urdf), which points to [SCHUNK's ROS driver/model source](https://github.com/SCHUNK-SE-Co-KG/schunk_svh_ros_driver) | Confirm the model/mesh terms with SCHUNK and the source repository before downloading or adapting. |
| Ability Hand | [DexSuite dex-urdf](https://github.com/dexsuite/dex-urdf), which points to [PSYONIC's ability-hand-api](https://github.com/psyonicinc/ability-hand-api) | The dex-urdf source table lists no license for this model; obtain permission or explicit terms from PSYONIC before use. |
| Dex3 hand | [Unitree ROS dexterous-hand description](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/dexterous_hand_description/dex3_1) | The repository is BSD-3-Clause; check and retain notices for the exact model files. FoLD-compatible URDF variants are locally modified. |
| DexRobot hand | [DexRobot URDF models](https://github.com/DexRobot/dexrobot_urdf) | Read that repository's `LICENSE` and verify it covers the selected meshes and model revision. |
| MANO hand | [MANO project site](https://mano.is.tuebingen.mpg.de/) | License-gated research model; not redistributable through FoLD. |
| XHand | Obtain the exact model revision from the robot vendor or the organization that supplied it to you | The provenance of the XHand files in the research workspace has not been established well enough to recommend a public mirror. Do not substitute an unrelated model with the same name. |

Some upstream collections aggregate assets with different licenses. For example, the `dex-urdf` table lists different terms for different hands, and lists no license for Ability Hand. Do not infer that the collection's repository license overrides each model's source terms.

## Local directory layout

Keep the downloaded files out of Git and install them locally using the paths expected by the checked-out configuration. The main reproduction walkthrough uses:

```text
fold/assets/
├── arctic/
│   ├── box/
│   │   ├── decomp/box_decomp.urdf
│   │   ├── bottom_watertight_tiny.stl  # or .obj, as expected by this revision
│   │   ├── top_watertight_tiny.stl     # or .obj
│   │   └── ...                # source visual meshes/textures and object templates
│   ├── mixer/
│   ├── ...
│   └── processed/s01/       # locally prepared sequence .npy files
├── inspire_hand/            # selected Inspire URDF revision and meshes
├── mano_hand/               # only when the applicable MANO assets are authorized
├── plane/                   # simple project ground-plane mesh/URDF
└── retargeted/inspire_hand/s01/  # locally generated retarget result files
```

Use the repository's expected relative mesh paths or update the local URDF paths. Keep source notices and license files alongside the downloaded assets. Before running, check that the selected URDF's joint names, mesh paths, units, collision geometry, and inertial values match the FoLD hand/object configuration. The upstream model may need a local simulator-compatibility conversion; keep that conversion and its provenance documented locally.

The ignored `fold/assets/` path is populated locally by each user after obtaining models/data under the applicable terms. Follow [`ENVIRONMENT_SETUP_AND_REPRODUCTION.md`](ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for expected paths. No real-world arm scenes, robot-arm trajectories, camera recordings, or sensor recordings are needed for this simulation reproduction.

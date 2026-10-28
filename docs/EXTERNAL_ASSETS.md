# ARCTIC Data Preparation

FoLD's Inspire hand model and its runtime meshes are included in the repository; see [ATTRIBUTION.md](../fold/assets/inspire_hand/ATTRIBUTION.md) for their source and license. Download the ARCTIC files under the applicable terms, including the object templates used for anchor generation. Keep the ARCTIC data root available; the reproduction guide sets `ARCTIC_ROOT` to this directory. For a new sequence, prepare the ARCTIC processed file and place the FoLD-converted result at:

```text
fold/assets/arctic/processed/<subject>/<object>_<action>.npy
```

For example, `s01/use_01` on Box is saved as `fold/assets/arctic/processed/s01/box_use_01.npy`.

## Prepare a sequence

1. Download the sequence using the [official ARCTIC data instructions](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) and follow the [ARCTIC license](https://github.com/zc-alexfan/arctic/blob/master/LICENSE).
2. In the ARCTIC environment, run its sequence preprocessing. For Box `s01/use_01`, the raw file is `downloads/data/raw_seqs/s01/box_use_01.mano.npy`:

   ```bash
   python scripts_data/process_seqs.py \
     --mano_p downloads/data/raw_seqs/s01/box_use_01.mano.npy \
     --export_verts
   ```

   ARCTIC writes `outputs/processed_verts/seqs/s01/box_use_01.npy`.
3. From the FoLD repository root and its environment, convert that output into FoLD's format:

   ```bash
   export ARCTIC_REPO=/path/to/arctic
   python fold/retargeting/process_arctic.py \
     --processed_fname "$ARCTIC_REPO/outputs/processed_verts/seqs/s01/box_use_01.npy" \
     --save --save_dir fold/assets/arctic/processed
   ```

   The result is `fold/assets/arctic/processed/s01/box_use_01.npy`. Use the corresponding subject and object/action names for other sequences.

The repository's example processed files are listed in [`EXAMPLE_DATA.md`](EXAMPLE_DATA.md). For a new sequence, generate its matching full-sequence Inspire retarget with `fold/retargeting/parallel_retarget.py`; see the reproduction guide.

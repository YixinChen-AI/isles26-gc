# ISLES 2026 Dataset503 container

This directory implements the official Grand Challenge invoke API and both
ISLES 2026 outputs. The authoritative interface and gates are documented in
`../configs/container_contract.md`.

Execution policy for this project:

- do not build or run this container on the Mac;
- build-only tests use a CPU SLURM job on `brainroi`;
- model inference and end-to-end tests use one GPU SLURM job on `brainroi`;
- the immutable checkpoint archive is split into sub-2-GB assets in the public
  `YixinChen-AI/isles26-weights` GitHub Release and downloaded during the Grand
  Challenge cloud build into `/opt/app/model`;
- the small `isles26_model_manifest.json` is versioned with each algorithm tag,
  allowing default and calibrated submissions to reuse the exact same weights.

Required model layout:

```text
model/
├── isles26_model_manifest.json
└── nnUNetTrainerISLES26Baseline1000__nnUNetResEncUNetLPlans__3d_fullres/
    ├── dataset.json
    ├── plans.json
    ├── final: fold_0 ... fold_9/checkpoint_final.pth
    └── preliminary_interim: fold_0 ... fold_4/checkpoint_epoch_0500.pth
```

The checkpoints in this bundle are inference-only copies produced by
`scripts/stage_isles26_model_bundle.py`; training checkpoints remain immutable.
`do_save.sh` refuses to package without `ISLES26_MODEL_STAGE` or into an
existing artifact directory. It atomically emits both
`container.tar.gz` and `model.tar.gz`, matching the official ISLES 2026
template. The Docker context excludes local `model` artifacts. The Dockerfile
verifies the reconstructed archive against its pinned SHA-256 before extraction
and then installs the policy manifest from the algorithm tag.

The ISLES 2026 Preliminary settings must request the allowed NVIDIA T4 GPU
(16 GiB VRAM) and no more than 32 GiB main memory. The cluster E2E runs on the
project-mandated `gpu04` RTX 4090, caps the container at 32 GiB DRAM, and
fails an invoke exceeding the official 420-second per-case limit. A 4090
runtime is only a lower-bound timing signal for T4; the platform submission
remains the authoritative T4 compatibility check.

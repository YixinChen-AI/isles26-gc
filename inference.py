"""ISLES-26 (ATLAS R2.1) inference — ANTs affine MNI registration + 5-fold nnUNet ResEncL.

GC I/O:
  in : /input/images/<socket>/<uuid>.mha  (native-space skull-stripped T1w)
  out: /output/images/stroke-lesion-segmentation/output.mha  (binary mask, native space)

Pipeline:
  1. Read input (MHA/NIfTI, any orientation)
  2. ANTs affine registration -> MNI152NLin2009aSym (1mm)
  3. nnUNetv2_predict 5-fold ensemble (Dataset501, ResEncL)
  4. Apply inverse transform -> native space
  5. Write binary uint8 MHA
"""
import os
import sys
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import ants
import numpy as np
import SimpleITK as sitk

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
NNUNET_RESULTS = os.environ.get("nnUNet_results", "/opt/app/resources/nnUNet_results")
MNI_TEMPLATE = Path("/opt/app/resources/MNI152_T1_1mm.nii.gz")

DATASET_ID = "501"
CONFIG = "3d_fullres"
CHECKPOINT = "checkpoint_best"

# Output socket — try both common names for robustness
OUTPUT_SOCKETS = [
    "stroke-lesion-segmentation",
    "lesion-segmentation",
    "brain-lesion-segmentation",
]


def find_input() -> Path:
    for ext in ("*.mha", "*.mhd", "*.nii.gz", "*.nii"):
        cands = sorted(glob(str(INPUT_PATH / "images" / "**" / ext), recursive=True))
        if cands:
            return Path(cands[0])
    sys.exit(f"[FATAL] no input image found under {INPUT_PATH}; "
             f"tree={list(INPUT_PATH.rglob('*'))}")


def sitk_to_ants(sitk_img: sitk.Image) -> ants.ANTsImage:
    arr = sitk.GetArrayFromImage(sitk_img)  # Z,Y,X
    spacing = list(reversed(sitk_img.GetSpacing()))
    origin = list(reversed(sitk_img.GetOrigin()))
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
    # ANTs uses X,Y,Z convention
    ants_img = ants.from_numpy(
        arr.transpose(2, 1, 0).astype(np.float32),
        origin=list(reversed(origin)),
        spacing=list(reversed(spacing)),
    )
    return ants_img


def register_to_mni(input_path: Path, tmp_dir: str):
    """Affine-only registration to MNI152 1mm template. Returns (fwdtx, invtx)."""
    fixed = ants.image_read(str(MNI_TEMPLATE))
    moving = ants.image_read(str(input_path))

    print(f"[reg] input shape={moving.shape} spacing={moving.spacing}", flush=True)
    result = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine",
        verbose=False,
    )
    reg_path = os.path.join(tmp_dir, "ISLES26_0001_0000.nii.gz")
    ants.image_write(result["warpedmovout"], reg_path)
    print(f"[reg] registered -> {reg_path}  shape={result['warpedmovout'].shape}", flush=True)
    return result["fwdtransforms"], result["invtransforms"]


def run_nnunet(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    env = {**os.environ, "nnUNet_results": NNUNET_RESULTS,
           "nnUNet_raw": "/tmp/nnunet_raw",
           "nnUNet_preprocessed": "/tmp/nnunet_prep"}
    cmd = [
        "nnUNetv2_predict",
        "-i", input_dir,
        "-o", output_dir,
        "-d", DATASET_ID,
        "-c", CONFIG,
        "-f", "all",
        "--chk", CHECKPOINT,
        "--disable_progress_bar",
    ]
    print(f"[nnunet] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def apply_inverse_transform(mask_mni_path: str, inv_transforms, ref_native: ants.ANTsImage) -> np.ndarray:
    """Bring MNI-space binary mask back to native space."""
    mask_mni = ants.image_read(mask_mni_path)
    mask_native = ants.apply_transforms(
        fixed=ref_native,
        moving=mask_mni,
        transformlist=inv_transforms,
        interpolator="nearestNeighbor",
    )
    return (mask_native.numpy() > 0.5).astype(np.uint8)


def write_output(arr_xyz: np.ndarray, ref_sitk: sitk.Image):
    """Write binary segmentation as MHA, preserving spatial metadata of ref."""
    arr_zyx = arr_xyz.transpose(2, 1, 0)  # X,Y,Z -> Z,Y,X for SimpleITK
    out_img = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    out_img.CopyInformation(ref_sitk)

    for socket in OUTPUT_SOCKETS:
        out_dir = OUTPUT_PATH / "images" / socket
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "output.mha"
        sitk.WriteImage(out_img, str(out_path))
        print(f"[out] written -> {out_path}", flush=True)


def main():
    input_path = find_input()
    print(f"[main] input={input_path}", flush=True)

    ref_sitk = sitk.ReadImage(str(input_path))

    with tempfile.TemporaryDirectory() as tmp:
        nnunet_in = os.path.join(tmp, "nnunet_input")
        nnunet_out = os.path.join(tmp, "nnunet_output")
        os.makedirs(nnunet_in)

        # Check if input is already close to MNI shape (skip reg if so)
        size = ref_sitk.GetSize()  # X,Y,Z
        is_mni = (abs(size[0] - 197) <= 5 and abs(size[1] - 233) <= 5 and abs(size[2] - 189) <= 5)

        if is_mni:
            print("[reg] Input looks like MNI space — skipping registration", flush=True)
            reg_path = os.path.join(nnunet_in, "ISLES26_0001_0000.nii.gz")
            sitk.WriteImage(ref_sitk, reg_path)
            inv_transforms = None
        else:
            print("[reg] Registering to MNI (affine)...", flush=True)
            fwd_tx, inv_tx = register_to_mni(input_path, nnunet_in)
            inv_transforms = inv_tx

        run_nnunet(nnunet_in, nnunet_out)

        # Find output segmentation
        pred_files = sorted(glob(os.path.join(nnunet_out, "*.nii.gz")))
        if not pred_files:
            sys.exit(f"[FATAL] no prediction found in {nnunet_out}")
        pred_path = pred_files[0]
        print(f"[main] prediction={pred_path}", flush=True)

        if inv_transforms is not None:
            moving_native = ants.image_read(str(input_path))
            arr_native = apply_inverse_transform(pred_path, inv_transforms, moving_native)
        else:
            pred_sitk = sitk.ReadImage(pred_path)
            arr_native = sitk.GetArrayFromImage(pred_sitk).transpose(2, 1, 0)  # Z,Y,X -> X,Y,Z
            arr_native = (arr_native > 0.5).astype(np.uint8)

        write_output(arr_native, ref_sitk)

    print("[main] Done.", flush=True)


if __name__ == "__main__":
    main()

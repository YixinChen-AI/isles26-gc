"""ISLES-26 (ATLAS R2.1) inference — native-space 5-fold nnUNet ResEncL.

GC I/O:
  in : /input/images/<socket>/<uuid>.mha  (native-space skull-stripped T1w)
  out: /output/images/stroke-lesion-segmentation/output.mha  (binary mask, native space)

Pipeline:
  1. Read input (MHA/NIfTI, any orientation)
  2. nnUNetv2_predict 5-fold ensemble (Dataset502, ResEncL, native space)
  3. Write binary uint8 MHA
"""
import os
import sys
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
NNUNET_RESULTS = os.environ.get("nnUNet_results", "/opt/app/resources/nnUNet_results")

DATASET_ID = "502"
TRAINER = "nnUNetTrainer500epochs"
PLANS = "nnUNetResEncUNetLPlans"
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
        "-tr", TRAINER,
        "-p", PLANS,
        "-f", "all",
        "--chk", CHECKPOINT,
        "--disable_progress_bar",
    ]
    print(f"[nnunet] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


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
    print(f"[main] shape={ref_sitk.GetSize()} spacing={ref_sitk.GetSpacing()}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        nnunet_in = os.path.join(tmp, "nnunet_input")
        nnunet_out = os.path.join(tmp, "nnunet_output")
        os.makedirs(nnunet_in)

        # Write input directly in native space (no registration needed for Dataset502)
        inp_path = os.path.join(nnunet_in, "ISLES26_0001_0000.nii.gz")
        sitk.WriteImage(ref_sitk, inp_path)
        print(f"[main] wrote input -> {inp_path}", flush=True)

        run_nnunet(nnunet_in, nnunet_out)

        pred_files = sorted(glob(os.path.join(nnunet_out, "*.nii.gz")))
        if not pred_files:
            sys.exit(f"[FATAL] no prediction found in {nnunet_out}")
        pred_path = pred_files[0]
        print(f"[main] prediction={pred_path}", flush=True)

        pred_sitk = sitk.ReadImage(pred_path)
        arr_zyx = sitk.GetArrayFromImage(pred_sitk)          # Z,Y,X
        arr_xyz = (arr_zyx.transpose(2, 1, 0) > 0.5).astype(np.uint8)  # -> X,Y,Z

        write_output(arr_xyz, ref_sitk)

    print("[main] Done.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate Grand Challenge ISLES 2026 image outputs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk


def single_image(directory: Path) -> Path:
    files = sorted(
        Path(path)
        for pattern in ("*.mha", "*.mhd", "*.nii", "*.nii.gz")
        for path in glob.glob(str(directory / pattern))
    )
    if len(files) != 1:
        raise SystemExit(f"expected one image in {directory}, got {files}")
    return files[0]


def same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetSize() == right.GetSize()
        and np.allclose(left.GetSpacing(), right.GetSpacing(), atol=1e-6)
        and np.allclose(left.GetOrigin(), right.GetOrigin(), atol=1e-5)
        and np.allclose(left.GetDirection(), right.GetDirection(), atol=1e-6)
    )


def remove_small_components(
    binary: np.ndarray,
    *,
    voxel_volume_mm3: float,
    minimum_volume_mm3: float,
) -> np.ndarray:
    binary = np.asarray(binary, dtype=bool)
    if minimum_volume_mm3 <= 0 or not binary.any():
        return binary.astype(np.uint8)
    instances, count = ndi.label(
        binary, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    if count == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    sizes = np.bincount(instances.ravel())
    minimum_voxels = max(
        1, int(np.ceil(minimum_volume_mm3 / voxel_volume_mm3))
    )
    keep = sizes >= minimum_voxels
    keep[0] = False
    return keep[instances].astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    reference = sitk.ReadImage(
        str(single_image(args.input_dir / "images" / "t1-brain-mri"))
    )
    segmentation = sitk.ReadImage(
        str(
            single_image(
                args.output_dir
                / "images"
                / "stroke-lesion-segmentation"
            )
        )
    )
    probability = sitk.ReadImage(
        str(
            single_image(
                args.output_dir / "images" / "lesion-probability-map"
            )
        )
    )
    if not same_geometry(reference, segmentation):
        raise SystemExit("segmentation geometry mismatch")
    if not same_geometry(reference, probability):
        raise SystemExit("probability geometry mismatch")
    segmentation_array = sitk.GetArrayFromImage(segmentation)
    probability_array = sitk.GetArrayFromImage(probability)
    if not np.issubdtype(segmentation_array.dtype, np.integer):
        raise SystemExit(
            f"segmentation is not integer: {segmentation_array.dtype}"
        )
    if segmentation_array.size and (
        int(segmentation_array.min()) < 0
        or int(segmentation_array.max()) > 1
    ):
        raise SystemExit("segmentation is not binary")
    if probability_array.dtype != np.float32:
        raise SystemExit(
            f"probability is not Float32: {probability_array.dtype}"
        )
    if not np.isfinite(probability_array).all():
        raise SystemExit("probability contains NaN or Inf")
    if (
        probability_array.min() < -1e-6
        or probability_array.max() > 1 + 1e-6
    ):
        raise SystemExit("probability lies outside [0,1]")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = remove_small_components(
        probability_array > float(manifest["probability_threshold"]),
        voxel_volume_mm3=float(np.prod(reference.GetSpacing())),
        minimum_volume_mm3=float(
            manifest["minimum_component_volume_mm3"]
        ),
    )
    if not np.array_equal(
        expected, segmentation_array.astype(np.uint8)
    ):
        raise SystemExit(
            "binary output does not reproduce manifest threshold/component "
            "policy"
        )
    print(f"size={reference.GetSize()}")
    print(f"spacing={reference.GetSpacing()}")
    print(f"foreground_voxels={int(segmentation_array.sum())}")
    print(
        f"probability_range="
        f"[{float(probability_array.min())},{float(probability_array.max())}]"
    )
    print("CONTAINER_OUTPUT_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

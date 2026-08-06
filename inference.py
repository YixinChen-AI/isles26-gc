"""ISLES 2026 native-space nnU-Net ensemble inference."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import string
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_ROOT = Path(os.environ.get("ISLES26_MODEL_ROOT", "/opt/ml/model"))
EXPECTED_INTERFACE = ("stroke-metadata", "t1-brain-mri")
EXPECTED_DATASET = "Dataset503_ISLES26R3Raw"
SEGMENTATION_SOCKET = "stroke-lesion-segmentation"
PROBABILITY_SOCKET = "lesion-probability-map"
ALLOWED_IMAGE_SUFFIXES = (".mha", ".mhd", ".nii", ".nii.gz")


@dataclass(frozen=True)
class ModelBundle:
    predictor: nnUNetPredictor
    manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_layout(
    manifest: dict[str, Any],
) -> tuple[Path, tuple[int, ...], set[str]]:
    if manifest.get("dataset") != EXPECTED_DATASET:
        raise ValueError(
            f"expected dataset {EXPECTED_DATASET}, got "
            f"{manifest.get('dataset')}"
        )
    folds = tuple(int(fold) for fold in manifest["folds"])
    bundle_purpose = str(manifest.get("bundle_purpose", "final"))
    expected_folds = (
        tuple(range(5))
        if bundle_purpose == "preliminary_interim"
        else tuple(range(10))
    )
    if bundle_purpose not in {"final", "preliminary_interim"}:
        raise ValueError(f"unsupported bundle purpose: {bundle_purpose}")
    if folds != expected_folds:
        raise ValueError(
            f"expected {bundle_purpose} folds {expected_folds}, got {folds}"
        )
    checkpoint_name = str(manifest["checkpoint_name"])
    if (
        Path(checkpoint_name).name != checkpoint_name
        or not checkpoint_name.endswith(".pth")
    ):
        raise ValueError(f"unsafe checkpoint name: {checkpoint_name}")
    expected_checkpoint = (
        "checkpoint_epoch_0500.pth"
        if bundle_purpose == "preliminary_interim"
        else "checkpoint_final.pth"
    )
    if checkpoint_name != expected_checkpoint:
        raise ValueError(
            f"expected {bundle_purpose} checkpoint {expected_checkpoint}, "
            f"got {checkpoint_name}"
        )
    if "source_fold_mapping" in manifest:
        raise ValueError(
            f"{bundle_purpose} bundle must preserve one-to-one fold identity"
        )
    model_directory = str(manifest["model_directory"])
    if Path(model_directory).name != model_directory:
        raise ValueError(f"unsafe model directory: {model_directory}")
    threshold = float(manifest["probability_threshold"])
    minimum_volume = float(manifest["minimum_component_volume_mm3"])
    if not 0 < threshold < 1 or minimum_volume < 0:
        raise ValueError(
            f"invalid post-processing policy: {threshold}, {minimum_volume}"
        )

    expected_files = {
        f"{model_directory}/dataset.json",
        f"{model_directory}/plans.json",
        *{
            f"{model_directory}/fold_{fold}/{checkpoint_name}"
            for fold in folds
        },
    }
    declared_files = set(manifest["files"])
    if declared_files != expected_files:
        raise ValueError(
            "model file layout mismatch: "
            f"missing={sorted(expected_files - declared_files)} "
            f"extra={sorted(declared_files - expected_files)}"
        )
    hexadecimal = set(string.hexdigits)
    for relative_name, expected_sha in manifest["files"].items():
        path = Path(relative_name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or any(part in {"", "."} for part in path.parts)
        ):
            raise ValueError(f"unsafe model artifact path: {relative_name}")
        if len(expected_sha) != 64 or any(
            character not in hexadecimal for character in expected_sha
        ):
            raise ValueError(f"invalid SHA-256 for {relative_name}")
    model_dir = MODEL_ROOT / model_directory
    return model_dir, folds, expected_files


def load_manifest() -> tuple[dict[str, Any], Path, tuple[int, ...], set[str]]:
    manifest_path = MODEL_ROOT / "isles26_model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "model_directory",
        "dataset",
        "folds",
        "checkpoint_name",
        "probability_threshold",
        "minimum_component_volume_mm3",
        "files",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"model manifest lacks fields: {sorted(missing)}")
    model_dir, folds, expected_files = validate_manifest_layout(manifest)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"missing model directory: {model_dir}")
    return manifest, model_dir, folds, expected_files


def verify_model_files(
    manifest: dict[str, Any], expected_files: set[str]
) -> None:
    root = MODEL_ROOT.resolve()
    for relative_name in sorted(expected_files):
        expected_sha = manifest["files"][relative_name]
        path = MODEL_ROOT / relative_name
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"model artifact escapes model root: {path}")
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"missing model artifact: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {relative_name}: "
                f"{observed_sha} != {expected_sha}"
            )


def init_model() -> ModelBundle:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for ISLES 2026 inference")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    manifest, model_dir, folds, expected_files = load_manifest()
    verify_model_files(manifest, expected_files)

    device = torch.device("cuda", 0)
    print(
        f"[init] torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
        f"model={model_dir} folds={folds}",
        flush=True,
    )
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=folds,
        checkpoint_name=str(manifest["checkpoint_name"]),
    )
    print(f"[init] {len(folds)}-fold model loaded", flush=True)
    return ModelBundle(predictor=predictor, manifest=manifest)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def get_interface_key() -> tuple[str, ...]:
    inputs_path = INPUT_PATH / "inputs.json"
    inputs = load_json(inputs_path)
    if not isinstance(inputs, list):
        raise ValueError("inputs.json must contain a list")
    return tuple(sorted(item["socket"]["slug"] for item in inputs))


def find_single_image(directory: Path) -> Path:
    candidates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and any(path.name.lower().endswith(s) for s in ALLOWED_IMAGE_SUFFIXES)
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one T1 image in {directory}, found {candidates}"
        )
    return candidates[0]


def same_geometry(left: sitk.Image, right: sitk.Image) -> bool:
    return (
        left.GetDimension() == right.GetDimension() == 3
        and left.GetSize() == right.GetSize()
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
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    instances, count = ndi.label(binary, structure=structure)
    if count == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    sizes = np.bincount(instances.ravel())
    minimum_voxels = max(
        1, int(np.ceil(minimum_volume_mm3 / voxel_volume_mm3))
    )
    keep = sizes >= minimum_voxels
    keep[0] = False
    return keep[instances].astype(np.uint8)


def write_output(
    *,
    socket: str,
    array_zyx: np.ndarray,
    reference: sitk.Image,
    pixel_id: int,
) -> Path:
    output_dir = OUTPUT_PATH / "images" / socket
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    image = sitk.GetImageFromArray(array_zyx)
    image = sitk.Cast(image, pixel_id)
    image.CopyInformation(reference)
    output_path = output_dir / "output.mha"
    sitk.WriteImage(image, str(output_path), useCompression=True)
    reread = sitk.ReadImage(str(output_path))
    if not same_geometry(reference, reread):
        raise RuntimeError(f"output geometry mismatch for {socket}")
    return output_path


def run(model: ModelBundle) -> None:
    if model is None:
        raise RuntimeError("model is not initialized")
    interface_key = get_interface_key()
    if interface_key != EXPECTED_INTERFACE:
        raise ValueError(
            f"unsupported interface {interface_key}; "
            f"expected {EXPECTED_INTERFACE}"
        )

    t1_path = find_single_image(INPUT_PATH / "images" / "t1-brain-mri")
    metadata = load_json(INPUT_PATH / "stroke-metadata.json")
    if not isinstance(metadata, dict):
        raise ValueError("stroke-metadata.json must contain an object")
    reference = sitk.ReadImage(str(t1_path))
    if reference.GetDimension() != 3:
        raise ValueError(f"T1 must be 3D, got {reference.GetDimension()}D")
    input_array = sitk.GetArrayFromImage(reference)
    if not np.isfinite(input_array).all():
        raise ValueError("T1 contains NaN or Inf")
    print(
        f"[invoke] input={t1_path.name} size={reference.GetSize()} "
        f"spacing={reference.GetSpacing()} metadata_keys={sorted(metadata)}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="isles26-", dir="/tmp") as tmp:
        tmp_path = Path(tmp)
        nnunet_input = tmp_path / "input"
        nnunet_output = tmp_path / "output"
        nnunet_input.mkdir()
        nnunet_output.mkdir()
        staged_input = nnunet_input / "ISLES26GC_0000.nii.gz"
        sitk.WriteImage(reference, str(staged_input), useCompression=True)

        model.predictor.predict_from_files(
            str(nnunet_input),
            str(nnunet_output),
            save_probabilities=True,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        segmentation_path = nnunet_output / "ISLES26GC.nii.gz"
        probability_path = nnunet_output / "ISLES26GC.npz"
        if not segmentation_path.is_file() or not probability_path.is_file():
            raise FileNotFoundError(
                f"nnU-Net outputs missing in {nnunet_output}: "
                f"{list(nnunet_output.iterdir())}"
            )
        nnunet_segmentation = sitk.ReadImage(str(segmentation_path))
        if not same_geometry(reference, nnunet_segmentation):
            raise RuntimeError("nnU-Net prediction geometry differs from input")
        probabilities = np.load(probability_path)["probabilities"]
        if probabilities.ndim != 4 or probabilities.shape[0] != 2:
            raise ValueError(
                f"expected binary probabilities [2,Z,Y,X], got "
                f"{probabilities.shape}"
            )
        probability = np.asarray(probabilities[1], dtype=np.float32)
        if probability.shape != tuple(reversed(reference.GetSize())):
            raise ValueError(
                f"probability shape {probability.shape} does not match "
                f"input {tuple(reversed(reference.GetSize()))}"
            )
        if not np.isfinite(probability).all():
            raise ValueError("probability map contains NaN or Inf")
        if probability.min() < -1e-5 or probability.max() > 1 + 1e-5:
            raise ValueError("probability map lies outside [0,1]")
        probability = np.clip(probability, 0, 1)

        threshold = float(model.manifest["probability_threshold"])
        if not 0 < threshold < 1:
            raise ValueError(f"invalid probability threshold: {threshold}")
        binary = (probability > threshold).astype(np.uint8)
        binary = remove_small_components(
            binary,
            voxel_volume_mm3=float(np.prod(reference.GetSpacing())),
            minimum_volume_mm3=float(
                model.manifest["minimum_component_volume_mm3"]
            ),
        )

    segmentation_output = write_output(
        socket=SEGMENTATION_SOCKET,
        array_zyx=binary,
        reference=reference,
        pixel_id=sitk.sitkUInt8,
    )
    probability_output = write_output(
        socket=PROBABILITY_SOCKET,
        array_zyx=probability,
        reference=reference,
        pixel_id=sitk.sitkFloat32,
    )
    print(
        f"[invoke] segmentation={segmentation_output} "
        f"probability={probability_output} foreground={int(binary.sum())}",
        flush=True,
    )


if __name__ == "__main__":
    run(init_model())

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
MODEL_ROOT = Path(os.environ.get("ISLES26_MODEL_ROOT", "/opt/app/model"))
EXPECTED_INTERFACE = ("stroke-metadata", "t1-brain-mri")
EXPECTED_DATASET = "Dataset503_ISLES26R3Raw"
SEGMENTATION_SOCKET = "stroke-lesion-segmentation"
PROBABILITY_SOCKET = "lesion-probability-map"
ALLOWED_IMAGE_SUFFIXES = (".mha", ".mhd", ".nii", ".nii.gz")


@dataclass(frozen=True)
class ModelBundle:
    predictor: nnUNetPredictor
    manifest: dict[str, Any]
    components: tuple["ModelComponent", ...] = ()


@dataclass(frozen=True)
class ModelComponent:
    name: str
    model_dir: Path
    folds: tuple[int, ...]
    checkpoint_name: str
    weight: float
    input_mode: str = "t1"


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
    bundle_purpose = str(manifest.get("bundle_purpose", "final"))
    expected_folds = (
        tuple(range(5))
        if bundle_purpose == "preliminary_interim"
        else tuple(range(10))
    )
    if bundle_purpose not in {"final", "preliminary_interim"}:
        raise ValueError(f"unsupported bundle purpose: {bundle_purpose}")
    expected_checkpoint = (
        "checkpoint_epoch_0500.pth"
        if bundle_purpose == "preliminary_interim"
        else "checkpoint_final.pth"
    )
    if "source_fold_mapping" in manifest:
        raise ValueError(
            f"{bundle_purpose} bundle must preserve one-to-one fold identity"
        )
    threshold = float(manifest["probability_threshold"])
    minimum_volume = float(manifest["minimum_component_volume_mm3"])
    if not 0 < threshold < 1 or minimum_volume < 0:
        raise ValueError(
            f"invalid post-processing policy: {threshold}, {minimum_volume}"
        )
    policy = manifest.get("postprocessing_policy")
    if policy is not None:
        if policy.get("family") != "relative_mean_volume_confidence_guard":
            raise ValueError(f"unsupported post-processing policy: {policy}")
        fraction = float(policy["minimum_fraction_of_mean"])
        confidence = float(policy["minimum_mean_probability"])
        if not 0 < fraction < 1 or not threshold < confidence <= 1:
            raise ValueError(f"invalid guarded relative-volume policy: {policy}")
        if minimum_volume != 0:
            raise ValueError(
                "guarded relative-volume policy cannot also use absolute volume"
            )

    raw_components = manifest.get("ensemble_models")
    if raw_components is None:
        raw_components = [
            {
                "name": "model",
                "model_directory": manifest["model_directory"],
                "folds": manifest["folds"],
                "checkpoint_name": manifest["checkpoint_name"],
                "weight": 1.0,
            }
        ]
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("ensemble_models must be a nonempty list")
    if bundle_purpose == "preliminary_interim" and len(raw_components) != 1:
        raise ValueError("preliminary_interim cannot use a multi-model ensemble")
    expected_files: set[str] = set()
    component_names: set[str] = set()
    model_directories: set[str] = set()
    weights = []
    normalized_components = []
    for component in raw_components:
        name = str(component["name"])
        if not name or any(character not in string.ascii_letters + string.digits + "_-" for character in name):
            raise ValueError(f"unsafe ensemble component name: {name}")
        if name in component_names:
            raise ValueError(f"duplicate ensemble component name: {name}")
        component_names.add(name)
        model_directory = str(component["model_directory"])
        if Path(model_directory).name != model_directory:
            raise ValueError(f"unsafe model directory: {model_directory}")
        if model_directory in model_directories:
            raise ValueError(f"duplicate ensemble model directory: {model_directory}")
        model_directories.add(model_directory)
        folds = tuple(int(fold) for fold in component["folds"])
        if folds != expected_folds:
            raise ValueError(
                f"expected {bundle_purpose} folds {expected_folds}, got {folds}"
            )
        checkpoint_name = str(component["checkpoint_name"])
        if Path(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith(".pth"):
            raise ValueError(f"unsafe checkpoint name: {checkpoint_name}")
        if checkpoint_name != expected_checkpoint:
            raise ValueError(
                f"expected {bundle_purpose} checkpoint {expected_checkpoint}, got {checkpoint_name}"
            )
        weight = float(component["weight"])
        if not np.isfinite(weight) or weight <= 0 or weight > 1:
            raise ValueError(f"invalid ensemble weight for {name}: {weight}")
        weights.append(weight)
        input_mode = str(component.get("input_mode", "t1"))
        if input_mode not in {"t1", "t1_physical_contralateral"}:
            raise ValueError(
                f"unsupported input mode for {name}: {input_mode}"
            )
        normalized_components.append((model_directory, folds, checkpoint_name))
        expected_files.update(
            {
                f"{model_directory}/dataset.json",
                f"{model_directory}/plans.json",
                *{
                    f"{model_directory}/fold_{fold}/{checkpoint_name}"
                    for fold in folds
                },
            }
        )
    if not np.isclose(sum(weights), 1.0, atol=1e-8):
        raise ValueError(f"ensemble weights must sum to one: {weights}")
    blend_mode = str(
        manifest.get("ensemble_blend_mode", "weighted_probability")
    )
    if blend_mode not in {"weighted_probability", "weighted_logit"}:
        raise ValueError(f"unsupported ensemble blend mode: {blend_mode}")
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
    first_directory, first_folds, _ = normalized_components[0]
    model_dir = MODEL_ROOT / first_directory
    return model_dir, first_folds, expected_files


def model_components(manifest: dict[str, Any]) -> tuple[ModelComponent, ...]:
    raw_components = manifest.get("ensemble_models")
    if raw_components is None:
        raw_components = [
            {
                "name": "model",
                "model_directory": manifest["model_directory"],
                "folds": manifest["folds"],
                "checkpoint_name": manifest["checkpoint_name"],
                "weight": 1.0,
            }
        ]
    return tuple(
        ModelComponent(
            name=str(component["name"]),
            model_dir=MODEL_ROOT / str(component["model_directory"]),
            folds=tuple(int(fold) for fold in component["folds"]),
            checkpoint_name=str(component["checkpoint_name"]),
            weight=float(component["weight"]),
            input_mode=str(component.get("input_mode", "t1")),
        )
        for component in raw_components
    )


def load_manifest() -> tuple[dict[str, Any], Path, tuple[int, ...], set[str]]:
    manifest_path = MODEL_ROOT / "isles26_model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "dataset",
        "probability_threshold",
        "minimum_component_volume_mm3",
        "files",
    }
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"model manifest lacks fields: {sorted(missing)}")
    if "ensemble_models" not in manifest:
        legacy_required = {"model_directory", "folds", "checkpoint_name"}
        legacy_missing = legacy_required - set(manifest)
        if legacy_missing:
            raise ValueError(f"model manifest lacks fields: {sorted(legacy_missing)}")
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
    components = model_components(manifest)

    device = torch.device("cuda", 0)
    print(
        f"[init] torch={torch.__version__} device={torch.cuda.get_device_name(0)} "
        f"model={model_dir} folds={folds} components="
        f"{[(item.name, item.weight) for item in components]}",
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
        checkpoint_name=components[0].checkpoint_name,
    )
    print(f"[init] {len(folds)}-fold model loaded", flush=True)
    return ModelBundle(
        predictor=predictor,
        manifest=manifest,
        components=components,
    )


def blend_probability_maps(
    weighted_probabilities: list[tuple[float, np.ndarray]],
    *,
    mode: str = "weighted_probability",
) -> np.ndarray:
    if not weighted_probabilities:
        raise ValueError("no probability maps to blend")
    reference_shape = weighted_probabilities[0][1].shape
    total_weight = 0.0
    blended = np.zeros(reference_shape, dtype=np.float32)
    for weight, probability in weighted_probabilities:
        probability = np.asarray(probability, dtype=np.float32)
        if probability.shape != reference_shape:
            raise ValueError(
                f"ensemble probability shape mismatch: {probability.shape} != {reference_shape}"
            )
        if not np.isfinite(probability).all():
            raise ValueError("ensemble probability contains NaN or Inf")
        if mode == "weighted_probability":
            contribution = probability
        elif mode == "weighted_logit":
            epsilon = np.float32(1e-5)
            clipped = np.clip(probability, epsilon, 1 - epsilon)
            contribution = np.log(clipped) - np.log1p(-clipped)
        else:
            raise ValueError(f"unsupported ensemble blend mode: {mode}")
        blended += float(weight) * contribution
        total_weight += float(weight)
    if not np.isclose(total_weight, 1.0, atol=1e-8):
        raise ValueError(f"ensemble probability weights sum to {total_weight}")
    if mode == "weighted_logit":
        blended = 1.0 / (1.0 + np.exp(-blended))
    return np.clip(blended, 0, 1)


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


def physical_contralateral_image(reference: sitk.Image) -> sitk.Image:
    """Reproduce Dataset504's physical-LPS brain-midline reflection."""
    image = sitk.Cast(reference, sitk.sitkFloat32)
    array = sitk.GetArrayFromImage(image)
    brain = np.isfinite(array) & (np.abs(array) > 1e-6)
    coordinates = np.argwhere(brain)
    if coordinates.size == 0:
        raise ValueError("cannot reflect an empty T1 foreground")
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    physical_x = []
    for z in (minimum[0], maximum[0]):
        for y in (minimum[1], maximum[1]):
            for x in (minimum[2], maximum[2]):
                point = image.TransformIndexToPhysicalPoint(
                    (int(x), int(y), int(z))
                )
                physical_x.append(float(point[0]))
    midline_x = (min(physical_x) + max(physical_x)) / 2.0
    transform = sitk.AffineTransform(3)
    transform.SetMatrix((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    transform.SetTranslation((2.0 * midline_x, 0.0, 0.0))
    mirrored = sitk.Resample(
        image,
        image,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    if not same_geometry(image, mirrored):
        raise RuntimeError("physical contralateral channel changed geometry")
    return mirrored


def stage_component_input(
    reference: sitk.Image,
    directory: Path,
    input_mode: str,
) -> tuple[Path, ...]:
    """Write exactly the nnU-Net channels declared for one component."""
    directory.mkdir(parents=True, exist_ok=False)
    original = directory / "ISLES26GC_0000.nii.gz"
    sitk.WriteImage(reference, str(original), useCompression=True)
    if input_mode == "t1":
        return (original,)
    if input_mode != "t1_physical_contralateral":
        raise ValueError(f"unsupported component input mode: {input_mode}")
    contralateral = directory / "ISLES26GC_0001.nii.gz"
    sitk.WriteImage(
        physical_contralateral_image(reference),
        str(contralateral),
        useCompression=True,
    )
    return original, contralateral


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


def apply_postprocessing(
    probability: np.ndarray,
    manifest: dict[str, Any],
    *,
    voxel_volume_mm3: float,
) -> np.ndarray:
    threshold = float(manifest["probability_threshold"])
    if not 0 < threshold < 1:
        raise ValueError(f"invalid probability threshold: {threshold}")
    binary = np.asarray(probability > threshold, dtype=bool)
    policy = manifest.get("postprocessing_policy")
    if policy is None:
        return remove_small_components(
            binary,
            voxel_volume_mm3=voxel_volume_mm3,
            minimum_volume_mm3=float(
                manifest["minimum_component_volume_mm3"]
            ),
        )
    if policy.get("family") != "relative_mean_volume_confidence_guard":
        raise ValueError(f"unsupported post-processing policy: {policy}")
    if not binary.any():
        return np.zeros_like(binary, dtype=np.uint8)
    instances, count = ndi.label(
        binary, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    if count == 0:
        return np.zeros_like(binary, dtype=np.uint8)
    sizes = np.bincount(instances.ravel(), minlength=count + 1)
    sums = np.bincount(
        instances.ravel(),
        weights=np.asarray(probability, dtype=np.float64).ravel(),
        minlength=count + 1,
    )
    component_ids = np.arange(1, count + 1, dtype=np.int64)
    minimum = (
        float(policy["minimum_fraction_of_mean"])
        * float(sizes[1:].mean())
    )
    volume_ok = sizes[1:] >= minimum
    confidence_ok = (
        sums[1:] / sizes[1:]
        >= float(policy["minimum_mean_probability"])
    )
    selected = np.zeros(count + 1, dtype=bool)
    selected[component_ids[volume_ok | confidence_ok]] = True
    selected[0] = False
    return selected[instances].astype(np.uint8)


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

        components = model.components
        if not components:
            components = (
                ModelComponent(
                    name="preinitialized",
                    model_dir=Path("."),
                    folds=(),
                    checkpoint_name="",
                    weight=1.0,
                    input_mode="t1",
                ),
            )
        weighted_probabilities: list[tuple[float, np.ndarray]] = []
        for index, component in enumerate(components):
            component_input = nnunet_input / component.name
            staged_channels = stage_component_input(
                reference,
                component_input,
                component.input_mode,
            )
            component_output = nnunet_output / component.name
            component_output.mkdir()
            if model.components:
                print(
                    f"[invoke] loading component={component.name} "
                    f"weight={component.weight} input_mode={component.input_mode} "
                    f"channels={len(staged_channels)}",
                    flush=True,
                )
                model.predictor.initialize_from_trained_model_folder(
                    str(component.model_dir),
                    use_folds=component.folds,
                    checkpoint_name=component.checkpoint_name,
                )
            model.predictor.predict_from_files(
                str(component_input),
                str(component_output),
                save_probabilities=True,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0,
            )
            segmentation_path = component_output / "ISLES26GC.nii.gz"
            probability_path = component_output / "ISLES26GC.npz"
            if not segmentation_path.is_file() or not probability_path.is_file():
                raise FileNotFoundError(
                    f"nnU-Net outputs missing in {component_output}: "
                    f"{list(component_output.iterdir())}"
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
            component_probability = np.asarray(
                probabilities[1], dtype=np.float32
            )
            if component_probability.shape != tuple(reversed(reference.GetSize())):
                raise ValueError(
                    f"probability shape {component_probability.shape} does not match "
                    f"input {tuple(reversed(reference.GetSize()))}"
                )
            if component_probability.min() < -1e-5 or component_probability.max() > 1 + 1e-5:
                raise ValueError("probability map lies outside [0,1]")
            weighted_probabilities.append(
                (component.weight, component_probability)
            )
        probability = blend_probability_maps(
            weighted_probabilities,
            mode=str(
                model.manifest.get(
                    "ensemble_blend_mode", "weighted_probability"
                )
            ),
        )

        binary = apply_postprocessing(
            probability,
            model.manifest,
            voxel_volume_mm3=float(np.prod(reference.GetSpacing())),
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

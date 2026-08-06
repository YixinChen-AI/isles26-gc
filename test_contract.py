#!/usr/bin/env python3
"""CPU-only contract tests; this file never runs model inference."""

from __future__ import annotations

import tempfile
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import inference
import validate_outputs


class FakePredictor:
    def predict_from_files(
        self,
        input_directory,
        output_directory,
        **_,
    ):
        source = sitk.ReadImage(
            str(Path(input_directory) / "ISLES26GC_0000.nii.gz")
        )
        array = sitk.GetArrayFromImage(source)
        probability = np.zeros_like(array, dtype=np.float32)
        probability[2:4, 3:5, 4:6] = 0.9
        segmentation = sitk.GetImageFromArray(
            (probability > 0.5).astype(np.uint8)
        )
        segmentation.CopyInformation(source)
        output = Path(output_directory)
        sitk.WriteImage(
            segmentation, str(output / "ISLES26GC.nii.gz")
        )
        np.savez_compressed(
            output / "ISLES26GC.npz",
            probabilities=np.stack((1 - probability, probability)),
        )


def synthetic_manifest() -> dict:
    model_directory = "nnUNetTrainer__Plans__3d_fullres"
    checkpoint = "checkpoint_final.pth"
    files = {
        f"{model_directory}/dataset.json": "0" * 64,
        f"{model_directory}/plans.json": "1" * 64,
    }
    files.update(
        {
            f"{model_directory}/fold_{fold}/{checkpoint}": f"{fold:x}" * 64
            for fold in range(10)
        }
    )
    return {
        "dataset": inference.EXPECTED_DATASET,
        "model_directory": model_directory,
        "folds": list(range(10)),
        "checkpoint_name": checkpoint,
        "probability_threshold": 0.5,
        "minimum_component_volume_mm3": 0.0,
        "files": files,
    }


def main() -> int:
    manifest = synthetic_manifest()
    _, folds, expected_files = inference.validate_manifest_layout(manifest)
    assert folds == tuple(range(10))
    assert len(expected_files) == 12
    unsafe = json.loads(json.dumps(manifest))
    unsafe["model_directory"] = "../escape"
    try:
        inference.validate_manifest_layout(unsafe)
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe model directory was accepted")
    incomplete = json.loads(json.dumps(manifest))
    incomplete["files"].pop(next(iter(incomplete["files"])))
    try:
        inference.validate_manifest_layout(incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete ten-fold file layout was accepted")
    preliminary = synthetic_manifest()
    preliminary["bundle_purpose"] = "preliminary_interim"
    preliminary["folds"] = list(range(5))
    preliminary["checkpoint_name"] = "checkpoint_epoch_0500.pth"
    preliminary["files"] = {
        name.replace(
            "/checkpoint_final.pth", "/checkpoint_epoch_0500.pth"
        ): digest
        for name, digest in preliminary["files"].items()
        if "/fold_" not in name
        or any(f"/fold_{fold}/" in name for fold in range(5))
    }
    _, preliminary_folds, preliminary_files = (
        inference.validate_manifest_layout(preliminary)
    )
    assert preliminary_folds == tuple(range(5))
    assert len(preliminary_files) == 7
    unmarked_five_fold = json.loads(json.dumps(preliminary))
    unmarked_five_fold.pop("bundle_purpose")
    try:
        inference.validate_manifest_layout(unmarked_five_fold)
    except ValueError:
        pass
    else:
        raise AssertionError("unmarked five-fold bundle was accepted")
    wrong_preliminary_checkpoint = json.loads(json.dumps(preliminary))
    wrong_preliminary_checkpoint["checkpoint_name"] = "checkpoint_final.pth"
    wrong_preliminary_checkpoint["files"] = {
        name.replace(
            "/checkpoint_epoch_0500.pth", "/checkpoint_final.pth"
        ): digest
        for name, digest in wrong_preliminary_checkpoint["files"].items()
    }
    try:
        inference.validate_manifest_layout(wrong_preliminary_checkpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong preliminary checkpoint was accepted")
    remapped_preliminary = json.loads(json.dumps(preliminary))
    remapped_preliminary["source_fold_mapping"] = {"0": 4}
    try:
        inference.validate_manifest_layout(remapped_preliminary)
    except ValueError:
        pass
    else:
        raise AssertionError("remapped preliminary fold was accepted")

    reference = sitk.GetImageFromArray(np.zeros((9, 10, 11), np.float32))
    reference.SetSpacing((0.8, 1.1, 2.3))
    reference.SetOrigin((3.0, -2.0, 8.5))
    reference.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert inference.same_geometry(reference, reference)

    components = np.zeros((9, 10, 11), np.uint8)
    components[1, 1, 1] = 1
    components[5:7, 5:7, 5:7] = 1
    filtered = inference.remove_small_components(
        components,
        voxel_volume_mm3=1.0,
        minimum_volume_mm3=2.0,
    )
    assert int(filtered.sum()) == 8
    validator_filtered = validate_outputs.remove_small_components(
        components,
        voxel_volume_mm3=2.0,
        minimum_volume_mm3=5.0,
    )
    assert np.array_equal(filtered, validator_filtered)

    with tempfile.TemporaryDirectory() as tmp:
        old_output = inference.OUTPUT_PATH
        inference.OUTPUT_PATH = Path(tmp)
        try:
            output = inference.write_output(
                socket=inference.SEGMENTATION_SOCKET,
                array_zyx=filtered,
                reference=reference,
                pixel_id=sitk.sitkUInt8,
            )
            reread = sitk.ReadImage(str(output))
            assert inference.same_geometry(reference, reread)
            assert set(np.unique(sitk.GetArrayFromImage(reread))) <= {0, 1}
        finally:
            inference.OUTPUT_PATH = old_output

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        t1_dir = input_root / "images" / "t1-brain-mri"
        t1_dir.mkdir(parents=True)
        output_root.mkdir()
        sitk.WriteImage(reference, str(t1_dir / "input.mha"))
        (input_root / "inputs.json").write_text(
            json.dumps(
                [
                    {"socket": {"slug": "t1-brain-mri"}},
                    {"socket": {"slug": "stroke-metadata"}},
                ]
            ),
            encoding="utf-8",
        )
        (input_root / "stroke-metadata.json").write_text(
            json.dumps({"CENTER": "unit-test", "CHRONICITY": None}),
            encoding="utf-8",
        )
        old_input = inference.INPUT_PATH
        old_output = inference.OUTPUT_PATH
        inference.INPUT_PATH = input_root
        inference.OUTPUT_PATH = output_root
        try:
            inference.run(
                inference.ModelBundle(
                    predictor=FakePredictor(),
                    manifest={
                        "probability_threshold": 0.5,
                        "minimum_component_volume_mm3": 0.0,
                    },
                )
            )
            segmentation = sitk.ReadImage(
                str(
                    output_root
                    / "images"
                    / inference.SEGMENTATION_SOCKET
                    / "output.mha"
                )
            )
            probability = sitk.ReadImage(
                str(
                    output_root
                    / "images"
                    / inference.PROBABILITY_SOCKET
                    / "output.mha"
                )
            )
            assert inference.same_geometry(reference, segmentation)
            assert inference.same_geometry(reference, probability)
            assert int(sitk.GetArrayFromImage(segmentation).sum()) == 8
        finally:
            inference.INPUT_PATH = old_input
            inference.OUTPUT_PATH = old_output
    print("CONTAINER_CPU_CONTRACT_TESTS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

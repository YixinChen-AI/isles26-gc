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
    dual = synthetic_manifest()
    dual.pop("model_directory")
    dual.pop("folds")
    dual.pop("checkpoint_name")
    dual["ensemble_models"] = []
    dual["files"] = {}
    for name, directory, weight, digest in (
        ("baseline", "Baseline__Plans__3d_fullres", 0.25, "a"),
        ("tversky", "Tversky__Plans__3d_fullres", 0.75, "b"),
    ):
        dual["ensemble_models"].append(
            {
                "name": name,
                "model_directory": directory,
                "folds": list(range(10)),
                "checkpoint_name": "checkpoint_final.pth",
                "weight": weight,
            }
        )
        dual["files"][f"{directory}/dataset.json"] = digest * 64
        dual["files"][f"{directory}/plans.json"] = digest * 64
        for fold in range(10):
            dual["files"][
                f"{directory}/fold_{fold}/checkpoint_final.pth"
            ] = digest * 64
    _, dual_folds, dual_files = inference.validate_manifest_layout(dual)
    assert dual_folds == tuple(range(10))
    assert len(dual_files) == 24
    dual_components = inference.model_components(dual)
    assert [component.name for component in dual_components] == [
        "baseline", "tversky"
    ]
    assert [component.weight for component in dual_components] == [0.25, 0.75]
    blended = inference.blend_probability_maps(
        [
            (0.25, np.zeros((2, 3, 4), dtype=np.float32)),
            (0.75, np.ones((2, 3, 4), dtype=np.float32)),
        ]
    )
    np.testing.assert_allclose(blended, 0.75)
    tri = json.loads(json.dumps(dual))
    tri["ensemble_models"][0]["weight"] = 0.20
    tri["ensemble_models"][1]["weight"] = 0.50
    recall_directory = "RecallTversky__Plans__3d_fullres"
    tri["ensemble_models"].append(
        {
            "name": "recall_tversky",
            "model_directory": recall_directory,
            "folds": list(range(10)),
            "checkpoint_name": "checkpoint_final.pth",
            "weight": 0.30,
            "input_mode": "t1",
        }
    )
    tri["files"][f"{recall_directory}/dataset.json"] = "c" * 64
    tri["files"][f"{recall_directory}/plans.json"] = "c" * 64
    for fold in range(10):
        tri["files"][
            f"{recall_directory}/fold_{fold}/checkpoint_final.pth"
        ] = "c" * 64
    _, tri_folds, tri_files = inference.validate_manifest_layout(tri)
    assert tri_folds == tuple(range(10))
    assert len(tri_files) == 36
    tri_components = inference.model_components(tri)
    assert [component.name for component in tri_components] == [
        "baseline", "tversky", "recall_tversky"
    ]
    assert [component.weight for component in tri_components] == [
        0.20, 0.50, 0.30
    ]
    assert [component.input_mode for component in tri_components] == [
        "t1", "t1", "t1"
    ]
    symmetry_tri = json.loads(json.dumps(tri))
    symmetry_tri["ensemble_models"][-1]["name"] = "symmetry"
    symmetry_tri["ensemble_models"][-1][
        "input_mode"
    ] = "t1_physical_contralateral"
    inference.validate_manifest_layout(symmetry_tri)
    assert inference.model_components(symmetry_tri)[-1].input_mode == (
        "t1_physical_contralateral"
    )
    invalid_input_mode = json.loads(json.dumps(symmetry_tri))
    invalid_input_mode["ensemble_models"][-1]["input_mode"] = "array_flip"
    try:
        inference.validate_manifest_layout(invalid_input_mode)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported component input mode was accepted")
    tri_blended = inference.blend_probability_maps(
        [
            (0.20, np.zeros((2, 3, 4), dtype=np.float32)),
            (0.50, np.full((2, 3, 4), 0.5, dtype=np.float32)),
            (0.30, np.ones((2, 3, 4), dtype=np.float32)),
        ]
    )
    np.testing.assert_allclose(tri_blended, 0.55)
    logit_blended = inference.blend_probability_maps(
        [
            (0.5, np.asarray([0.2, 0.8], dtype=np.float32)),
            (0.5, np.asarray([0.8, 0.2], dtype=np.float32)),
        ],
        mode="weighted_logit",
    )
    np.testing.assert_allclose(logit_blended, 0.5, rtol=2e-5, atol=2e-6)
    invalid_dual_mode = json.loads(json.dumps(dual))
    invalid_dual_mode["ensemble_blend_mode"] = "unknown"
    try:
        inference.validate_manifest_layout(invalid_dual_mode)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported ensemble blend mode was accepted")
    invalid_dual_weight = json.loads(json.dumps(dual))
    invalid_dual_weight["ensemble_models"][0]["weight"] = 0.5
    try:
        inference.validate_manifest_layout(invalid_dual_weight)
    except ValueError:
        pass
    else:
        raise AssertionError("non-normalized ensemble weights were accepted")
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

    reflection_array = np.arange(1, 1 + 3 * 4 * 7, dtype=np.float32).reshape(
        3, 4, 7
    )
    reflection_reference = sitk.GetImageFromArray(reflection_array)
    reflection_reference.SetSpacing((0.7, 1.2, 2.1))
    mirrored = inference.physical_contralateral_image(reflection_reference)
    assert inference.same_geometry(reflection_reference, mirrored)
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(mirrored),
        np.flip(reflection_array, axis=2),
        rtol=0,
        atol=1e-5,
    )
    with tempfile.TemporaryDirectory() as input_tmp:
        input_tmp_path = Path(input_tmp)
        one = inference.stage_component_input(
            reflection_reference, input_tmp_path / "one", "t1"
        )
        two = inference.stage_component_input(
            reflection_reference,
            input_tmp_path / "two",
            "t1_physical_contralateral",
        )
        assert [path.name for path in one] == ["ISLES26GC_0000.nii.gz"]
        assert [path.name for path in two] == [
            "ISLES26GC_0000.nii.gz",
            "ISLES26GC_0001.nii.gz",
        ]
        written_mirror = sitk.ReadImage(str(two[1]))
        assert inference.same_geometry(reflection_reference, written_mirror)
        np.testing.assert_allclose(
            sitk.GetArrayFromImage(written_mirror),
            np.flip(reflection_array, axis=2),
            rtol=0,
            atol=1e-5,
        )

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

    guarded_probability = np.zeros((9, 10, 11), np.float32)
    guarded_probability[1:4, 1:4, 1:4] = 0.9
    guarded_probability[6, 6, 6] = 0.8
    guarded_probability[7, 1:3, 1:3] = 0.21
    guarded_manifest = {
        "probability_threshold": 0.2,
        "minimum_component_volume_mm3": 0.0,
        "postprocessing_policy": {
            "family": "relative_mean_volume_confidence_guard",
            "minimum_fraction_of_mean": 0.5,
            "minimum_mean_probability": 0.55,
        },
    }
    guarded = inference.apply_postprocessing(
        guarded_probability, guarded_manifest, voxel_volume_mm3=1.0
    )
    validator_guarded = validate_outputs.apply_postprocessing(
        guarded_probability, guarded_manifest, voxel_volume_mm3=1.0
    )
    assert int(guarded.sum()) == 28
    assert np.array_equal(guarded, validator_guarded)

    guarded_layout = synthetic_manifest()
    guarded_layout.update(guarded_manifest)
    inference.validate_manifest_layout(guarded_layout)
    invalid_guard = json.loads(json.dumps(guarded_layout))
    invalid_guard["postprocessing_policy"]["minimum_mean_probability"] = 0.1
    try:
        inference.validate_manifest_layout(invalid_guard)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid confidence guard was accepted")

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

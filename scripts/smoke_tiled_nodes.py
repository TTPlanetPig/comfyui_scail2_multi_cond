from __future__ import annotations

import inspect
from pathlib import Path
import sys
import types


def install_fake_torch() -> None:
    torch = types.ModuleType("torch")

    class Tensor:
        ndim = 4
        shape = ()

    torch.Tensor = Tensor
    sys.modules["torch"] = torch


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_raises(expected: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        text = str(exc)
        if expected not in text:
            raise AssertionError(f"expected error containing {expected!r}, got {text!r}") from exc
        return
    raise AssertionError(f"expected error containing {expected!r}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    install_fake_torch()
    import nodes

    class FakeVideo(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (4, 960, 548, 3)

    assert_true("SCAIL2TiledLongVideo" in nodes.NODE_CLASS_MAPPINGS, "missing tiled long video node")
    assert_true(
        "SCAIL2TiledLongVideoWithSAM" in nodes.NODE_CLASS_MAPPINGS,
        "missing tiled long video internal SAM node",
    )
    assert_true(
        nodes.NODE_DISPLAY_NAME_MAPPINGS["SCAIL2TiledLongVideoWithSAM"]
        == "SCAIL-2 Tiled Long Video (Internal SAM)",
        "unexpected display name",
    )
    assert_true(
        nodes.SCAIL2TiledLongVideo.RETURN_NAMES
        == ("frames", "actual_tile_manifest", "tile_repaint_report", "summary", "debug_preview"),
        "unexpected tiled node return names",
    )

    external_optional = nodes.SCAIL2TiledLongVideo.INPUT_TYPES()["optional"]
    internal_optional = nodes.SCAIL2TiledLongVideoWithSAM.INPUT_TYPES()["optional"]
    assert_true("reference_1_mask" in external_optional, "external tiled node should expose reference masks")
    assert_true("reference_1_mask" not in internal_optional, "internal SAM node should not expose reference masks")
    assert_true({"sam_model", "sam_conditioning"} <= set(internal_optional), "internal SAM inputs missing")

    assert_true(nodes._tile_seed(123, 1, "same_seed") == 123, "same_seed changed tile 1")
    assert_true(nodes._tile_seed(123, 7, "same_seed") == 123, "same_seed changed tile 7")
    assert_true(nodes._tile_seed(123, 7, "offset_by_tile") != 123, "offset_by_tile did not offset")

    manual_required_keys = list(nodes.SCAIL2ManualTilePlanBuilder.INPUT_TYPES()["required"])
    assert_true(
        manual_required_keys[-1] == "coverage_policy",
        "coverage_policy must stay last so older saved workflows do not shift max_tile_pixels into the combo widget",
    )
    assert_true(
        manual_required_keys.index("max_tile_pixels") < manual_required_keys.index("coverage_policy"),
        "manual tile widget order should keep max_tile_pixels before coverage_policy",
    )
    target_w, target_h, target_info = nodes._resolve_tile_target_size(548, 960, 1080, 1920, 2.0)
    assert_true([target_w, target_h] == [1096, 1920], "target resolution should preserve source aspect from requested height")
    assert_true(target_info["adjusted_output_size"], "target resolution adjustment should report changed user request")
    target_w, target_h, target_info = nodes._resolve_tile_target_size(548, 960, 1080, 0, 2.0)
    assert_true([target_w, target_h] == [1080, 1892], "target resolution should derive missing height from source aspect")
    assert_true(
        target_info["resolution_basis"] == "output_width_preserve_aspect",
        "single-width target should use width as aspect-preserving basis",
    )

    two_by_two = nodes._build_2x2_tile_manifest(
        FakeVideo(),
        1096,
        1920,
        0.10,
        32,
        48,
        [0, 274, 548],
        [0, 480, 960],
        mode="smoke_2x2_edge_overlap",
    )
    assert_true(
        two_by_two["tiles"][0]["overlap_edges_px_source"] == {"left": 0, "right": 28, "top": 0, "bottom": 48},
        "top-left 2x2 tile should only overlap right and bottom edges",
    )
    assert_true(
        two_by_two["tiles"][0]["source_crop_bbox"] == [0, 0, 302, 528],
        "top-left 2x2 crop should not expand outside the canvas",
    )
    assert_true(
        two_by_two["tiles"][3]["overlap_edges_px_source"] == {"left": 28, "right": 0, "top": 48, "bottom": 0},
        "bottom-right 2x2 tile should only overlap left and top edges",
    )
    assert_true(
        two_by_two["tiles"][3]["source_crop_bbox"] == [246, 432, 548, 960],
        "bottom-right 2x2 crop should not expand outside the canvas",
    )

    separated_manual = nodes._build_rect_tile_manifest(
        FakeVideo(),
        1096,
        1920,
        0.10,
        32,
        48,
        [[0, 0, 200, 960], [348, 0, 548, 960]],
        mode="smoke_non_adjacent_manual_tiles",
    )
    assert_true(
        separated_manual["tiles"][0]["source_crop_bbox"] == [0, 0, 200, 960],
        "manual tile with a gap to its neighbor should not expand across the gap",
    )
    assert_true(
        separated_manual["tiles"][0]["overlap_edges_px_source"]["right"] == 0,
        "manual tile should not mark a non-adjacent right edge as overlap",
    )

    touching_manual = nodes._build_rect_tile_manifest(
        FakeVideo(),
        1096,
        1920,
        0.10,
        32,
        48,
        [[0, 0, 274, 960], [274, 0, 548, 960]],
        mode="smoke_adjacent_manual_tiles",
        enforce_tile_pixel_limit=False,
    )
    assert_true(
        touching_manual["tiles"][0]["source_crop_bbox"] == [0, 0, 302, 960],
        "manual tile touching a neighbor should expand into the shared edge",
    )
    assert_true(
        touching_manual["tiles"][1]["source_crop_bbox"] == [246, 0, 548, 960],
        "right manual tile touching a neighbor should expand into the shared edge",
    )

    x_edges = [0, 78, 156, 234, 312, 390, 469, 548]
    core_bboxes = [[x_edges[index], 0, x_edges[index + 1], 960] for index in range(7)]
    manifest = nodes._build_rect_tile_manifest(
        FakeVideo(),
        1096,
        1920,
        0.10,
        32,
        48,
        core_bboxes,
        mode="smoke_7_tile_manifest",
        resolution_snap_mode="nearest",
    )
    assert_true(manifest["tile_count"] == 7, "7-tile manifest did not preserve tile_count")
    for tile in manifest["tiles"]:
        width, height = tile["tile_generate_size"]
        assert_true(width % 32 == 0 and height % 32 == 0, f"tile size not aligned: {width}x{height}")
        assert_true(width * height <= nodes.DEFAULT_MAX_TILE_PIXELS, f"tile over default budget: {width}x{height}")

    gappy_manual_layout = [[0, 0, 250, 960], [320, 0, 548, 960]]
    gappy_coverage = nodes._manual_tile_coverage_gaps(gappy_manual_layout, 548, 960)
    assert_true(gappy_coverage["gaps"] == [[250, 0, 320, 960]], "manual coverage gap detection changed")
    filled_layout, fill_info, auto_filled_from = nodes._apply_manual_tile_coverage_policy(
        gappy_manual_layout,
        548,
        960,
        0.20,
        "auto_fill",
    )
    filled_coverage = nodes._manual_tile_coverage_gaps(filled_layout, 548, 960)
    assert_true(auto_filled_from == 2, "auto_fill should append filler tiles after user tiles")
    assert_true(fill_info["auto_filled_tile_count"] == 1, "auto_fill did not report one filler tile")
    assert_true(filled_coverage["gaps"] == [], "auto_fill did not cover manual tile gap")
    assert_raises(
        "leaves uncovered source areas",
        lambda: nodes._apply_manual_tile_coverage_policy(gappy_manual_layout, 548, 960, 0.20, "error"),
    )
    ignored_layout, ignore_info, ignored_from = nodes._apply_manual_tile_coverage_policy(
        gappy_manual_layout,
        548,
        960,
        0.20,
        "ignore",
    )
    assert_true(ignored_layout == gappy_manual_layout, "ignore policy should preserve gappy manual layout")
    assert_true(ignored_from is None, "ignore policy should not mark auto-filled tiles")
    assert_true(ignore_info["uncovered_after"]["uncovered_gap_count"] == 1, "ignore policy should report remaining gap")

    warnings = nodes._validate_tiled_long_video_manifest(
        manifest,
        FakeVideo(),
        nodes.DEFAULT_MAX_TILE_PIXELS,
        True,
    )
    assert_true(warnings == [], "valid manifest produced warnings")
    assert_raises(
        "source_size must match pose_video size",
        lambda: nodes._validate_tiled_long_video_manifest(
            {**manifest, "source_size": [640, 960]},
            FakeVideo(),
            nodes.DEFAULT_MAX_TILE_PIXELS,
            True,
        ),
    )
    assert_raises(
        "over max_tile_pixels",
        lambda: nodes._validate_tiled_long_video_manifest(
            {
                "source_size": [548, 960],
                "tile_count": 1,
                "tiles": [{"tile_number": 1, "tile_generate_size": [1280, 736]}],
            },
            FakeVideo(),
            nodes.DEFAULT_MAX_TILE_PIXELS,
            True,
        ),
    )

    segments = [{"reference": 1}]
    assert_raises(
        "requires pose_video_mask",
        lambda: nodes._validate_tiled_reference_inputs(
            segments,
            {1: object()},
            {},
            1,
            "replacement",
            pose_video_mask=None,
            use_internal_sam=False,
        ),
    )
    assert_raises(
        "requires sam_model and sam_conditioning",
        lambda: nodes._validate_tiled_reference_inputs(
            segments,
            {1: object()},
            {},
            1,
            "replacement",
            pose_video_mask=None,
            use_internal_sam=True,
        ),
    )
    assert_true(
        nodes._validate_tiled_reference_inputs(
            segments,
            {1: object()},
            {},
            1,
            "animation",
            pose_video_mask=None,
            use_internal_sam=True,
        )
        == [1],
        "internal SAM animation validation failed",
    )

    orchestrator_source = inspect.getsource(nodes._run_tiled_long_video)
    assert_true(
        "SCAIL2ScheduledLongVideoWithSAM()" not in orchestrator_source,
        "tiled orchestrator should not run SAM once per tile",
    )
    assert_true(
        "global_once_then_tile_crop" in inspect.getsource(nodes._build_tiled_global_sam_masks),
        "global SAM crop strategy marker missing",
    )
    tile_weight_source = inspect.getsource(nodes._tile_weight_mask)
    assert_true("outer * 0.5" not in tile_weight_source, "tile composite should not give the whole overlap crop fallback weight")
    assert_true("expand_px=feather_px" not in tile_weight_source, "tile composite should not expand core across the whole overlap")
    assert_true("blur_px=feather_px" in tile_weight_source, "tile composite should feather the core edge")

    print("smoke_tiled_nodes: ok")


if __name__ == "__main__":
    main()

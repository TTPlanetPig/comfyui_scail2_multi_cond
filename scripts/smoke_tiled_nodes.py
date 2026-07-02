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

    class FakeSourceVideo(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (4, 1280, 704, 3)

    class FakePackedReference(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (1, 1920, 1056, 3)

    class FakeDoubleImage(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (1, 1920, 1096, 3)

    class FakeHalfMask(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (1, 480, 274, 3)

    assert_true("SCAIL2TiledLongVideo" in nodes.NODE_CLASS_MAPPINGS, "missing tiled long video node")
    assert_true(
        "SCAIL2TiledLongVideoWithSAM" in nodes.NODE_CLASS_MAPPINGS,
        "missing tiled long video internal SAM node",
    )
    assert_true("SCAIL2PlanReferencePackBuilder" in nodes.NODE_CLASS_MAPPINGS, "missing reference pack builder node")
    assert_true(
        nodes.NODE_DISPLAY_NAME_MAPPINGS["SCAIL2TiledLongVideoWithSAM"]
        == "SCAIL-2 Tiled Long Video (Internal SAM)",
        "unexpected display name",
    )
    assert_true(
        nodes.SCAIL2PlanReferencePackBuilder.RETURN_NAMES
        == ("reference_pack_images", "reference_pack_manifest", "debug_preview", "summary"),
        "unexpected reference pack return names",
    )
    assert_true(
        nodes.SCAIL2TiledLongVideo.RETURN_NAMES
        == ("frames", "actual_tile_manifest", "tile_repaint_report", "summary", "debug_preview"),
        "unexpected tiled node return names",
    )

    external_optional = nodes.SCAIL2TiledLongVideo.INPUT_TYPES()["optional"]
    internal_optional = nodes.SCAIL2TiledLongVideoWithSAM.INPUT_TYPES()["optional"]
    scheduled_optional = nodes.SCAIL2ScheduledLongVideo.INPUT_TYPES()["optional"]
    scheduled_sam_optional = nodes.SCAIL2ScheduledLongVideoWithSAM.INPUT_TYPES()["optional"]
    assert_true("reference_pack_images" not in scheduled_optional, "scheduled node should not expose tiled reference packs")
    assert_true("reference_pack_images" not in scheduled_sam_optional, "scheduled SAM node should not expose tiled reference packs")
    assert_true("reference_1_mask" in external_optional, "external tiled node should expose reference masks")
    assert_true("reference_1_mask" not in internal_optional, "internal SAM node should not expose reference masks")
    assert_true("reference_pack_images" in external_optional, "external tiled node should expose reference pack images")
    assert_true("reference_pack_manifest" in external_optional, "external tiled node should expose reference pack manifest")
    assert_true("reference_pack_images" in internal_optional, "internal SAM tiled node should expose reference pack images")
    assert_true("reference_pack_manifest" in internal_optional, "internal SAM tiled node should expose reference pack manifest")
    assert_true({"sam_model", "sam_conditioning"} <= set(internal_optional), "internal SAM inputs missing")
    pack_required = nodes.SCAIL2PlanReferencePackBuilder.INPUT_TYPES()["required"]
    pack_optional = nodes.SCAIL2PlanReferencePackBuilder.INPUT_TYPES()["optional"]
    assert_true(pack_required["resize_mode"][0][-1] == "upscale_model", "reference pack should support upscale_model mode")
    assert_true(pack_required["content_alignment_policy"][1]["default"] == "error", "reference pack should reject content shifts by default")
    assert_true(pack_required["max_content_shift_px"][1]["default"] == 1, "reference pack should allow at most one shifted pixel by default")
    assert_true(pack_required["content_alignment_device"][1]["default"] == "auto", "reference pack content alignment should default to auto device")
    assert_true(pack_required["pack_mode"][1]["default"] == "per_reference", "reference pack should preserve old per-reference behavior by default")
    assert_true("per_segment" in pack_required["pack_mode"][0], "reference pack should support one packed reference per segment")
    assert_true(list(pack_required)[-1] == "pack_mode", "pack_mode should append last to avoid shifting older reference-pack widgets")
    assert_true("upscale_model" in pack_optional, "reference pack should expose optional upscale_model")
    assert_true("tile_manifest" in pack_optional, "reference pack should accept tile_manifest for exact target size")
    packed_reference = FakePackedReference()

    class FakeNodeOutput:
        outputs = (packed_reference,)

    assert_true(
        nodes._extract_comfy_image_tensor_output((packed_reference,)) is packed_reference,
        "upscale wrapper should extract legacy tuple outputs",
    )
    assert_true(
        nodes._extract_comfy_image_tensor_output({"result": (packed_reference,)}) is packed_reference,
        "upscale wrapper should extract dict result outputs",
    )
    assert_true(
        nodes._extract_comfy_image_tensor_output(FakeNodeOutput()) is packed_reference,
        "upscale wrapper should extract V3 NodeOutput-style outputs",
    )
    composite_required = nodes.SCAIL2TileCompositeVideo.INPUT_TYPES()["required"]
    tiled_required = nodes.SCAIL2TiledLongVideo.INPUT_TYPES()["required"]
    tiled_sam_required = nodes.SCAIL2TiledLongVideoWithSAM.INPUT_TYPES()["required"]
    assert_true("blend_mode" in composite_required, "tile composite should expose blend_mode")
    assert_true("seam_alignment" in composite_required, "tile composite should expose seam_alignment")
    assert_true(
        composite_required["seam_alignment_apply_mode"][1]["default"] == "shifted_canvas_crop",
        "seam alignment should default to shifted-canvas crop",
    )
    assert_true(composite_required["seam_alignment_device"][1]["default"] == "auto", "seam alignment should default to auto device")
    assert_true(composite_required["max_seam_shift_px"][1]["default"] == 4, "unexpected max_seam_shift_px default")
    assert_true(composite_required["seam_alignment_frames"][1]["default"] == 9, "unexpected seam_alignment_frames default")
    assert_true(composite_required["junction_mode"][1]["default"] == "weighted_average", "junction_mode should preserve old behavior by default")
    assert_true("top2_normalized" in composite_required["junction_mode"][0], "tile composite should expose top2 junction mode")
    assert_true("composite_blend_mode" in tiled_required, "tiled long video should expose composite_blend_mode")
    assert_true("seam_alignment" in tiled_required, "tiled long video should expose seam_alignment")
    assert_true("seam_alignment_apply_mode" in tiled_required, "tiled long video should expose seam_alignment_apply_mode")
    assert_true("seam_alignment_device" in tiled_required, "tiled long video should expose seam_alignment_device")
    assert_true(tiled_required["junction_mode"][1]["default"] == "weighted_average", "tiled junction mode should default to old blending")
    assert_true("seam_alignment" in tiled_sam_required, "internal SAM tiled long video should expose seam_alignment")
    assert_true(
        "seam_alignment_apply_mode" in tiled_sam_required,
        "internal SAM tiled long video should expose seam_alignment_apply_mode",
    )
    assert_true(
        "seam_alignment_device" in tiled_sam_required,
        "internal SAM tiled long video should expose seam_alignment_device",
    )
    assert_true(tiled_sam_required["junction_mode"][1]["default"] == "weighted_average", "internal SAM junction mode should default to old blending")
    assert_true(
        list(composite_required)[-4:]
        == ["seam_alignment_device", "max_seam_shift_px", "seam_alignment_frames", "junction_mode"],
        "tile composite seam widgets should append last",
    )
    lookahead_keys = [
        "lookahead_reference",
        "lookahead_lead_frames",
        "lookahead_search_window_frames",
        "lookahead_analysis_stride",
        "lookahead_min_visible_ratio",
        "lookahead_min_new_area_ratio",
        "lookahead_max_anchors_per_tile",
    ]
    assert_true(
        list(tiled_required)[-8:] == ["junction_mode", *lookahead_keys],
        "lookahead widgets should append after junction_mode",
    )
    assert_true(
        list(tiled_sam_required)[-8:] == ["junction_mode", *lookahead_keys],
        "internal SAM lookahead widgets should append after junction_mode",
    )
    assert_true(
        list(tiled_required).index("composite_blend_mode") < list(tiled_required).index("free_tail_window"),
        "composite_blend_mode should stay before free_tail_window",
    )
    assert_true(
        list(tiled_required).index("seam_alignment") < list(tiled_required).index("free_tail_window"),
        "seam_alignment should stay before free_tail_window",
    )
    assert_true(
        list(tiled_sam_required).index("seam_alignment") < list(tiled_sam_required).index("free_tail_window"),
        "internal SAM seam_alignment should stay before free_tail_window",
    )
    free_tail_spec = tiled_required["free_tail_window"]
    assert_true(free_tail_spec[0] == "INT", "free_tail_window should be a numeric tail-frame control")
    assert_true(free_tail_spec[1]["default"] == 0, "free_tail_window default should disable free tail")
    assert_true(free_tail_spec[1]["step"] == 4, "free_tail_window should step by four frames")
    assert_true(tiled_required["lookahead_reference"][1]["default"] is False, "lookahead should default off")
    assert_true(tiled_sam_required["lookahead_reference"][1]["default"] is False, "internal SAM lookahead should default off")
    assert_true(tiled_required["lookahead_lead_frames"][1]["default"] == 8, "unexpected lookahead lead default")

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
    parsed_pack = nodes._parse_reference_pack_manifest(
        '{"source_size":[704,1280],"target_size":[1056,1920],"references":[{"reference":2,"batch_index":0}]}'
    )
    assert_true(parsed_pack["references"][0]["reference"] == 2, "reference pack should preserve plan reference id")
    parsed_segment_pack = nodes._parse_reference_pack_manifest(
        '{"pack_mode":"per_segment","source_size":[704,1280],"target_size":[1056,1920],"references":['
        '{"reference":1,"source_reference":1,"segment_index":0,"batch_index":0},'
        '{"reference":2,"source_reference":1,"segment_index":1,"batch_index":1},'
        '{"reference":3,"source_reference":2,"segment_index":2,"batch_index":2}]}'
    )
    assert_true(parsed_segment_pack["pack_mode"] == "per_segment", "reference pack should preserve per-segment mode")
    assert_true(parsed_segment_pack["references"][1]["source_reference"] == 1, "per-segment pack should keep the source reference id")
    assert_raises(
        "per_segment reference_pack_manifest entries must include segment_index",
        lambda: nodes._parse_reference_pack_manifest(
            '{"pack_mode":"per_segment","references":[{"reference":1,"source_reference":1,"batch_index":0}]}'
        ),
    )
    duplicate_source_segments = nodes._parse_plan(
        '[{"frames": 10, "reference": 1, "prompt": "a", "negative": ""},'
        '{"frames": 8, "reference": 1, "prompt": "b", "negative": ""},'
        '{"frames": 6, "reference": 2, "prompt": "c", "negative": ""}]',
        pose_frame_count=24,
        max_frames=0,
    )
    remapped_plan, remapped_segments, remap_info = nodes._resolve_reference_pack_segment_plan(
        "original plan text",
        duplicate_source_segments,
        {"enabled": True, "manifest": parsed_segment_pack},
    )
    assert_true(remap_info["enabled"], "per-segment pack should rewrite the generation plan")
    assert_true([int(segment["reference"]) for segment in remapped_segments] == [1, 2, 3], "per-segment pack should assign one packed ref per segment")
    assert_true("8 | 2 | b" in remapped_plan, "remapped plan should preserve segment prompts while changing references")
    assert_true(remap_info["remap"][1]["source_reference"] == 1, "remap should preserve original repeated source reference")
    lookahead_segments = nodes._parse_plan(
        '[{"frames": 10, "reference": 1, "prompt": "base", "negative": ""}]',
        pose_frame_count=10,
        max_frames=0,
    )
    tile_plan, tile_segments, tile_refs, tile_masks, tile_lookahead = nodes._apply_tile_lookahead_to_generation(
        1,
        lookahead_segments,
        "original",
        {1: FakePackedReference()},
        {1: FakePackedReference()},
        {
            "enabled": True,
            "anchors_by_tile": {
                1: [
                    {
                        "tile_number": 1,
                        "start_frame": 4,
                        "entry_frame": 6,
                        "reference_frame": 8,
                        "batch_index": 0,
                    }
                ]
            },
            "reference_images": [FakePackedReference()],
            "reference_masks": [FakePackedReference()],
        },
    )
    assert_true(tile_lookahead["applied"], "tile lookahead should apply anchor")
    assert_true(2 in tile_refs and 2 in tile_masks, "tile lookahead should allocate a new reference slot")
    assert_true([segment["frames"] for segment in tile_segments] == [4, 6], "tile lookahead should split segment at start_frame")
    assert_true([segment["reference"] for segment in tile_segments] == [1, 2], "tile lookahead should switch to anchor reference")
    assert_true("6 | 2 | base" in tile_plan, "tile lookahead plan should preserve prompt while switching reference")
    five_segment_plan = (
        '[{"frames": 10, "reference": 1, "prompt": "a", "negative": ""},'
        '{"frames": 10, "reference": 2, "prompt": "b", "negative": ""},'
        '{"frames": 10, "reference": 3, "prompt": "c", "negative": ""},'
        '{"frames": 10, "reference": 4, "prompt": "d", "negative": ""},'
        '{"frames": 10, "reference": 5, "prompt": "e", "negative": ""}]'
    )
    four_active_segments = nodes._parse_plan(five_segment_plan, pose_frame_count=40, max_frames=0)
    clip_report = nodes._segment_plan_clip_report(
        five_segment_plan,
        four_active_segments,
        pose_frame_count=40,
        max_frames=0,
    )
    assert_true(clip_report["raw_segment_count"] == 5, "clip report should preserve raw segment count")
    assert_true(clip_report["active_segment_count"] == 4, "clip report should show active segment count after clipping")
    assert_true(clip_report["clipped_segment_indices"] == [4], "clip report should identify the clipped fifth segment")
    assert_true(clip_report["clip_cap_frames"] == 40, "clip report should expose the frame cap that caused clipping")
    assert_raises(
        "source_reference does not match",
        lambda: nodes._resolve_reference_pack_segment_plan(
            "original plan text",
            duplicate_source_segments,
            {
                "enabled": True,
                "manifest": {
                    **parsed_segment_pack,
                    "references": [
                        parsed_segment_pack["references"][0],
                        {**parsed_segment_pack["references"][1], "source_reference": 2},
                        parsed_segment_pack["references"][2],
                    ],
                },
            },
        ),
    )
    geometry_manifest = {
        "source_size": [704, 1280],
        "target_size": [1056, 1920],
        "tiles": [
            {
                "tile_number": 1,
                "source_crop_bbox": [0, 0, 352, 640],
                "target_crop_bbox": [0, 0, 528, 960],
            },
            {
                "tile_number": 2,
                "source_crop_bbox": [352, 640, 704, 1280],
                "target_crop_bbox": [528, 960, 1056, 1920],
            },
        ],
    }
    pack_info = {
        "enabled": True,
        "source_size": [704, 1280],
        "target_size": [1056, 1920],
        "references": [{"reference": 1, "batch_index": 0}],
    }
    geometry = nodes._validate_reference_pack_geometry(
        pack_info,
        geometry_manifest,
        {1: FakePackedReference()},
        FakeSourceVideo(),
    )
    assert_true(geometry["reference_count"] == 1, "reference pack geometry should count checked references")
    assert_true(len(geometry["checked_tile_crops"]) == 2, "reference pack geometry should check every tile")
    assert_true(
        geometry["content_alignment_checks"][0]["status"] == "skipped_missing_source_frame",
        "reference pack geometry should skip content checks when source_frame is unavailable",
    )
    bad_manifest = {
        **geometry_manifest,
        "tiles": [{**geometry_manifest["tiles"][0], "target_crop_bbox": [1, 0, 528, 960]}],
    }
    assert_raises(
        "reference pack pixel geometry mismatch",
        lambda: nodes._validate_reference_pack_geometry(pack_info, bad_manifest, {1: FakePackedReference()}, FakeSourceVideo()),
    )
    clipped_segments = nodes._parse_plan(
        '[{"frames": 100, "reference": 1, "prompt": "test", "negative": ""}]',
        pose_frame_count=49,
        max_frames=0,
    )
    assert_true(sum(int(segment["frames"]) for segment in clipped_segments) == 49, "plan should clip to video length")
    clipped_chunks = nodes._build_chunk_plan(clipped_segments, 81, 5)
    assert_true(len(clipped_chunks) == 1, "49 clipped frames should fit in one base chunk")
    assert_true(nodes._normalize_free_tail_window_frames(True) == 4, "legacy true should map to 4 free-tail frames")
    assert_true(nodes._normalize_free_tail_window_frames(False) == 0, "legacy false should disable free tail")
    assert_true(nodes._normalize_free_tail_window_frames(6) == 8, "free-tail frames should snap up to a multiple of four")
    assert_true(
        nodes._free_tail_window_target_length(49, 81, 49, 4) == 53,
        "49-frame free tail should add one latent step, not fill the whole 81-frame window",
    )
    assert_true(
        nodes._free_tail_window_target_length(49, 81, 49, 8) == 57,
        "49-frame free tail should support two latent tail steps",
    )
    assert_true(
        nodes._free_tail_window_target_length(49, 81, 49, 0) is None,
        "zero free-tail frames should disable the final tail window",
    )
    assert_true(
        nodes._free_tail_window_target_length(81, 81, 81, 4) is None,
        "full 81-frame windows should report no room for a free tail",
    )
    source_region = [246, 432, 548, 960]
    assert_true(
        nodes._tile_tensor_crop_bbox(source_region, [548, 960], FakeVideo()) == source_region,
        "same-size tile input should use the manifest source bbox exactly",
    )
    assert_true(
        nodes._tile_tensor_crop_bbox(source_region, [548, 960], FakeDoubleImage()) == [492, 864, 1096, 1920],
        "double-size tile input should receive the same source region scaled to its own pixels",
    )
    assert_true(
        nodes._tile_tensor_crop_bbox(source_region, [548, 960], FakeHalfMask()) == [123, 216, 274, 480],
        "half-size tile mask should receive the same source region scaled to its own pixels",
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
    assert_true("core_feather" in tile_weight_source, "tile composite should keep the original core feather mode")
    assert_true("ttp_seam" in tile_weight_source, "tile composite should expose the TTP-style seam feather mode")
    composite_source = inspect.getsource(nodes.SCAIL2TileCompositeVideo.composite)
    assert_true("_estimate_tile_seam_offsets" in composite_source, "tile composite should run temporal seam alignment")
    assert_true("_shift_image_batch_integer" in composite_source, "tile composite should apply seam offsets before paste")
    assert_true("_covered_viewport_crop_bbox" in composite_source, "shifted seam alignment should crop the covered viewport")
    assert_true("shifted_canvas_crop" in composite_source, "tile composite should support shifted-canvas crop mode")
    assert_true("mask.repeat(frame_count" not in composite_source, "tile composite should not duplicate seam masks for every frame")
    assert_true("top2_normalized" in composite_source, "tile composite should support top2 normalized junction blending")
    assert_true("top_index_1" in composite_source and "top_index_2" in composite_source, "top2 mode should track the two strongest tile weights")
    crop_source = inspect.getsource(nodes._covered_viewport_crop_bbox)
    assert_true("largest_fully_covered_rectangle" in crop_source, "shifted canvas crop should remove uncovered black borders")
    assert_true("cropped_uncovered_pixels" in crop_source, "shifted canvas crop should report remaining uncovered pixels")
    estimate_source = inspect.getsource(nodes._estimate_overlap_shift)
    score_source = inspect.getsource(nodes._score_overlap_shift_samples)
    assert_true("_resolve_seam_alignment_device" in estimate_source, "seam alignment should choose CPU/GPU device")
    assert_true("torch.stack(score_tensors)" in score_source, "seam alignment should avoid per-candidate CPU sync")
    orchestrator_source = inspect.getsource(nodes._run_tiled_long_video)
    assert_true("seam_alignment" in orchestrator_source, "tiled orchestrator should pass seam alignment options")
    assert_true("seam_alignment_apply_mode" in orchestrator_source, "tiled orchestrator should pass seam apply mode")
    assert_true("seam_alignment_device" in orchestrator_source, "tiled orchestrator should pass seam device")
    assert_true("junction_mode" in orchestrator_source, "tiled orchestrator should pass the junction blending mode")
    assert_true("_resolve_reference_pack_segment_plan" in orchestrator_source, "tiled orchestrator should remap per-segment reference packs")
    assert_true("generation_segment_plan" in orchestrator_source, "tiled orchestrator should pass the remapped generation plan")
    assert_true("_validate_reference_pack_geometry" in orchestrator_source, "tiled orchestrator should verify reference pack pixel geometry")
    assert_true("effective_reference_count" in orchestrator_source, "tiled orchestrator should expand reference_count for reference packs")
    content_source = inspect.getsource(nodes._check_reference_content_alignment)
    assert_true("_estimate_overlap_shift" in content_source, "reference pack should estimate content registration, not only canvas size")
    assert_true("max_content_shift_px" in content_source, "reference pack content shift should be thresholded")

    print("smoke_tiled_nodes: ok")


if __name__ == "__main__":
    main()

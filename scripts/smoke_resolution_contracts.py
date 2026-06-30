from __future__ import annotations

import json
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


def assert_scail_size(size: list[int], label: str) -> None:
    width, height = [int(value) for value in size]
    assert_true(width % 32 == 0 and height % 32 == 0, f"{label} must be 32-aligned, got {width}x{height}")


def assert_exact_aspect(size: tuple[int, int], target: tuple[int, int], label: str) -> None:
    width, height = [int(value) for value in size]
    target_w, target_h = [int(value) for value in target]
    assert_true(width * target_h == height * target_w, f"{label} aspect mismatch: {width}x{height} vs {target_w}x{target_h}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    install_fake_torch()
    import nodes

    class FakeFaceCrop(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (49, 708, 1280, 3)

    class FakeTileSource(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (8, 960, 548, 3)

    assert_true(nodes.SCAIL_RESOLUTION_ALIGN == 32, "SCAIL resolution alignment should be 32")
    assert_true(nodes._normalize_scail_resolution_align(1) == 32, "legacy tiny align should normalize to 32")
    assert_true(nodes._normalize_scail_resolution_align(48) == 64, "non-32 align should normalize upward")
    assert_true(nodes._infer_generation_size(FakeFaceCrop()) == (1280, 704), "708px pose height should floor to 704 for SCAIL")

    matched_w, matched_h, aspect_info = nodes._match_size_to_aspect_grid(1001, 550, 1280, 704)
    assert_exact_aspect((matched_w, matched_h), (1280, 704), "aligned reference crop")
    assert_true(aspect_info["aspect_exact"], "aspect matcher should report exact target aspect")

    head_required = nodes.SCAIL2HeadTrackCrop.INPUT_TYPES()["required"]
    assert_true(head_required["square_align"][1]["min"] == 32, "head crop square_align UI should start at 32")
    assert_true(head_required["square_align"][1]["step"] == 32, "head crop square_align UI should step by 32")

    automatic_required = nodes.SCAIL2TilePlanBuilder.INPUT_TYPES()["required"]
    manual_required = nodes.SCAIL2ManualTilePlanBuilder.INPUT_TYPES()["required"]
    for label, required in [("auto tile", automatic_required), ("manual tile", manual_required)]:
        assert_true(required["tile_align"][1]["min"] == 32, f"{label} tile_align UI should start at 32")
        assert_true(required["tile_align"][1]["step"] == 32, f"{label} tile_align UI should step by 32")

    auto_manifest = nodes._build_2x2_tile_manifest(
        FakeTileSource(),
        1096,
        1920,
        0.10,
        8,
        48,
        [0, 274, 548],
        [0, 480, 960],
        mode="smoke_scail_aligned_auto_tile",
        enforce_tile_pixel_limit=False,
    )
    assert_true(auto_manifest["tile_align"] == 32, "auto tile manifest should normalize tile_align to 32")
    for tile in auto_manifest["tiles"]:
        assert_scail_size(tile["tile_generate_size"], f"auto tile {tile['tile_number']}")

    manual_manifest = nodes._build_rect_tile_manifest(
        FakeTileSource(),
        1096,
        1920,
        0.10,
        16,
        48,
        [[0, 0, 274, 960], [274, 0, 548, 960]],
        mode="smoke_scail_aligned_manual_tile",
        enforce_tile_pixel_limit=False,
    )
    assert_true(manual_manifest["tile_align"] == 32, "manual tile manifest should normalize tile_align to 32")
    for tile in manual_manifest["tiles"]:
        assert_scail_size(tile["tile_generate_size"], f"manual tile {tile['tile_number']}")

    bad_manifest = {
        "version": 1,
        "source_size": [548, 960],
        "tile_count": 1,
        "tiles": [
            {
                "index": 0,
                "tile_number": 1,
                "source_crop_bbox": [0, 0, 548, 960],
                "target_crop_bbox": [0, 0, 1280, 708],
                "tile_generate_size": [1280, 708],
            }
        ],
    }
    assert_raises(
        "divisible by 32",
        lambda: nodes._validate_tiled_long_video_manifest(
            bad_manifest,
            FakeTileSource(),
            0,
            False,
        ),
    )

    print("smoke_resolution_contracts: ok")


if __name__ == "__main__":
    main()

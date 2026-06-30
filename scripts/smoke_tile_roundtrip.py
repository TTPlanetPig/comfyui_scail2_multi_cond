from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np


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


def synthetic_canvas(height: int, width: int) -> np.ndarray:
    y = np.arange(height, dtype=np.float32).reshape(height, 1)
    x = np.arange(width, dtype=np.float32).reshape(1, width)
    return np.stack(
        [
            (x % 251) / 250.0 + np.zeros((height, width), dtype=np.float32),
            (y % 239) / 238.0 + np.zeros((height, width), dtype=np.float32),
            ((x + y) % 223) / 222.0,
        ],
        axis=-1,
    ).astype(np.float32)


def linear_core_feather_mask(height: int, width: int, crop_bbox: list[int], core_bbox: list[int], feather_px: int) -> np.ndarray:
    x0, y0, _x1, _y1 = [int(value) for value in crop_bbox]
    cx0, cy0, cx1, cy1 = [int(value) for value in core_bbox]
    lx0 = max(0, min(width, cx0 - x0))
    ly0 = max(0, min(height, cy0 - y0))
    lx1 = max(0, min(width, cx1 - x0))
    ly1 = max(0, min(height, cy1 - y0))
    feather = max(0, int(feather_px))
    y = np.arange(height, dtype=np.float32).reshape(height, 1) + 0.5
    x = np.arange(width, dtype=np.float32).reshape(1, width) + 0.5
    dx = np.maximum(np.maximum(lx0 - x, x - lx1), 0.0)
    dy = np.maximum(np.maximum(ly0 - y, y - ly1), 0.0)
    distance = np.maximum(dx, dy)
    mask = np.zeros((height, width), dtype=np.float32)
    inside = (x >= lx0) & (x <= lx1) & (y >= ly0) & (y <= ly1)
    mask[inside] = 1.0
    if feather > 0:
        mask = np.maximum(mask, np.clip(1.0 - distance / float(feather), 0.0, 1.0))
    return mask


def ttp_seam_mask(height: int, width: int, crop_bbox: list[int], core_bbox: list[int], feather_px: int) -> np.ndarray:
    x0, y0, _x1, _y1 = [int(value) for value in crop_bbox]
    cx0, cy0, cx1, cy1 = [int(value) for value in core_bbox]
    lx0 = max(0, min(width, cx0 - x0))
    ly0 = max(0, min(height, cy0 - y0))
    lx1 = max(0, min(width, cx1 - x0))
    ly1 = max(0, min(height, cy1 - y0))
    feather = max(0, int(feather_px))
    if feather <= 0:
        mask = np.zeros((height, width), dtype=np.float32)
        mask[ly0:ly1, lx0:lx1] = 1.0
        return mask

    def axis_weights(size: int, start: int, end: int) -> np.ndarray:
        coords = np.arange(size, dtype=np.float32) + 0.5
        weights = np.ones((size,), dtype=np.float32)
        left_context = max(0, int(start))
        if left_context > 0:
            blend = max(1.0, float(min(feather, left_context * 2)))
            ramp_start = float(start) - blend / 2.0
            ramp_end = float(start) + blend / 2.0
            weights = np.minimum(weights, np.clip((coords - ramp_start) / max(1e-6, ramp_end - ramp_start), 0.0, 1.0))
        right_context = max(0, int(size) - int(end))
        if right_context > 0:
            blend = max(1.0, float(min(feather, right_context * 2)))
            ramp_start = float(end) - blend / 2.0
            ramp_end = float(end) + blend / 2.0
            weights = np.minimum(weights, np.clip(1.0 - ((coords - ramp_start) / max(1e-6, ramp_end - ramp_start)), 0.0, 1.0))
        return weights

    return axis_weights(height, ly0, ly1).reshape(height, 1) * axis_weights(width, lx0, lx1).reshape(1, width)


def composite_from_target_crops(target: np.ndarray, manifest: dict, feather_px: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    height, width, _channels = target.shape
    output = np.zeros_like(target)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    contribution_count = np.zeros((height, width), dtype=np.int32)
    for tile in manifest["tiles"]:
        x0, y0, x1, y1 = [int(value) for value in tile["target_crop_bbox"]]
        crop = target[y0:y1, x0:x1, :]
        crop_h, crop_w = crop.shape[:2]
        if mode == "ttp_seam":
            mask = ttp_seam_mask(crop_h, crop_w, [x0, y0, x1, y1], tile["target_core_bbox"], feather_px)
        else:
            mask = linear_core_feather_mask(crop_h, crop_w, [x0, y0, x1, y1], tile["target_core_bbox"], feather_px)
        output[y0:y1, x0:x1, :] += crop * mask[:, :, None]
        weights[y0:y1, x0:x1, :] += mask[:, :, None]
        contribution_count[y0:y1, x0:x1] += (mask > 0.05).astype(np.int32)
    return output / np.maximum(weights, 1e-6), contribution_count


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    install_fake_torch()
    import nodes

    class FakeVideo(sys.modules["torch"].Tensor):
        ndim = 4
        shape = (4, 960, 548, 3)

    manifest = nodes._build_2x2_tile_manifest(
        FakeVideo(),
        1096,
        1920,
        0.10,
        32,
        48,
        [0, 274, 548],
        [0, 480, 960],
        mode="roundtrip_2x2",
    )
    target_w, target_h = manifest["target_size"]
    target = synthetic_canvas(int(target_h), int(target_w))
    metrics = {}
    for mode in ("core_feather", "ttp_seam"):
        reconstructed, contribution_count = composite_from_target_crops(target, manifest, 48, mode)
        max_abs = float(np.max(np.abs(reconstructed - target)))
        ambiguous_pixels = int(np.count_nonzero(contribution_count > 1))
        metrics[mode] = {
            "max_abs": max_abs,
            "ambiguous_pixels": ambiguous_pixels,
            "ambiguous_ratio": float(ambiguous_pixels / max(1, target_w * target_h)),
        }
        assert_true(max_abs <= 1e-6, f"{mode} roundtrip introduced coordinate error: {max_abs}")

    assert_true(
        metrics["ttp_seam"]["ambiguous_pixels"] < metrics["core_feather"]["ambiguous_pixels"],
        "ttp_seam should reduce the area where multiple generated tiles are blended",
    )
    reduction = 1.0 - (metrics["ttp_seam"]["ambiguous_pixels"] / max(1, metrics["core_feather"]["ambiguous_pixels"]))
    print(
        "smoke_tile_roundtrip: ok "
        f"core_feather ambiguous={metrics['core_feather']['ambiguous_pixels']} "
        f"ttp_seam ambiguous={metrics['ttp_seam']['ambiguous_pixels']} "
        f"reduction={reduction:.3f}"
    )


if __name__ == "__main__":
    main()

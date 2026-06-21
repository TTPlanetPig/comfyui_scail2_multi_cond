from __future__ import annotations

import gc
import hashlib
import inspect
import json
from typing import Any, Optional

import torch


CATEGORY = "SCAIL-2/Scheduled"
MAX_REFERENCES = 8
DEFAULT_PLAN = """# frames | reference | prompt | negative | boundary_overlap
49 | 1 | first segment prompt | |
121 | 2 | second segment prompt | | 1
73 | 3 | third segment prompt | | 1
157 | 4 | fourth segment prompt | | 1
"""


def _shape(value: Any) -> list[int]:
    return list(value.shape) if hasattr(value, "shape") else []


def _stable_fingerprint(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tensor_fingerprint(value: Optional[torch.Tensor]) -> Any:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__}
    data = value.detach()
    marker: dict[str, Any] = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
    }
    if data.numel() == 0:
        marker["sample"] = ""
        return marker
    flat = data.reshape(-1)
    step = max(1, int(flat.numel()) // 4096)
    sample = flat[::step][:4096].detach().cpu().float().contiguous()
    marker["sample"] = hashlib.sha256(sample.numpy().tobytes()).hexdigest()
    return marker


def _cache_marker(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_fingerprint(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _cache_marker(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_cache_marker(item) for item in value]
    return {"type": type(value).__name__, "id": id(value)}


def _clone_cached_result(result: tuple) -> tuple:
    return tuple(result)


def _empty_cache(force: bool = False) -> None:
    if force:
        gc.collect()
    try:
        import comfy.model_management

        comfy.model_management.cleanup_models_gc()
        try:
            comfy.model_management.soft_empty_cache(force=force)
        except TypeError:
            comfy.model_management.soft_empty_cache()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _node_result(value: Any) -> tuple:
    if hasattr(value, "result"):
        result = value.result
        if result is None:
            return ()
        return tuple(result)
    if isinstance(value, tuple):
        return value
    return (value,)


def _get_scail_nodes_module():
    try:
        from comfy_extras import nodes_scail

        return nodes_scail
    except ImportError:
        from comfy_extras import nodes_wan

        return nodes_wan


def _create_scail_masks(
    driving_track_data,
    reference_track_data,
    object_indices: str,
    sort_by: str,
    replacement_mode: bool,
):
    scail_nodes = _get_scail_nodes_module()
    SCAIL2ColoredMask = getattr(scail_nodes, "SCAIL2ColoredMask", None)
    if SCAIL2ColoredMask is None:
        raise RuntimeError("SCAIL2ColoredMask is unavailable in this ComfyUI build.")
    result = _node_result(
        SCAIL2ColoredMask.execute(
            driving_track_data,
            object_indices,
            sort_by,
            bool(replacement_mode),
            ref_track_data=reference_track_data,
        )
    )
    if len(result) != 2:
        raise RuntimeError("SCAIL2ColoredMask returned an unexpected result.")
    return result[0], result[1]


def _ceil_to_4n_plus_1(value: int) -> int:
    value = max(1, int(value))
    return 1 + ((value - 1 + 3) // 4) * 4


def _floor_to_4n_plus_1(value: int) -> int:
    value = max(1, int(value))
    return 1 + ((value - 1) // 4) * 4


def _normalize_overlap(value: int, max_chunk_frames: int) -> int:
    requested = max(0, int(value))
    if requested <= 0:
        return 0
    return min(_floor_to_4n_plus_1(requested), 33, int(max_chunk_frames) - 4)


def _parse_plan_input(segment_plan: str) -> list[dict[str, Any]]:
    text = str(segment_plan or "").strip()
    if not text:
        raise ValueError("segment_plan must not be empty.")

    if text.startswith("["):
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"segment_plan JSON is invalid: {exc}") from exc
        if not isinstance(raw, list) or not raw:
            raise ValueError("segment_plan JSON must be a non-empty array.")
        return raw

    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if parts and parts[0].lower() in {"frames", "frame", "帧数"}:
            continue
        if len(parts) < 3:
            raise ValueError(
                f"segment_plan line {line_number} must use: frames | reference | prompt | negative | boundary_overlap"
            )
        row: dict[str, Any] = {
            "frames": parts[0],
            "reference": parts[1],
            "prompt": parts[2],
            "negative": parts[3] if len(parts) > 3 else "",
        }
        if len(parts) > 4 and parts[4]:
            row["boundary_overlap"] = parts[4]
        rows.append(row)

    if not rows:
        raise ValueError("segment_plan text produced no segment rows.")
    return rows


def _parse_plan(segment_plan: str, pose_frame_count: Optional[int] = None, max_frames: int = 0) -> list[dict[str, Any]]:
    try:
        raw = _parse_plan_input(segment_plan)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"segment_plan could not be parsed: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("segment_plan must be a non-empty list.")

    segments: list[dict[str, Any]] = []
    cursor = 0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"segment_plan[{index}] must be an object.")
        frames = int(item.get("frames", 0))
        if frames <= 0:
            raise ValueError(f"segment_plan[{index}].frames must be greater than 0.")
        reference = int(item.get("reference", 1))
        if reference < 1 or reference > MAX_REFERENCES:
            raise ValueError(f"segment_plan[{index}].reference must be between 1 and {MAX_REFERENCES}.")
        prompt = str(item.get("prompt", ""))
        negative = str(item.get("negative", ""))
        boundary_overlap = item.get("boundary_overlap", None)
        if boundary_overlap is not None:
            boundary_overlap = int(boundary_overlap)
            if boundary_overlap < 0:
                raise ValueError(f"segment_plan[{index}].boundary_overlap must be greater than or equal to 0.")
        segments.append(
            {
                "index": index,
                "start": cursor,
                "end": cursor + frames,
                "frames": frames,
                "reference": reference,
                "boundary_overlap": boundary_overlap,
                "prompt": prompt,
                "negative": negative,
            }
        )
        cursor += frames

    total = cursor
    cap = int(max_frames) if max_frames and int(max_frames) > 0 else None
    if pose_frame_count is not None:
        cap = min(cap, pose_frame_count) if cap is not None else pose_frame_count
    if cap is not None and total > cap:
        clipped: list[dict[str, Any]] = []
        for segment in segments:
            if segment["start"] >= cap:
                break
            clipped_segment = dict(segment)
            clipped_segment["end"] = min(segment["end"], cap)
            clipped_segment["frames"] = clipped_segment["end"] - clipped_segment["start"]
            if clipped_segment["frames"] > 0:
                clipped.append(clipped_segment)
        segments = clipped

    if not segments:
        raise ValueError("segment_plan produces no frames after max_frames/video length clipping.")
    return segments


def _clean_plan_cell(value: Any) -> str:
    text = str(value or "").replace("|", "/")
    return " ".join(part.strip() for part in text.splitlines() if part.strip())


def _format_plan_rows(rows: list[dict[str, Any]]) -> str:
    lines = ["# frames | reference | prompt | negative | boundary_overlap"]
    for row in rows:
        boundary = row.get("boundary_overlap", None)
        boundary_text = "" if boundary is None or int(boundary) < 0 else str(int(boundary))
        lines.append(
            " | ".join(
                [
                    str(int(row["frames"])),
                    str(int(row["reference"])),
                    _clean_plan_cell(row.get("prompt", "")),
                    _clean_plan_cell(row.get("negative", "")),
                    boundary_text,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _build_chunk_plan(segments: list[dict[str, Any]], max_chunk_frames: int, overlap_frames: int) -> list[dict[str, Any]]:
    max_chunk_frames = min(81, _ceil_to_4n_plus_1(max(17, int(max_chunk_frames))))
    overlap = _normalize_overlap(overlap_frames, max_chunk_frames)
    produced = 0
    has_previous = False
    last_segment_reference = None
    chunk_index = 0
    chunks: list[dict[str, Any]] = []
    for segment in segments:
        remaining = int(segment["frames"])
        segment_kept = 0
        is_reference_change_segment = (
            last_segment_reference is not None and int(segment["reference"]) != int(last_segment_reference)
        )
        while remaining > 0:
            boundary_override = (
                segment.get("boundary_overlap")
                if is_reference_change_segment and segment_kept == 0
                else None
            )
            effective_overlap = (
                _normalize_overlap(boundary_override, max_chunk_frames)
                if boundary_override is not None
                else overlap
            )
            use_previous = has_previous and effective_overlap > 0
            max_keep = max_chunk_frames if not use_previous else max_chunk_frames - effective_overlap
            wanted_keep = min(remaining, max_keep)
            if wanted_keep <= 0:
                raise RuntimeError("Internal planner produced an empty chunk.")
            raw_length = wanted_keep if not use_previous else wanted_keep + effective_overlap
            length = _ceil_to_4n_plus_1(raw_length)
            if length < 17 and remaining > 1:
                length = 17
            if length > max_chunk_frames:
                length = max_chunk_frames
                wanted_keep = length if not use_previous else length - effective_overlap
            if wanted_keep <= 0:
                raise RuntimeError("overlap_frames leaves no room for new frames.")
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "segment_index": int(segment["index"]),
                    "segment_start": int(segment["start"]),
                    "segment_end": int(segment["end"]),
                    "reference": int(segment["reference"]),
                    "boundary_overlap": boundary_override,
                    "prompt": segment["prompt"],
                    "negative": segment["negative"],
                    "generate_length": int(length),
                    "discard_head": int(effective_overlap if use_previous else 0),
                    "keep_frames": int(wanted_keep),
                    "output_start": int(produced),
                    "output_end": int(produced + wanted_keep),
                }
            )
            produced += int(wanted_keep)
            remaining -= int(wanted_keep)
            segment_kept += int(wanted_keep)
            has_previous = True
            chunk_index += 1
        last_segment_reference = int(segment["reference"])
    return chunks


def _infer_generation_size(pose_video: torch.Tensor) -> tuple[int, int]:
    if not isinstance(pose_video, torch.Tensor) or pose_video.ndim != 4:
        raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
    height = max(32, (int(pose_video.shape[1]) // 32) * 32)
    width = max(32, (int(pose_video.shape[2]) // 32) * 32)
    return width, height


def _first_image(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    if value.shape[0] <= 0:
        raise ValueError(f"{name} has no images.")
    return value[:1].detach().contiguous()


def _encode_text(clip, text: str):
    import nodes

    return nodes.CLIPTextEncode().encode(clip, text)[0]


def _encode_clip_vision(clip_vision, image: torch.Tensor):
    import nodes

    return nodes.CLIPVisionEncode().encode(clip_vision, image, "none")[0]


def _run_sam3_track(
    images: torch.Tensor,
    model,
    conditioning,
    detection_threshold: float,
    max_objects: int,
    detect_interval: int,
):
    from comfy_extras.nodes_sam3 import SAM3_VideoTrack

    result = _node_result(
        SAM3_VideoTrack.execute(
            images,
            model,
            initial_mask=None,
            conditioning=conditioning,
            detection_threshold=float(detection_threshold),
            max_objects=int(max_objects),
            detect_interval=max(1, int(detect_interval)),
        )
    )
    if len(result) != 1:
        raise RuntimeError("SAM3_VideoTrack returned an unexpected result.")
    return result[0]


def _apply_reference_mask(reference_image: torch.Tensor, reference_image_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if reference_image_mask is None:
        return reference_image
    if reference_image_mask.shape[1:3] != reference_image.shape[1:3]:
        import torch.nn.functional as F

        resized = F.interpolate(
            reference_image_mask[:1].detach().float().movedim(-1, 1),
            size=(int(reference_image.shape[1]), int(reference_image.shape[2])),
            mode="nearest",
        ).movedim(1, -1)
    else:
        resized = reference_image_mask[:1].detach()
    alpha = (resized[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(dtype=reference_image.dtype)
    return (reference_image[:1].detach() * alpha).contiguous()


def _resize_image_tensor_like(image: torch.Tensor, target: torch.Tensor, *, mode: str = "nearest") -> torch.Tensor:
    if list(image.shape[1:3]) == list(target.shape[1:3]):
        return image
    import torch.nn.functional as F

    kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
    resized = F.interpolate(
        image.detach().float().movedim(-1, 1),
        size=(int(target.shape[1]), int(target.shape[2])),
        mode=mode,
        **kwargs,
    ).movedim(1, -1)
    return resized.to(dtype=target.dtype).contiguous()


def _sample_for_decode(*, model, positive, negative, sampler, sigmas, latent, seed: int, cfg: float) -> dict:
    from comfy_extras.nodes_custom_sampler import SamplerCustom

    sampled = _node_result(
        SamplerCustom.execute(
            model,
            True,
            int(seed),
            float(cfg),
            positive,
            negative,
            sampler,
            sigmas,
            latent,
        )
    )
    if not sampled:
        raise RuntimeError("SamplerCustom returned no latent output.")
    latent_to_decode = sampled[1] if len(sampled) > 1 else sampled[0]
    if not isinstance(latent_to_decode, dict) or "samples" not in latent_to_decode:
        raise RuntimeError("SamplerCustom returned an invalid latent output.")
    samples = latent_to_decode["samples"]
    if hasattr(samples, "detach"):
        samples = samples.detach().contiguous()
    out = {"samples": samples}
    return out


def _decode_latent_to_frames(vae, latent_to_decode: dict) -> torch.Tensor:
    import nodes

    decoded = nodes.VAEDecode().decode(vae, latent_to_decode)[0]
    frames = decoded.detach().cpu().contiguous().clamp(0, 1)
    del decoded, latent_to_decode
    _empty_cache()
    return frames


def _run_original_color_transfer(frames: torch.Tensor, reference_frame: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    info: dict[str, Any] = {
        "applied": False,
        "method": "ColorTransfer",
        "transfer_method": "reinhard_lab",
        "source_stats": "target_frame",
    }
    try:
        import nodes

        node_cls = nodes.NODE_CLASS_MAPPINGS.get("ColorTransfer")
        if node_cls is None:
            info["reason"] = "ColorTransfer_not_registered"
            return frames, info

        node = node_cls()
        function_name = getattr(node, "FUNCTION", getattr(node_cls, "FUNCTION", None))
        if not function_name or not hasattr(node, function_name):
            info["reason"] = "ColorTransfer_function_missing"
            return frames, info

        fn = getattr(node, function_name)
        kwargs: dict[str, Any] = {}
        params = inspect.signature(fn).parameters
        for name, param in params.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in {"image_target", "target", "target_image", "images", "image"}:
                kwargs[name] = frames
            elif name in {"image_ref", "ref", "reference", "reference_image"}:
                kwargs[name] = reference_frame
            elif name == "method":
                kwargs[name] = "reinhard_lab"
            elif name == "source_stats":
                kwargs[name] = "target_frame"
            elif name == "strength":
                kwargs[name] = 0
            elif name in {"source_frame", "target_frame", "source_index", "target_index", "frame", "frame_index"}:
                kwargs[name] = 0
            elif param.default is inspect._empty:
                info["reason"] = f"unsupported_required_input:{name}"
                return frames, info

        result = _node_result(fn(**kwargs))
        if not result or not hasattr(result[0], "shape"):
            info["reason"] = "ColorTransfer_invalid_result"
            return frames, info
        corrected = result[0].detach().cpu().contiguous().clamp(0, 1)
        if list(corrected.shape) != list(frames.shape):
            info["reason"] = f"ColorTransfer_shape_mismatch:{_shape(corrected)}"
            return frames, info
        info["applied"] = True
        return corrected, info
    except Exception as exc:
        info["reason"] = f"ColorTransfer_error:{type(exc).__name__}:{exc}"
        return frames, info


def _fallback_match_chunk_color_to_overlap(
    frames: torch.Tensor,
    current_overlap: Optional[torch.Tensor],
    reference_overlap: Optional[torch.Tensor],
    strength: float = 0.22,
    fade_frames: int = 16,
) -> tuple[torch.Tensor, dict[str, Any]]:
    info: dict[str, Any] = {"applied": False}
    if current_overlap is None or reference_overlap is None or frames.ndim != 4:
        info["reason"] = "missing_overlap"
        return frames, info
    overlap_count = min(int(current_overlap.shape[0]), int(reference_overlap.shape[0]))
    if overlap_count <= 0:
        info["reason"] = "empty_overlap"
        return frames, info

    current = current_overlap[-overlap_count:, :, :, :3].detach().to(frames.device, torch.float32)
    reference = reference_overlap[-overlap_count:, :, :, :3].detach().to(frames.device, torch.float32)
    if current.shape[1:3] != reference.shape[1:3]:
        import torch.nn.functional as F

        reference = F.interpolate(
            reference.movedim(-1, 1),
            size=(int(current.shape[1]), int(current.shape[2])),
            mode="bilinear",
            align_corners=False,
        ).movedim(1, -1)

    current_mean = current.mean(dim=(0, 1, 2))
    reference_mean = reference.mean(dim=(0, 1, 2))
    current_std = current.std(dim=(0, 1, 2), unbiased=False).clamp_min(1e-4)
    reference_std = reference.std(dim=(0, 1, 2), unbiased=False).clamp_min(1e-4)
    scale = (reference_std / current_std).clamp(0.9, 1.1)
    shift = (reference_mean - current_mean * scale).clamp(-0.08, 0.08)

    corrected_rgb = frames[:, :, :, :3].to(torch.float32) * scale.view(1, 1, 1, 3) + shift.view(1, 1, 1, 3)
    blend = torch.full((int(frames.shape[0]), 1, 1, 1), float(strength), device=frames.device, dtype=torch.float32)
    fade_count = min(max(1, int(fade_frames)), int(frames.shape[0]))
    if fade_count < int(frames.shape[0]):
        fade = torch.linspace(float(strength), 0.0, fade_count, device=frames.device, dtype=torch.float32)
        blend.zero_()
        blend[:fade_count, :, :, :] = fade.view(fade_count, 1, 1, 1)
    blended_rgb = torch.lerp(frames[:, :, :, :3].to(torch.float32), corrected_rgb, blend).clamp(0, 1)
    corrected = frames.clone()
    corrected[:, :, :, :3] = blended_rgb.to(dtype=frames.dtype)
    info.update(
        {
            "applied": True,
            "overlap_frames": int(overlap_count),
            "strength": float(strength),
            "fade_frames": int(fade_count),
            "scale": [float(x) for x in scale.detach().cpu()],
            "shift": [float(x) for x in shift.detach().cpu()],
        }
    )
    return corrected.contiguous(), info


def _match_chunk_color_like_original(
    frames: torch.Tensor,
    reference_frame: Optional[torch.Tensor],
    current_overlap: Optional[torch.Tensor],
    reference_overlap: Optional[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if reference_frame is not None and int(reference_frame.shape[0]) > 0:
        corrected, info = _run_original_color_transfer(frames, reference_frame[-1:].contiguous())
        if info.get("applied"):
            return corrected, info
        fallback_reason = info.get("reason", "unknown")
    else:
        fallback_reason = "missing_reference_frame"

    corrected, fallback = _fallback_match_chunk_color_to_overlap(frames, current_overlap, reference_overlap)
    fallback["method"] = "fallback_rgb_overlap"
    fallback["fallback_reason"] = fallback_reason
    return corrected, fallback


class SCAIL2SegmentPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_plan": ("STRING", {"default": DEFAULT_PLAN, "multiline": True}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "pose_frame_count": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "max_chunk_frames": ("INT", {"default": 81, "min": 17, "max": 81, "step": 4}),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 33, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(
        self,
        segment_plan: str,
        max_frames: int = 0,
        pose_frame_count: int = 0,
        max_chunk_frames: int = 81,
        overlap_frames: int = 5,
    ):
        pose_count = int(pose_frame_count) if int(pose_frame_count) > 0 else None
        segments = _parse_plan(segment_plan, pose_frame_count=pose_count, max_frames=max_frames)
        chunks = _build_chunk_plan(segments, max_chunk_frames, overlap_frames)
        planned_frames = sum(segment["frames"] for segment in segments)
        summary = {
            "segments": segments,
            "chunks": chunks,
            "total_frames": planned_frames,
            "pose_frame_count": pose_count,
            "warning": (
                f"segment_plan totals {planned_frames} frames, less than pose_frame_count {pose_count}; "
                "the remaining pose frames will not be generated."
                if pose_count is not None and planned_frames < pose_count
                else None
            ),
        }
        return (json.dumps(summary, indent=2),)


class SCAIL2SegmentPlanBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        required: dict[str, Any] = {
            "segment_count": ("INT", {"default": 2, "min": 1, "max": MAX_REFERENCES, "step": 1}),
        }
        default_frames = [49, 121, 73, 157, 73, 73, 73, 73]
        for index in range(1, MAX_REFERENCES + 1):
            required[f"segment_{index}_frames"] = (
                "INT",
                {"default": default_frames[index - 1], "min": 1, "max": 100000, "step": 1},
            )
            required[f"segment_{index}_reference"] = (
                "INT",
                {"default": min(index, MAX_REFERENCES), "min": 1, "max": MAX_REFERENCES, "step": 1},
            )
            required[f"segment_{index}_prompt"] = (
                "STRING",
                {"default": f"segment {index} prompt", "multiline": True},
            )
            required[f"segment_{index}_negative"] = (
                "STRING",
                {"default": "", "multiline": True},
            )
            required[f"segment_{index}_boundary_overlap"] = (
                "INT",
                {"default": -1 if index == 1 else 1, "min": -1, "max": 33, "step": 1},
            )
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("segment_plan", "summary")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, segment_count: int, **kwargs):
        count = max(1, min(MAX_REFERENCES, int(segment_count)))
        rows: list[dict[str, Any]] = []
        cursor = 0
        for index in range(1, count + 1):
            frames = int(kwargs.get(f"segment_{index}_frames", 0))
            if frames <= 0:
                raise ValueError(f"segment_{index}_frames must be greater than 0.")
            reference = int(kwargs.get(f"segment_{index}_reference", 1))
            if reference < 1 or reference > MAX_REFERENCES:
                raise ValueError(f"segment_{index}_reference must be between 1 and {MAX_REFERENCES}.")
            boundary_overlap = int(kwargs.get(f"segment_{index}_boundary_overlap", -1))
            row = {
                "frames": frames,
                "reference": reference,
                "prompt": kwargs.get(f"segment_{index}_prompt", ""),
                "negative": kwargs.get(f"segment_{index}_negative", ""),
                "boundary_overlap": boundary_overlap if boundary_overlap >= 0 else None,
                "start": cursor,
                "end": cursor + frames,
            }
            rows.append(row)
            cursor += frames

        segment_plan = _format_plan_rows(rows)
        summary = {
            "segments": [
                {
                    "index": index,
                    "frames": int(row["frames"]),
                    "range": [int(row["start"]), int(row["end"])],
                    "reference": int(row["reference"]),
                    "boundary_overlap": row["boundary_overlap"],
                    "prompt": _clean_plan_cell(row["prompt"]),
                    "negative": _clean_plan_cell(row["negative"]),
                }
                for index, row in enumerate(rows)
            ],
            "total_frames": int(cursor),
            "segment_plan": segment_plan,
        }
        return (segment_plan, json.dumps(summary, indent=2))


class SCAIL2MultiReferenceColoredMask:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for index in range(1, MAX_REFERENCES + 1):
            optional[f"reference_{index}_track_data"] = ("SAM3_TRACK_DATA",)
        return {
            "required": {
                "driving_track_data": ("SAM3_TRACK_DATA",),
                "reference_count": ("INT", {"default": 2, "min": 1, "max": MAX_REFERENCES, "step": 1}),
                "object_indices": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated tracked object indices to include, matching the native SCAIL-2 colored-mask node. Empty = all.",
                    },
                ),
                "sort_by": (
                    ["none", "left_to_right", "area"],
                    {
                        "default": "left_to_right",
                        "tooltip": "Native SCAIL-2 identity color ordering. Applies to both driving and reference tracks.",
                    },
                ),
                "replacement_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",) + ("IMAGE",) * MAX_REFERENCES + ("STRING",)
    RETURN_NAMES = (
        "pose_video_mask",
        "reference_1_mask",
        "reference_2_mask",
        "reference_3_mask",
        "reference_4_mask",
        "reference_5_mask",
        "reference_6_mask",
        "reference_7_mask",
        "reference_8_mask",
        "summary",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        driving_track_data,
        reference_count: int = 2,
        object_indices: str = "",
        sort_by: str = "left_to_right",
        replacement_mode: bool = True,
        **kwargs,
    ):
        count = max(1, min(MAX_REFERENCES, int(reference_count)))
        object_indices = str(object_indices or "")
        sort_by = sort_by if sort_by in {"none", "left_to_right", "area"} else "left_to_right"
        pose_video_mask = None
        reference_masks: list[Optional[torch.Tensor]] = [None] * MAX_REFERENCES
        entries: list[dict[str, Any]] = []

        for index in range(1, count + 1):
            ref_track = kwargs.get(f"reference_{index}_track_data")
            if ref_track is None:
                entries.append({"reference": index, "status": "missing_track_data"})
                continue
            current_pose_mask, reference_mask = _create_scail_masks(
                driving_track_data,
                ref_track,
                object_indices,
                sort_by,
                bool(replacement_mode),
            )
            if pose_video_mask is None:
                pose_video_mask = current_pose_mask.detach().contiguous()
            reference_masks[index - 1] = reference_mask.detach().contiguous()
            entries.append(
                {
                    "reference": index,
                    "status": "ok",
                    "pose_video_mask_shape": _shape(current_pose_mask),
                    "reference_mask_shape": _shape(reference_mask),
                }
            )

        if pose_video_mask is None:
            raise ValueError("At least one reference_N_track_data input must be connected.")

        for index, mask in enumerate(reference_masks):
            if mask is None:
                reference_masks[index] = torch.zeros_like(pose_video_mask[:1]).contiguous()

        summary = {
            "reference_count": count,
            "object_indices": object_indices,
            "sort_by": sort_by,
            "replacement_mode": bool(replacement_mode),
            "pose_video_mask_shape": _shape(pose_video_mask),
            "references": entries,
        }
        return (pose_video_mask, *reference_masks, json.dumps(summary, indent=2))


class SCAIL2ScheduledLongVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "pose_video_mask": ("IMAGE",),
        }
        for index in range(1, MAX_REFERENCES + 1):
            optional[f"reference_{index}"] = ("IMAGE",)
            optional[f"reference_{index}_mask"] = ("IMAGE",)

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "clip_vision": ("CLIP_VISION",),
                "pose_video": ("IMAGE",),
                "segment_plan": ("STRING", {"default": DEFAULT_PLAN, "multiline": True}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "mode": (["replacement", "animation"], {"default": "replacement"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "max_chunk_frames": ("INT", {"default": 81, "min": 17, "max": 81, "step": 4}),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 33, "step": 1}),
                "reference_count": ("INT", {"default": 2, "min": 1, "max": MAX_REFERENCES, "step": 1}),
                "color_correction": ("BOOLEAN", {"default": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("frames", "used_pose_video_mask", "used_reference_mask_timeline", "summary")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                fingerprint_inputs[key] = value
            else:
                fingerprint_inputs[key] = type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def generate(
        self,
        model,
        clip,
        vae,
        sampler,
        sigmas,
        clip_vision,
        pose_video: torch.Tensor,
        segment_plan: str,
        seed: int,
        cfg: float,
        mode: str,
        max_frames: int,
        max_chunk_frames: int,
        overlap_frames: int,
        reference_count: int,
        color_correction: bool,
        pose_video_mask=None,
        **kwargs,
    ):
        if not isinstance(pose_video, torch.Tensor) or pose_video.ndim != 4:
            raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")

        call_cache_key = _stable_fingerprint(
            {
                "node": "SCAIL2ScheduledLongVideo",
                "model": _cache_marker(model),
                "clip": _cache_marker(clip),
                "vae": _cache_marker(vae),
                "sampler": _cache_marker(sampler),
                "sigmas": _cache_marker(sigmas),
                "clip_vision": _cache_marker(clip_vision),
                "pose_video": _tensor_fingerprint(pose_video),
                "segment_plan": segment_plan,
                "seed": int(seed),
                "cfg": float(cfg),
                "mode": mode,
                "max_frames": int(max_frames),
                "max_chunk_frames": int(max_chunk_frames),
                "overlap_frames": int(overlap_frames),
                "reference_count": int(reference_count),
                "color_correction": bool(color_correction),
                "pose_video_mask": _tensor_fingerprint(pose_video_mask),
                "dynamic_inputs": _cache_marker(kwargs),
            }
        )
        cached_key = getattr(self, "_last_generate_cache_key", None)
        cached_result = getattr(self, "_last_generate_cache_result", None)
        if cached_key == call_cache_key and cached_result is not None:
            print("[SCAIL2ScheduledLongVideo] cache hit; returning previous result without sampling.")
            return _clone_cached_result(cached_result)

        total_pose_frames = int(pose_video.shape[0])
        segments = _parse_plan(segment_plan, pose_frame_count=total_pose_frames, max_frames=max_frames)
        width, height = _infer_generation_size(pose_video)
        max_chunk_frames = min(81, _ceil_to_4n_plus_1(max(17, int(max_chunk_frames))))
        max_chunk_frames = min(81, max_chunk_frames)
        requested_overlap = max(0, int(overlap_frames))
        overlap = 0 if requested_overlap == 0 else min(_floor_to_4n_plus_1(requested_overlap), 33, max_chunk_frames - 4)
        replacement_mode = mode == "replacement"
        planned_chunks = _build_chunk_plan(segments, max_chunk_frames, overlap)
        planned_frames = sum(segment["frames"] for segment in segments)
        print(
            "[SCAIL2ScheduledLongVideo] "
            f"pose_frames={total_pose_frames} planned_frames={planned_frames} "
            f"chunks={len(planned_chunks)} max_chunk_frames={max_chunk_frames} overlap={overlap}"
        )
        if planned_frames < total_pose_frames:
            print(
                "[SCAIL2ScheduledLongVideo] WARNING "
                f"segment_plan totals {planned_frames} frames; "
                f"{total_pose_frames - planned_frames} pose frames will be ignored."
            )
        for chunk in planned_chunks:
            print(
                "[SCAIL2ScheduledLongVideo] plan "
                f"chunk={chunk['chunk_index']} segment={chunk['segment_index']} "
                f"ref={chunk['reference']} gen={chunk['generate_length']} "
                f"discard={chunk['discard_head']} keep={chunk['keep_frames']} "
                f"out={chunk['output_start']}:{chunk['output_end']}"
            )

        references: dict[int, torch.Tensor] = {}
        reference_masks: dict[int, torch.Tensor] = {}
        active_reference_count = max(1, min(MAX_REFERENCES, int(reference_count)))
        for index in range(1, active_reference_count + 1):
            image = kwargs.get(f"reference_{index}")
            if image is not None:
                references[index] = _first_image(image, f"reference_{index}")
            mask = kwargs.get(f"reference_{index}_mask")
            if mask is not None:
                reference_masks[index] = mask.detach().contiguous()
        if not references:
            raise ValueError("At least one reference_N image must be connected.")

        used_refs = sorted({int(segment["reference"]) for segment in segments})
        missing = [index for index in used_refs if index not in references]
        if missing:
            raise ValueError(f"segment_plan references missing image input(s): {missing}")

        if replacement_mode:
            if pose_video_mask is None:
                raise ValueError("replacement mode requires pose_video_mask from Create SCAIL-2 Colored Mask.")
            missing_masks = [index for index in used_refs if index not in reference_masks]
            if missing_masks:
                raise ValueError(
                    "replacement mode requires reference_N_mask for every used reference. "
                    f"Missing: {missing_masks}"
                )

        reference_cache: dict[int, dict[str, Any]] = {}
        prompt_cache: dict[tuple[str, str], tuple[Any, Any]] = {}

        def get_prompt(prompt: str, negative: str):
            key = (prompt, negative)
            if key not in prompt_cache:
                prompt_cache[key] = (_encode_text(clip, prompt), _encode_text(clip, negative))
            return prompt_cache[key]

        def get_reference(index: int):
            if index in reference_cache:
                return reference_cache[index]
            reference_image = references[index]
            reference_mask = reference_masks.get(index)
            clip_image = _apply_reference_mask(reference_image, reference_mask) if replacement_mode else reference_image
            clip_vision_output = _encode_clip_vision(clip_vision, clip_image)
            reference_cache[index] = {
                "reference_image": reference_image,
                "reference_image_mask": reference_mask,
                "clip_vision_output": clip_vision_output,
                "reference_shape": _shape(reference_image),
                "reference_mask_shape": _shape(reference_mask),
            }
            return reference_cache[index]

        prompt_pairs = sorted({(segment["prompt"], segment["negative"]) for segment in segments})
        print(
            "[SCAIL2ScheduledLongVideo] prewarm "
            f"references={used_refs} prompt_pairs={len(prompt_pairs)}"
        )
        for index in used_refs:
            get_reference(index)
        for prompt, negative in prompt_pairs:
            get_prompt(prompt, negative)
        print("[SCAIL2ScheduledLongVideo] prewarm done")

        WanSCAILToVideo = _get_scail_nodes_module().WanSCAILToVideo
        stitched: list[torch.Tensor] = []
        stitched_pose_masks: list[torch.Tensor] = []
        stitched_reference_masks: list[torch.Tensor] = []
        previous_frames = None
        produced = 0
        chunk_index = 0
        chunk_summaries: list[dict[str, Any]] = []
        last_segment_reference = None

        for segment in segments:
            remaining = int(segment["frames"])
            segment_kept = 0
            segment_reference = int(segment["reference"])
            is_reference_change_segment = (
                last_segment_reference is not None and segment_reference != int(last_segment_reference)
            )
            ref_info = get_reference(segment_reference)
            positive, negative = get_prompt(segment["prompt"], segment["negative"])

            while remaining > 0:
                boundary_override = (
                    segment.get("boundary_overlap")
                    if is_reference_change_segment and segment_kept == 0
                    else None
                )
                effective_overlap = (
                    _normalize_overlap(boundary_override, max_chunk_frames)
                    if boundary_override is not None
                    else overlap
                )
                has_previous = previous_frames is not None and effective_overlap > 0
                actual_overlap = (
                    min(int(effective_overlap), int(previous_frames.shape[0]))
                    if has_previous and previous_frames is not None
                    else 0
                )
                max_keep = max_chunk_frames if not has_previous else max_chunk_frames - actual_overlap
                wanted_keep = min(remaining, max_keep)
                if wanted_keep <= 0:
                    raise RuntimeError("Internal planner produced an empty chunk.")
                raw_length = wanted_keep if not has_previous else wanted_keep + actual_overlap
                length = _ceil_to_4n_plus_1(raw_length)
                if length < 17 and remaining > 1:
                    length = 17
                if length > max_chunk_frames:
                    length = max_chunk_frames
                    wanted_keep = length if not has_previous else length - actual_overlap
                if wanted_keep <= 0:
                    raise RuntimeError("overlap_frames leaves no room for new frames.")
                video_frame_offset = int(produced)
                internal_window_offset = max(0, int(produced) - int(actual_overlap))
                print(
                    "[SCAIL2ScheduledLongVideo] run "
                    f"chunk={chunk_index} segment={segment['index']} ref={segment['reference']} "
                    f"gen={length} discard={'pending'} keep_target={wanted_keep} "
                    f"offset_input={video_frame_offset} internal_window_offset={internal_window_offset} "
                    f"produced_before={produced}"
                )
                previous_frames_for_scail = (
                    previous_frames[-actual_overlap:].contiguous()
                    if previous_frames is not None and actual_overlap > 0
                    else None
                )

                scail_out = _node_result(
                    WanSCAILToVideo.execute(
                        positive,
                        negative,
                        vae,
                        width,
                        height,
                        int(length),
                        1,
                        1.0,
                        0.0,
                        1.0,
                        int(video_frame_offset),
                        int(actual_overlap) if actual_overlap > 0 else 1,
                        replacement_mode=replacement_mode,
                        reference_image=ref_info["reference_image"],
                        clip_vision_output=ref_info["clip_vision_output"],
                        pose_video=pose_video,
                        pose_video_mask=pose_video_mask,
                        reference_image_mask=ref_info["reference_image_mask"],
                        previous_frames=previous_frames_for_scail,
                    )
                )
                if len(scail_out) != 4:
                    raise RuntimeError("WanSCAILToVideo returned an unexpected result.")
                chunk_positive, chunk_negative, latent, _next_offset = scail_out
                latent_to_decode = _sample_for_decode(
                    model=model,
                    positive=chunk_positive,
                    negative=chunk_negative,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent=latent,
                    seed=int(seed) + chunk_index,
                    cfg=float(cfg),
                )
                decoded = _decode_latent_to_frames(vae, latent_to_decode)

                discard_head = min(actual_overlap, int(decoded.shape[0])) if has_previous else 0
                current_overlap = decoded[:discard_head].contiguous() if discard_head > 0 else None
                reference_overlap = previous_frames[-discard_head:].contiguous() if previous_frames is not None and discard_head > 0 else None
                kept = decoded[discard_head : discard_head + wanted_keep].contiguous()
                if int(kept.shape[0]) <= 0:
                    raise RuntimeError("A chunk produced no keepable frames.")
                if color_correction and discard_head > 0:
                    kept, color_summary = _match_chunk_color_like_original(
                        kept,
                        previous_frames[-1:].contiguous() if previous_frames is not None else None,
                        current_overlap,
                        reference_overlap,
                    )
                else:
                    color_summary = {"applied": False}

                stitched.append(kept.detach().cpu().contiguous())
                if replacement_mode:
                    pose_mask_window = pose_video_mask[
                        video_frame_offset : video_frame_offset + int(decoded.shape[0])
                    ].detach().cpu().contiguous()
                    kept_pose_mask = pose_mask_window[discard_head : discard_head + wanted_keep].contiguous()
                    if int(kept_pose_mask.shape[0]) != int(kept.shape[0]):
                        kept_pose_mask = torch.zeros_like(kept)
                    else:
                        kept_pose_mask = _resize_image_tensor_like(kept_pose_mask, kept)
                    stitched_pose_masks.append(kept_pose_mask)

                    reference_mask = ref_info["reference_image_mask"]
                    if reference_mask is None:
                        kept_reference_mask = torch.zeros_like(kept)
                    else:
                        kept_reference_mask = reference_mask[:1].detach().cpu().contiguous()
                        kept_reference_mask = kept_reference_mask.repeat(int(kept.shape[0]), 1, 1, 1)
                        kept_reference_mask = _resize_image_tensor_like(kept_reference_mask, kept)
                    stitched_reference_masks.append(kept_reference_mask)
                produced += int(kept.shape[0])
                segment_kept += int(kept.shape[0])
                remaining -= int(kept.shape[0])
                if overlap > 0:
                    previous_frames = torch.cat(stitched, dim=0)[-overlap:].contiguous()
                else:
                    previous_frames = None

                chunk_summaries.append(
                    {
                        "chunk_index": chunk_index,
                        "segment_index": int(segment["index"]),
                        "segment_frame_start": int(segment["start"]),
                        "segment_frame_end": int(segment["end"]),
                        "reference": int(segment["reference"]),
                        "boundary_overlap": boundary_override,
                        "prompt": segment["prompt"],
                        "negative": segment["negative"],
                        "video_frame_offset": int(video_frame_offset),
                        "internal_window_offset": int(internal_window_offset),
                        "output_start": int(produced - kept.shape[0]),
                        "generate_length": int(length),
                        "discard_head": int(discard_head),
                        "kept_frames": int(kept.shape[0]),
                        "produced_total": int(produced),
                        "color_correction": color_summary,
                    }
                )
                print(
                    "[SCAIL2ScheduledLongVideo] done "
                    f"chunk={chunk_index} kept={int(kept.shape[0])} produced_total={produced}"
                )
                chunk_index += 1
                del decoded, kept, current_overlap, reference_overlap, scail_out, latent
            last_segment_reference = segment_reference

        frames = torch.cat(stitched, dim=0).contiguous().clamp(0, 1)
        if stitched_pose_masks:
            used_pose_video_mask = torch.cat(stitched_pose_masks, dim=0).contiguous().clamp(0, 1)
        else:
            used_pose_video_mask = torch.zeros_like(frames)
        if stitched_reference_masks:
            used_reference_mask_timeline = torch.cat(stitched_reference_masks, dim=0).contiguous().clamp(0, 1)
        else:
            used_reference_mask_timeline = torch.zeros_like(frames)
        summary = {
            "mode": mode,
            "width": int(width),
            "height": int(height),
            "source_pose_frames": int(total_pose_frames),
            "planned_frames": int(sum(segment["frames"] for segment in segments)),
            "generated_frames": int(frames.shape[0]),
            "max_chunk_frames": int(max_chunk_frames),
            "overlap_frames": int(overlap),
            "reference_count": int(active_reference_count),
            "cfg": float(cfg),
            "seed_start": int(seed),
            "segments": segments,
            "planned_chunks": planned_chunks,
            "references_used": used_refs,
            "prewarmed_prompt_pairs": int(len(prompt_cache)),
            "reference_cache": {
                str(index): {
                    "reference_shape": value["reference_shape"],
                    "reference_mask_shape": value["reference_mask_shape"],
                }
                for index, value in reference_cache.items()
            },
            "chunks": chunk_summaries,
        }
        result = (frames, used_pose_video_mask, used_reference_mask_timeline, json.dumps(summary, indent=2))
        self._last_generate_cache_key = call_cache_key
        self._last_generate_cache_result = _clone_cached_result(result)
        _empty_cache(force=True)
        return result


class SCAIL2ScheduledLongVideoWithSAM(SCAIL2ScheduledLongVideo):
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "sam_model": ("MODEL",),
            "sam_conditioning": ("CONDITIONING",),
        }
        for index in range(1, MAX_REFERENCES + 1):
            optional[f"reference_{index}"] = ("IMAGE",)

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "clip_vision": ("CLIP_VISION",),
                "pose_video": ("IMAGE",),
                "segment_plan": ("STRING", {"default": DEFAULT_PLAN, "multiline": True}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "mode": (["replacement", "animation"], {"default": "replacement"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "max_chunk_frames": ("INT", {"default": 81, "min": 17, "max": 81, "step": 4}),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 33, "step": 1}),
                "reference_count": ("INT", {"default": 2, "min": 1, "max": MAX_REFERENCES, "step": 1}),
                "color_correction": ("BOOLEAN", {"default": True}),
                "object_indices": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated driving-video object indices to include after sorting. Empty = all.",
                    },
                ),
                "reference_object_indices": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma-separated reference-image object indices. Empty = all reference objects, recommended for single-person reference images.",
                    },
                ),
                "sort_by": (
                    ["none", "left_to_right", "area"],
                    {
                        "default": "left_to_right",
                        "tooltip": "Native SCAIL-2 identity color ordering before object index filtering.",
                    },
                ),
                "sam_detection_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "sam_max_objects": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "sam_detect_interval": ("INT", {"default": 2, "min": 1, "max": 999, "step": 1}),
            },
            "optional": optional,
        }

    RETURN_TYPES = SCAIL2ScheduledLongVideo.RETURN_TYPES
    RETURN_NAMES = SCAIL2ScheduledLongVideo.RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        model,
        clip,
        vae,
        sampler,
        sigmas,
        clip_vision,
        pose_video: torch.Tensor,
        segment_plan: str,
        seed: int,
        cfg: float,
        mode: str,
        max_frames: int,
        max_chunk_frames: int,
        overlap_frames: int,
        reference_count: int,
        color_correction: bool,
        object_indices: str = "",
        reference_object_indices: str = "",
        sort_by: str = "left_to_right",
        sam_detection_threshold: float = 0.5,
        sam_max_objects: int = 2,
        sam_detect_interval: int = 2,
        sam_model=None,
        sam_conditioning=None,
        **kwargs,
    ):
        if not isinstance(pose_video, torch.Tensor) or pose_video.ndim != 4:
            raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")

        active_reference_count = max(1, min(MAX_REFERENCES, int(reference_count)))
        segments = _parse_plan(segment_plan, pose_frame_count=int(pose_video.shape[0]), max_frames=max_frames)
        used_refs = sorted({int(segment["reference"]) for segment in segments})
        object_indices = str(object_indices or "")
        reference_object_indices = str(reference_object_indices or "")
        sort_by = sort_by if sort_by in {"none", "left_to_right", "area"} else "left_to_right"
        replacement_mode = mode == "replacement"

        references: dict[int, torch.Tensor] = {}
        for index in range(1, active_reference_count + 1):
            image = kwargs.get(f"reference_{index}")
            if image is not None:
                references[index] = _first_image(image, f"reference_{index}")
        missing = [index for index in used_refs if index not in references]
        if missing:
            raise ValueError(f"segment_plan references missing image input(s): {missing}")

        call_cache_key = _stable_fingerprint(
            {
                "node": "SCAIL2ScheduledLongVideoWithSAM",
                "model": _cache_marker(model),
                "clip": _cache_marker(clip),
                "vae": _cache_marker(vae),
                "sampler": _cache_marker(sampler),
                "sigmas": _cache_marker(sigmas),
                "clip_vision": _cache_marker(clip_vision),
                "pose_video": _tensor_fingerprint(pose_video),
                "sam_model": _cache_marker(sam_model),
                "sam_conditioning": _cache_marker(sam_conditioning),
                "segment_plan": segment_plan,
                "seed": int(seed),
                "cfg": float(cfg),
                "mode": mode,
                "max_frames": int(max_frames),
                "max_chunk_frames": int(max_chunk_frames),
                "overlap_frames": int(overlap_frames),
                "reference_count": int(reference_count),
                "color_correction": bool(color_correction),
                "object_indices": object_indices,
                "reference_object_indices": reference_object_indices,
                "sort_by": sort_by,
                "sam_detection_threshold": float(sam_detection_threshold),
                "sam_max_objects": int(sam_max_objects),
                "sam_detect_interval": int(sam_detect_interval),
                "dynamic_inputs": _cache_marker(kwargs),
            }
        )
        cached_key = getattr(self, "_last_internal_sam_cache_key", None)
        cached_result = getattr(self, "_last_internal_sam_cache_result", None)
        if cached_key == call_cache_key and cached_result is not None:
            print("[SCAIL2ScheduledLongVideoWithSAM] cache hit; returning previous result without SAM tracking.")
            return _clone_cached_result(cached_result)

        if not replacement_mode:
            generate_kwargs: dict[str, Any] = {}
            for index, image in references.items():
                generate_kwargs[f"reference_{index}"] = image
            frames, used_pose_video_mask, used_reference_mask_timeline, summary_text = super().generate(
                model,
                clip,
                vae,
                sampler,
                sigmas,
                clip_vision,
                pose_video,
                segment_plan,
                seed,
                cfg,
                mode,
                max_frames,
                max_chunk_frames,
                overlap_frames,
                reference_count,
                color_correction,
                **generate_kwargs,
            )
            try:
                summary = json.loads(summary_text)
            except Exception:
                summary = {"scheduled_summary": summary_text}
            summary["internal_sam"] = {
                "skipped": True,
                "reason": "animation mode does not use SCAIL replacement masks",
            }
            result = (frames, used_pose_video_mask, used_reference_mask_timeline, json.dumps(summary, indent=2))
            self._last_internal_sam_cache_key = call_cache_key
            self._last_internal_sam_cache_result = _clone_cached_result(result)
            return result

        if sam_model is None or sam_conditioning is None:
            raise ValueError("replacement mode requires sam_model and sam_conditioning.")

        print(
            "[SCAIL2ScheduledLongVideoWithSAM] "
            f"tracking pose_video frames={int(pose_video.shape[0])} refs={used_refs} "
            f"object_indices='{object_indices}' reference_object_indices='{reference_object_indices}' "
            f"sort_by={sort_by}"
        )
        driving_track_data = _run_sam3_track(
            pose_video,
            sam_model,
            sam_conditioning,
            detection_threshold=float(sam_detection_threshold),
            max_objects=int(sam_max_objects),
            detect_interval=int(sam_detect_interval),
        )

        pose_video_mask = None
        reference_masks: dict[int, torch.Tensor] = {}
        track_summary: list[dict[str, Any]] = []
        for index in used_refs:
            reference_track_data = _run_sam3_track(
                references[index],
                sam_model,
                sam_conditioning,
                detection_threshold=float(sam_detection_threshold),
                max_objects=int(sam_max_objects),
                detect_interval=1,
            )

            current_pose_mask, _ = _create_scail_masks(
                driving_track_data,
                reference_track_data,
                object_indices,
                sort_by,
                replacement_mode,
            )
            _, reference_mask = _create_scail_masks(
                driving_track_data,
                reference_track_data,
                reference_object_indices,
                sort_by,
                replacement_mode,
            )
            if pose_video_mask is None:
                pose_video_mask = current_pose_mask.detach().contiguous()
            reference_masks[index] = reference_mask.detach().contiguous()
            track_summary.append(
                {
                    "reference": index,
                    "reference_shape": _shape(references[index]),
                    "reference_mask_shape": _shape(reference_mask),
                }
            )

        if pose_video_mask is None:
            raise RuntimeError("Internal SAM tracking produced no pose_video_mask.")

        generate_kwargs: dict[str, Any] = {"pose_video_mask": pose_video_mask}
        for index, image in references.items():
            generate_kwargs[f"reference_{index}"] = image
        for index, mask in reference_masks.items():
            generate_kwargs[f"reference_{index}_mask"] = mask

        frames, used_pose_video_mask, used_reference_mask_timeline, summary_text = super().generate(
            model,
            clip,
            vae,
            sampler,
            sigmas,
            clip_vision,
            pose_video,
            segment_plan,
            seed,
            cfg,
            mode,
            max_frames,
            max_chunk_frames,
            overlap_frames,
            reference_count,
            color_correction,
            **generate_kwargs,
        )
        try:
            summary = json.loads(summary_text)
        except Exception:
            summary = {"scheduled_summary": summary_text}
        summary["internal_sam"] = {
            "object_indices": object_indices,
            "reference_object_indices": reference_object_indices,
            "sort_by": sort_by,
            "sam_detection_threshold": float(sam_detection_threshold),
            "sam_max_objects": int(sam_max_objects),
            "sam_detect_interval": int(sam_detect_interval),
            "references_tracked": track_summary,
            "pose_video_mask_shape": _shape(pose_video_mask),
        }
        result = (frames, used_pose_video_mask, used_reference_mask_timeline, json.dumps(summary, indent=2))
        self._last_internal_sam_cache_key = call_cache_key
        self._last_internal_sam_cache_result = _clone_cached_result(result)
        return result


NODE_CLASS_MAPPINGS = {
    "SCAIL2SegmentPlanBuilder": SCAIL2SegmentPlanBuilder,
    "SCAIL2SegmentPlanner": SCAIL2SegmentPlanner,
    "SCAIL2MultiReferenceColoredMask": SCAIL2MultiReferenceColoredMask,
    "SCAIL2ScheduledLongVideo": SCAIL2ScheduledLongVideo,
    "SCAIL2ScheduledLongVideoWithSAM": SCAIL2ScheduledLongVideoWithSAM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2SegmentPlanBuilder": "SCAIL-2 Segment Plan Builder",
    "SCAIL2SegmentPlanner": "SCAIL-2 Segment Planner",
    "SCAIL2MultiReferenceColoredMask": "SCAIL-2 Multi Reference Colored Mask",
    "SCAIL2ScheduledLongVideo": "SCAIL-2 Scheduled Long Video",
    "SCAIL2ScheduledLongVideoWithSAM": "SCAIL-2 Scheduled Long Video (Internal SAM)",
}

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import time
from typing import Any, Optional

import torch


CATEGORY = "SCAIL-2/Scheduled"
MAX_REFERENCES = 8
DEFAULT_PLAN = """# frames | reference | prompt | negative | boundary_overlap
49 | 1 | first segment prompt | | 5
121 | 2 | second segment prompt | | 5
73 | 3 | third segment prompt | | 5
157 | 4 | fourth segment prompt | | 5
"""
_INSIGHTFACE_APP_CACHE: dict[tuple[str, str, int], Any] = {}
_MEDIAPIPE_FACE_DETECTION_CACHE: dict[tuple[int, float], Any] = {}


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
    return {"type": type(value).__name__}


def _prompt_graph(prompt: Any) -> Optional[dict[str, Any]]:
    if not isinstance(prompt, dict):
        return None
    nested = prompt.get("prompt")
    if isinstance(nested, dict):
        return nested
    return prompt


def _prompt_get_node(graph: dict[str, Any], node_id: Any) -> Optional[dict[str, Any]]:
    for key in (node_id, str(node_id), int(node_id) if str(node_id).isdigit() else None):
        if key is None:
            continue
        node = graph.get(key)
        if isinstance(node, dict):
            return node
    return None


def _prompt_is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _prompt_upstream_fingerprint(prompt: Any, unique_id: Any) -> Any:
    graph = _prompt_graph(prompt)
    if graph is None or unique_id is None:
        return None

    visited: set[str] = set()
    stack: list[Any] = [unique_id]
    collected: dict[str, Any] = {}
    while stack:
        node_id = stack.pop()
        node_key = str(node_id)
        if node_key in visited:
            continue
        visited.add(node_key)
        node = _prompt_get_node(graph, node_id)
        if node is None:
            continue
        inputs = node.get("inputs", {}) if isinstance(node.get("inputs", {}), dict) else {}
        normalized_inputs: dict[str, Any] = {}
        for input_name in sorted(inputs):
            value = inputs[input_name]
            if _prompt_is_link(value) and _prompt_get_node(graph, value[0]) is not None:
                source_id, source_output = value
                normalized_inputs[input_name] = ["LINK", str(source_id), int(source_output)]
                stack.append(source_id)
            else:
                normalized_inputs[input_name] = _cache_marker(value)
        collected[node_key] = {
            "class_type": node.get("class_type"),
            "inputs": normalized_inputs,
        }
    return _stable_fingerprint(collected)


def _clone_cached_result(result: tuple) -> tuple:
    return tuple(result)


def _scail2_cache_root() -> str:
    try:
        import folder_paths

        root = folder_paths.get_output_directory()
    except Exception:
        root = os.path.join(os.getcwd(), "output")
    return os.path.join(root, "scail2_cache", "long_video")


def _safe_cache_token(value: Any) -> str:
    text = str(value if value is not None else "global")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)
    return safe[:96] or "global"


def _single_slot_cache_paths(node_name: str, unique_id: Any) -> tuple[str, str, str]:
    cache_dir = os.path.join(_scail2_cache_root(), _safe_cache_token(node_name), _safe_cache_token(unique_id))
    return cache_dir, os.path.join(cache_dir, "cache.pt"), os.path.join(cache_dir, "meta.json")


def _torch_load_cpu(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_single_slot_disk_cache(node_name: str, unique_id: Any, cache_key: str) -> Optional[tuple]:
    _cache_dir, cache_path, meta_path = _single_slot_cache_paths(node_name, unique_id)
    if not os.path.exists(cache_path):
        return None
    try:
        payload = _torch_load_cpu(cache_path)
        if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("key") != cache_key:
            return None
        result = payload.get("result")
        if not isinstance(result, (list, tuple)) or len(result) != 4:
            return None
        frames, pose_mask, reference_mask, summary = result
        if not all(isinstance(item, torch.Tensor) for item in (frames, pose_mask, reference_mask)):
            return None
        if not isinstance(summary, str):
            return None
        now = time.time()
        try:
            os.utime(cache_path, (now, now))
            if os.path.exists(meta_path):
                os.utime(meta_path, (now, now))
        except Exception:
            pass
        return (
            frames.detach().cpu().contiguous(),
            pose_mask.detach().cpu().contiguous(),
            reference_mask.detach().cpu().contiguous(),
            summary,
        )
    except Exception as exc:
        print(f"[SCAIL2DiskCache] cache read failed; recomputing. reason={exc}")
        return None


def _save_single_slot_disk_cache(node_name: str, unique_id: Any, cache_key: str, result: tuple) -> None:
    cache_dir, cache_path, meta_path = _single_slot_cache_paths(node_name, unique_id)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        frames, pose_mask, reference_mask, summary = result
        payload = {
            "version": 1,
            "key": cache_key,
            "saved_at": time.time(),
            "node": node_name,
            "unique_id": str(unique_id),
            "result": (
                frames.detach().cpu().contiguous(),
                pose_mask.detach().cpu().contiguous(),
                reference_mask.detach().cpu().contiguous(),
                str(summary),
            ),
        }
        tmp_path = cache_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, cache_path)
        meta = {
            "version": 1,
            "key": cache_key,
            "saved_at": payload["saved_at"],
            "node": node_name,
            "unique_id": str(unique_id),
            "frames_shape": _shape(frames),
            "pose_mask_shape": _shape(pose_mask),
            "reference_mask_shape": _shape(reference_mask),
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"[SCAIL2DiskCache] cache write failed; continuing without disk cache. reason={exc}")


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


def _normalize_summary_segments(summary: dict[str, Any], max_frames: int = 0) -> list[dict[str, Any]]:
    raw_segments = summary.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("planner_summary must contain a non-empty chunks, planned_chunks, or segments list.")

    segments: list[dict[str, Any]] = []
    cursor = 0
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise ValueError(f"planner_summary segment {index} must be an object.")
        frames = int(item.get("frames", 0))
        range_value = item.get("range")
        start = item.get("start", None)
        end = item.get("end", None)
        if isinstance(range_value, list) and len(range_value) >= 2:
            start = range_value[0] if start is None else start
            end = range_value[1] if end is None else end
        if start is None:
            start = cursor
        if end is None:
            end = int(start) + frames
        start = int(start)
        end = int(end)
        if frames <= 0:
            frames = end - start
        if frames <= 0 or end <= start:
            raise ValueError(f"planner_summary segment {index} has invalid frame range {start}:{end}.")
        segments.append(
            {
                "index": int(item.get("index", index)),
                "start": start,
                "end": end,
                "frames": frames,
                "reference": int(item.get("reference", 1)),
                "boundary_overlap": item.get("boundary_overlap", None),
                "prompt": str(item.get("prompt", "")),
                "negative": str(item.get("negative", "")),
            }
        )
        cursor = end

    cap = int(max_frames) if int(max_frames) > 0 else None
    if cap is not None and segments and int(segments[-1]["end"]) > cap:
        clipped: list[dict[str, Any]] = []
        for segment in segments:
            if int(segment["start"]) >= cap:
                break
            clipped_segment = dict(segment)
            clipped_segment["end"] = min(int(segment["end"]), cap)
            clipped_segment["frames"] = int(clipped_segment["end"]) - int(clipped_segment["start"])
            if int(clipped_segment["frames"]) > 0:
                clipped.append(clipped_segment)
        segments = clipped
    if not segments:
        raise ValueError("planner_summary segments produce no frames after max_frames/video length clipping.")
    return segments


def _parse_planner_chunks(
    planner_summary: str,
    max_chunk_frames: int = 81,
    overlap_frames: int = 5,
    max_frames: int = 0,
) -> tuple[list[dict[str, Any]], str]:
    text = str(planner_summary or "").strip()
    if not text:
        raise ValueError("planner_summary is required when mode is planner_summary.")
    try:
        summary = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"planner_summary must be valid JSON from SCAIL2SegmentPlanner: {exc}") from exc
    if not isinstance(summary, dict):
        raise ValueError("planner_summary must be a JSON object.")
    chunks = summary.get("chunks", summary.get("planned_chunks"))
    if not isinstance(chunks, list) or not chunks:
        segments = _normalize_summary_segments(summary, max_frames=max_frames)
        return _build_chunk_plan(segments, max_chunk_frames, overlap_frames), "segment_summary"
    normalized: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"planner_summary chunk {index} must be an object.")
        try:
            output_start = int(chunk["output_start"])
            output_end = int(chunk["output_end"])
        except KeyError as exc:
            raise ValueError(f"planner_summary chunk {index} is missing {exc.args[0]}.") from exc
        if output_start < 0 or output_end <= output_start:
            raise ValueError(f"planner_summary chunk {index} has invalid output range {output_start}:{output_end}.")
        normalized.append({**chunk, "output_start": output_start, "output_end": output_end})
    return normalized, "planner_summary"


def _chunk_boundary_indices(
    chunks: list[dict[str, Any]],
    frame_count: int,
    include_final_anchor: bool,
    boundary_anchor_mode: str = "overlap_last_frame",
) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    if not chunks:
        raise ValueError("chunk plan produced no chunks.")
    normalized_anchor_mode = (
        "overlap_first_frame"
        if str(boundary_anchor_mode or "").strip() == "overlap_first_frame"
        else "overlap_last_frame"
    )
    anchors: list[int] = []
    starts: list[int] = []
    rows: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        output_start = int(chunk["output_start"])
        output_end = int(chunk["output_end"])
        discard_head = int(chunk.get("discard_head", 0))
        if index == 0:
            anchor_index = 0
            boundary_anchor_source = "video_start"
        elif normalized_anchor_mode == "overlap_first_frame" and discard_head > 0:
            anchor_index = output_start - discard_head
            boundary_anchor_source = "overlap_first"
        else:
            anchor_index = output_start - 1
            boundary_anchor_source = "overlap_last"
        start_index = output_start
        if anchor_index < 0 or start_index < 0 or output_end <= output_start:
            raise ValueError(f"chunk {index} has invalid output range {output_start}:{output_end}.")
        if anchor_index >= frame_count or start_index >= frame_count or output_end > frame_count:
            raise ValueError(
                f"chunk {index} references frames beyond the input video: "
                f"range={output_start}:{output_end}, video_frames={frame_count}."
            )
        anchors.append(anchor_index)
        starts.append(start_index)
        rows.append(
            {
                "chunk_index": int(chunk.get("chunk_index", index)),
                "reference": int(chunk.get("reference", 1)),
                "output_range_0_based": [output_start, output_end],
                "output_range_1_based_inclusive": [output_start + 1, output_end],
                "boundary_anchor_frame_0_based": anchor_index,
                "boundary_anchor_frame_1_based": anchor_index + 1,
                "boundary_anchor_mode": normalized_anchor_mode,
                "boundary_anchor_source": boundary_anchor_source,
                "new_chunk_start_frame_0_based": start_index,
                "new_chunk_start_frame_1_based": start_index + 1,
                "discard_head": discard_head,
                "keep_frames": int(chunk.get("keep_frames", output_end - output_start)),
                "generate_length": int(chunk.get("generate_length", output_end - output_start)),
                "boundary_overlap": chunk.get("boundary_overlap", None),
            }
        )
    if include_final_anchor:
        final_index = int(chunks[-1]["output_end"]) - 1
        if final_index < 0 or final_index >= frame_count:
            raise ValueError(f"final anchor frame {final_index} is outside the input video.")
        anchors.append(final_index)
    return anchors, starts, rows


def _extract_frame_batch(video: torch.Tensor, indices: list[int], name: str) -> torch.Tensor:
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError("video must be a ComfyUI IMAGE tensor.")
    if int(video.shape[0]) <= 0:
        raise ValueError("video has no frames.")
    if not indices:
        raise ValueError(f"{name} produced no frame indices.")
    frame_count = int(video.shape[0])
    for index in indices:
        if int(index) < 0 or int(index) >= frame_count:
            raise ValueError(f"{name} frame index {index} is outside input video frame count {frame_count}.")
    selector = torch.tensor(indices, device=video.device, dtype=torch.long)
    return video.index_select(0, selector).detach().contiguous()


def _tensor_frame_to_pil(frame: torch.Tensor, thumbnail_width: int):
    from PIL import Image

    value = frame.detach().cpu().float().clamp(0, 1)
    if int(value.shape[-1]) > 3:
        value = value[..., :3]
    if int(value.shape[-1]) < 3:
        value = value[..., :1].repeat(1, 1, 3)
    height = int(value.shape[0])
    width = int(value.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("Cannot build contact sheet from an empty frame.")
    array = (value.numpy() * 255.0).round().astype("uint8")
    image = Image.fromarray(array, mode="RGB")
    thumb_w = max(96, int(thumbnail_width))
    thumb_h = max(1, int(round(float(height) * float(thumb_w) / float(width))))
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    return image.resize((thumb_w, thumb_h), resample)


def _build_keyframe_contact_sheet(
    boundary_anchor_frames: torch.Tensor,
    new_chunk_start_frames: torch.Tensor,
    rows: list[dict[str, Any]],
    include_final_anchor: bool,
    columns: int,
    thumbnail_width: int,
) -> torch.Tensor:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    chunk_count = len(rows)
    if chunk_count <= 0:
        raise ValueError("Cannot build contact sheet without chunk rows.")
    columns = max(1, min(12, int(columns)))
    label_h = 46
    padding = 8
    font = ImageFont.load_default()

    sample = _tensor_frame_to_pil(boundary_anchor_frames[0], thumbnail_width)
    thumb_w, thumb_h = sample.size
    cell_w = thumb_w + padding * 2
    cell_h = thumb_h + label_h + padding * 2
    group_count = (chunk_count + columns - 1) // columns
    extra_final_row = 1 if include_final_anchor and int(boundary_anchor_frames.shape[0]) > chunk_count else 0
    sheet_w = cell_w * columns
    sheet_h = cell_h * (group_count * 2 + extra_final_row)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)

    def draw_cell(frame: torch.Tensor, label_lines: list[str], col: int, row: int) -> None:
        x = col * cell_w
        y = row * cell_h
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(202, 207, 214), width=1)
        image = _tensor_frame_to_pil(frame, thumb_w)
        sheet.paste(image, (x + padding, y + padding))
        text_y = y + padding + thumb_h + 5
        for line in label_lines[:3]:
            draw.text((x + padding, text_y), line, fill=(20, 24, 31), font=font)
            text_y += 13

    for index, row in enumerate(rows):
        group = index // columns
        col = index % columns
        boundary_row = group * 2
        start_row = boundary_row + 1
        chunk_number = int(row["chunk_index"]) + 1
        output_range = row["output_range_1_based_inclusive"]
        boundary_source = str(row.get("boundary_anchor_source", "overlap_last")).replace("_", " ")
        draw_cell(
            boundary_anchor_frames[index],
            [
                f"chunk {chunk_number} {boundary_source}",
                f"frame {int(row['boundary_anchor_frame_1_based'])}",
                f"out {output_range[0]}-{output_range[1]}",
            ],
            col,
            boundary_row,
        )
        draw_cell(
            new_chunk_start_frames[index],
            [
                f"chunk {chunk_number} new start",
                f"frame {int(row['new_chunk_start_frame_1_based'])}",
                f"keep {int(row['keep_frames'])}",
            ],
            col,
            start_row,
        )

    if extra_final_row:
        final_frame_number = int(rows[-1]["output_range_1_based_inclusive"][1])
        draw_cell(
            boundary_anchor_frames[-1],
            ["final anchor", f"frame {final_frame_number}", "end of plan"],
            0,
            group_count * 2,
        )

    array = torch.from_numpy(np.array(sheet).astype("float32") / 255.0)
    return array.unsqueeze(0).contiguous()


def _build_paired_keyframes(
    boundary_anchor_frames: torch.Tensor,
    new_chunk_start_frames: torch.Tensor,
    rows: list[dict[str, Any]],
    include_final_anchor: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    frames: list[torch.Tensor] = []
    manifest: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        output_range = row["output_range_1_based_inclusive"]
        frames.append(boundary_anchor_frames[index : index + 1])
        manifest.append(
            {
                "batch_index": len(manifest),
                "chunk_index": int(row["chunk_index"]),
                "chunk_number_1_based": int(row["chunk_index"]) + 1,
                "kind": "boundary_anchor",
                "display_kind": str(row.get("boundary_anchor_source", "boundary_anchor")),
                "frame_0_based": int(row["boundary_anchor_frame_0_based"]),
                "frame_1_based": int(row["boundary_anchor_frame_1_based"]),
                "output_range_1_based_inclusive": output_range,
            }
        )
        frames.append(new_chunk_start_frames[index : index + 1])
        manifest.append(
            {
                "batch_index": len(manifest),
                "chunk_index": int(row["chunk_index"]),
                "chunk_number_1_based": int(row["chunk_index"]) + 1,
                "kind": "new_chunk_start",
                "frame_0_based": int(row["new_chunk_start_frame_0_based"]),
                "frame_1_based": int(row["new_chunk_start_frame_1_based"]),
                "output_range_1_based_inclusive": output_range,
            }
        )
    if include_final_anchor and int(boundary_anchor_frames.shape[0]) > len(rows):
        final_index = int(boundary_anchor_frames.shape[0]) - 1
        final_frame = int(rows[-1]["output_range_1_based_inclusive"][1])
        frames.append(boundary_anchor_frames[final_index : final_index + 1])
        manifest.append(
            {
                "batch_index": len(manifest),
                "chunk_index": int(rows[-1]["chunk_index"]),
                "chunk_number_1_based": int(rows[-1]["chunk_index"]) + 1,
                "kind": "final_anchor",
                "frame_0_based": final_frame - 1,
                "frame_1_based": final_frame,
                "output_range_1_based_inclusive": rows[-1]["output_range_1_based_inclusive"],
            }
        )
    if not frames:
        raise ValueError("paired_keyframes produced no frames.")
    return torch.cat(frames, dim=0).contiguous(), manifest


def _matrix_safe_filename_part(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "_").replace("/", "_")
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return safe.strip("._") or "keyframe"


def _save_keyframe_matrix_images(
    images: torch.Tensor,
    summary: str,
    filename_prefix: str,
    save_location: str,
    display_group: str = "both",
) -> dict[str, Any]:
    import os

    import numpy as np
    from PIL import Image

    try:
        import folder_paths
    except Exception as exc:
        raise RuntimeError("ComfyUI folder_paths is required to save keyframe matrix images.") from exc

    if not isinstance(images, torch.Tensor) or images.ndim != 4 or int(images.shape[0]) <= 0:
        raise ValueError("paired_keyframes must be a non-empty ComfyUI IMAGE batch.")
    try:
        parsed = json.loads(str(summary or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"summary must be JSON from SCAIL2ChunkKeyframeExtractor: {exc}") from exc
    manifest = parsed.get("paired_keyframes_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("summary must contain paired_keyframes_manifest from SCAIL2ChunkKeyframeExtractor.")
    if len(manifest) != int(images.shape[0]):
        raise ValueError(
            f"paired_keyframes count ({int(images.shape[0])}) does not match manifest count ({len(manifest)})."
        )
    normalized_display_group = str(display_group or "both").strip() or "both"
    allowed_kinds_by_group = {
        "both": None,
        "overlap_boundary_only": {"boundary_anchor", "final_anchor"},
        "new_chunk_start_only": {"new_chunk_start"},
    }
    if normalized_display_group not in allowed_kinds_by_group:
        normalized_display_group = "both"
    allowed_kinds = allowed_kinds_by_group[normalized_display_group]

    location = "output" if str(save_location or "").strip() == "output" else "temp"
    base_dir = folder_paths.get_output_directory() if location == "output" else folder_paths.get_temp_directory()
    subfolder = "scail_keyframe_matrix"
    target_dir = os.path.join(base_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    prefix = _matrix_safe_filename_part(filename_prefix or "scail_keyframe")
    fingerprint = _stable_fingerprint(
        {
            "prefix": prefix,
            "shape": _shape(images),
            "manifest": manifest,
        }
    )[:8]

    items: list[dict[str, Any]] = []
    for index, row in enumerate(manifest):
        row_kind = str(row.get("kind", "keyframe"))
        if allowed_kinds is not None and row_kind not in allowed_kinds:
            continue
        frame = images[index].detach().cpu().float().clamp(0, 1)
        if int(frame.shape[-1]) > 3:
            frame = frame[..., :3]
        if int(frame.shape[-1]) < 3:
            frame = frame[..., :1].repeat(1, 1, 3)
        array = (frame.numpy() * 255.0).round().astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")

        kind = _matrix_safe_filename_part(row.get("kind", "keyframe"))
        chunk_index = int(row.get("chunk_index", 0))
        chunk_number = int(row.get("chunk_number_1_based", chunk_index + 1))
        frame_number = int(row.get("frame_1_based", index + 1))
        filename = f"{prefix}_{fingerprint}_{index:03d}_chunk{chunk_index}_{kind}_frame{frame_number}.png"
        image.save(os.path.join(target_dir, filename))
        item = {
            "batch_index": int(index),
            "chunk_index": chunk_index,
            "chunk_number_1_based": chunk_number,
            "kind": row_kind,
            "display_kind": str(row.get("display_kind", row_kind)),
            "frame_1_based": frame_number,
            "frame_0_based": int(row.get("frame_0_based", frame_number - 1)),
            "output_range_1_based_inclusive": row.get("output_range_1_based_inclusive"),
            "filename": filename,
            "subfolder": subfolder,
            "type": location,
        }
        item["label"] = f"{index:03d} | chunk {chunk_number} | {item['display_kind']} | frame {frame_number}"
        items.append(item)

    return {
        "items": items,
        "count": len(items),
        "type": location,
        "subfolder": subfolder,
        "filename_prefix": prefix,
        "display_group": normalized_display_group,
    }


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


def _fit_image_to_size(
    image: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    fit_mode: str = "center_crop",
    mode: str = "bilinear",
) -> torch.Tensor:
    import torch.nn.functional as F

    target_h = max(1, int(target_h))
    target_w = max(1, int(target_w))
    if int(image.shape[1]) == target_h and int(image.shape[2]) == target_w:
        return image.contiguous()
    normalized = str(fit_mode or "center_crop").strip()
    if normalized not in {"center_crop", "pad", "stretch"}:
        normalized = "center_crop"
    if normalized == "stretch":
        return _resize_image_tensor_like(image, torch.empty((1, target_h, target_w, int(image.shape[-1]))), mode=mode)

    src_h = max(1, int(image.shape[1]))
    src_w = max(1, int(image.shape[2]))
    scale = max(target_h / src_h, target_w / src_w) if normalized == "center_crop" else min(target_h / src_h, target_w / src_w)
    resize_h = max(1, int(round(src_h * scale)))
    resize_w = max(1, int(round(src_w * scale)))
    kwargs = {"align_corners": False} if mode in {"bilinear", "bicubic"} else {}
    resized = F.interpolate(
        image.detach().float().movedim(-1, 1),
        size=(resize_h, resize_w),
        mode=mode,
        **kwargs,
    ).movedim(1, -1)
    if int(resized.shape[1]) > target_h:
        y0 = (int(resized.shape[1]) - target_h) // 2
        resized = resized[:, y0 : y0 + target_h, :, :]
        resize_h = target_h
    if int(resized.shape[2]) > target_w:
        x0 = (int(resized.shape[2]) - target_w) // 2
        resized = resized[:, :, x0 : x0 + target_w, :]
        resize_w = target_w
    if normalized == "center_crop":
        y0 = max(0, (resize_h - target_h) // 2)
        x0 = max(0, (resize_w - target_w) // 2)
        return resized[:, y0 : y0 + target_h, x0 : x0 + target_w, :].to(dtype=image.dtype).contiguous()

    fitted = torch.zeros((int(image.shape[0]), target_h, target_w, int(image.shape[-1])), dtype=resized.dtype, device=resized.device)
    y0 = max(0, (target_h - resize_h) // 2)
    x0 = max(0, (target_w - resize_w) // 2)
    fitted[:, y0 : y0 + resize_h, x0 : x0 + resize_w, :] = resized
    return fitted.to(dtype=image.dtype).contiguous()


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


def _normalize_video_mask(mask: torch.Tensor, frame_count: int, height: int, width: int, name: str = "mask") -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI MASK tensor.")
    value = mask.detach().cpu().float()
    if value.ndim == 4:
        value = value.amax(dim=-1)
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,H,W] or [B,H,W,C].")
    if int(value.shape[0]) == 1 and int(frame_count) > 1:
        value = value.repeat(int(frame_count), 1, 1)
    if int(value.shape[0]) != int(frame_count):
        raise ValueError(f"{name} frame count {int(value.shape[0])} does not match video frame count {frame_count}.")
    if int(value.shape[1]) != int(height) or int(value.shape[2]) != int(width):
        import torch.nn.functional as F

        value = F.interpolate(
            value.unsqueeze(1),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return value.clamp(0, 1).contiguous()


def _resize_mask_batch(mask: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    import torch.nn.functional as F

    value = mask.detach().float()
    if value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim == 4:
        value = value.movedim(-1, 1)
    else:
        raise ValueError("mask must have shape [B,H,W] or [B,H,W,C].")
    kwargs = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(value, size=(int(height), int(width)), **kwargs).squeeze(1).clamp(0, 1).contiguous()


def _binary_mask_morph(mask: torch.Tensor, expand_px: int = 0, contract_px: int = 0, blur_px: int = 0) -> torch.Tensor:
    import torch.nn.functional as F

    value = mask.detach().float().clamp(0, 1)
    if value.ndim == 3:
        value = value.unsqueeze(1)
    elif value.ndim == 4:
        value = value.movedim(-1, 1)
    else:
        raise ValueError("mask must have shape [B,H,W] or [B,H,W,C].")
    value = (value > 0.05).float()
    if int(expand_px) > 0:
        kernel = int(expand_px) * 2 + 1
        value = F.max_pool2d(value, kernel_size=kernel, stride=1, padding=int(expand_px))
    if int(contract_px) > 0:
        kernel = int(contract_px) * 2 + 1
        value = 1.0 - F.max_pool2d(1.0 - value, kernel_size=kernel, stride=1, padding=int(contract_px))
    if int(blur_px) > 0:
        kernel = int(blur_px) * 2 + 1
        value = F.avg_pool2d(value, kernel_size=kernel, stride=1, padding=int(blur_px))
    return value.squeeze(1).clamp(0, 1).contiguous()


def _bbox_from_mask_frame(mask: torch.Tensor) -> Optional[tuple[int, int, int, int]]:
    coords = torch.nonzero(mask > 0.05, as_tuple=False)
    if int(coords.shape[0]) <= 0:
        return None
    y0 = int(coords[:, 0].min().item())
    y1 = int(coords[:, 0].max().item()) + 1
    x0 = int(coords[:, 1].min().item())
    x1 = int(coords[:, 1].max().item()) + 1
    return x0, y0, x1, y1


def _largest_component_mask_numpy(binary):
    import numpy as np

    if int(binary.sum()) <= 0:
        return binary.astype(bool), {"components": 0, "largest_area": 0, "removed_area": 0}

    try:
        import cv2

        component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary.astype("uint8"), connectivity=8)
        if int(component_count) <= 1:
            area = int(binary.sum())
            return binary.astype(bool), {"components": 1, "largest_area": area, "removed_area": 0}
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        largest_area = int(areas[largest_label - 1])
        total_area = int(binary.sum())
        return labels == largest_label, {
            "components": int(component_count) - 1,
            "largest_area": largest_area,
            "removed_area": max(0, total_area - largest_area),
        }
    except Exception:
        pass

    try:
        from scipy import ndimage

        labels, component_count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        if int(component_count) <= 1:
            area = int(binary.sum())
            return binary.astype(bool), {"components": int(component_count), "largest_area": area, "removed_area": 0}
        areas = np.bincount(labels.reshape(-1))[1:]
        largest_label = int(np.argmax(areas)) + 1
        largest_area = int(areas[largest_label - 1])
        total_area = int(binary.sum())
        return labels == largest_label, {
            "components": int(component_count),
            "largest_area": largest_area,
            "removed_area": max(0, total_area - largest_area),
        }
    except Exception:
        pass

    from collections import deque

    binary_bool = binary.astype(bool)
    height, width = binary_bool.shape
    visited = np.zeros_like(binary_bool, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    component_count = 0
    for start_y, start_x in np.argwhere(binary_bool):
        start_y = int(start_y)
        start_x = int(start_x)
        if visited[start_y, start_x]:
            continue
        component_count += 1
        pixels: list[tuple[int, int]] = []
        queue = deque([(start_y, start_x)])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if visited[ny, nx] or not binary_bool[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(pixels) > len(best_pixels):
            best_pixels = pixels

    keep = np.zeros_like(binary_bool, dtype=bool)
    for y, x in best_pixels:
        keep[y, x] = True
    total_area = int(binary_bool.sum())
    largest_area = int(len(best_pixels))
    return keep, {
        "components": int(component_count),
        "largest_area": largest_area,
        "removed_area": max(0, total_area - largest_area),
    }


def _keep_largest_mask_components(mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    import numpy as np

    value = mask.detach().cpu().float().clamp(0, 1)
    if value.ndim != 3:
        raise ValueError("mask must have shape [B,H,W].")
    kept_frames: list[torch.Tensor] = []
    frame_stats: list[dict[str, int]] = []
    for index in range(int(value.shape[0])):
        frame = value[index]
        binary = (frame > 0.05).numpy().astype(np.uint8)
        keep, stats = _largest_component_mask_numpy(binary)
        keep_tensor = torch.from_numpy(keep).to(dtype=frame.dtype)
        kept_frames.append((frame * keep_tensor).clamp(0, 1))
        frame_stats.append({"frame": int(index), **stats})

    frames_with_components = [item for item in frame_stats if int(item["components"]) > 0]
    frames_with_removed = [item for item in frame_stats if int(item["removed_area"]) > 0]
    largest_areas = [int(item["largest_area"]) for item in frames_with_components]
    summary = {
        "frames": int(value.shape[0]),
        "frames_with_components": int(len(frames_with_components)),
        "frames_with_removed_components": int(len(frames_with_removed)),
        "max_components": max((int(item["components"]) for item in frame_stats), default=0),
        "removed_pixels_total": int(sum(int(item["removed_area"]) for item in frame_stats)),
        "largest_pixels_min": int(min(largest_areas)) if largest_areas else 0,
        "largest_pixels_max": int(max(largest_areas)) if largest_areas else 0,
        "sample_removed_frames": frames_with_removed[:12],
    }
    return torch.stack(kept_frames, dim=0).contiguous(), summary


def _image_batch_frame_to_rgb_uint8(image: torch.Tensor, index: int, name: str) -> Any:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    frame_count = int(image.shape[0])
    if frame_count <= 0:
        raise ValueError(f"{name} has no frames.")
    frame_index = max(0, min(frame_count - 1, int(index)))
    frame = image[frame_index].detach().cpu().float().clamp(0, 1)
    if int(frame.shape[-1]) > 3:
        frame = frame[..., :3]
    if int(frame.shape[-1]) < 3:
        frame = frame[..., :1].repeat(1, 1, 3)
    return (frame.numpy() * 255.0).round().astype("uint8")


def _rgb_uint8_to_image_tensor(image: Any) -> torch.Tensor:
    import numpy as np

    value = np.asarray(image)
    if value.ndim != 3 or int(value.shape[-1]) < 3:
        raise ValueError("image array must have shape [H,W,3].")
    value = value[:, :, :3].astype("float32") / 255.0
    return torch.from_numpy(value).unsqueeze(0).contiguous().clamp(0, 1)


def _insightface_providers(provider_mode: str) -> tuple[str, list[str], int]:
    normalized = str(provider_mode or "auto").strip().lower()
    available: list[str] = []
    try:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
    except Exception:
        available = []

    if normalized == "cpu":
        return "cpu", ["CPUExecutionProvider"], -1
    if normalized == "cuda":
        if not available or "CUDAExecutionProvider" in available:
            return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
        raise RuntimeError(
            "onnxruntime does not expose CUDAExecutionProvider. Install onnxruntime-gpu or set provider to cpu."
        )
    if "CUDAExecutionProvider" in available:
        return "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    return "cpu", ["CPUExecutionProvider"], -1


def _get_insightface_app(model_name: str, provider_mode: str, det_size: int):
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        raise RuntimeError(
            "SCAIL-2 Align Reference Face To Crop requires InsightFace. Install insightface and "
            "onnxruntime-gpu (or onnxruntime for CPU), and make sure the InsightFace model such as "
            "buffalo_l is available."
        ) from exc

    provider_key, providers, ctx_id = _insightface_providers(provider_mode)
    normalized_model = str(model_name or "buffalo_l").strip() or "buffalo_l"
    normalized_det_size = max(160, min(2048, int(det_size)))
    cache_key = (normalized_model, provider_key, normalized_det_size)
    app = _INSIGHTFACE_APP_CACHE.get(cache_key)
    if app is None:
        app = FaceAnalysis(name=normalized_model, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(normalized_det_size, normalized_det_size))
        _INSIGHTFACE_APP_CACHE[cache_key] = app
    return app, {"model_name": normalized_model, "provider": provider_key, "det_size": int(normalized_det_size)}


def _select_face_info(
    candidates: list[dict[str, Any]],
    image_w: int,
    image_h: int,
    select_mode: str,
    name: str,
    backend: str,
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError(f"{backend} detected no face in {name}. Use a clearer frame/reference image.")
    normalized = str(select_mode or "largest").strip()
    if normalized not in {"largest", "center"}:
        normalized = "largest"

    if normalized == "center":
        cx = float(image_w) / 2.0
        cy = float(image_h) / 2.0
        return min(
            candidates,
            key=lambda item: (float(item["center"][0]) - cx) ** 2 + (float(item["center"][1]) - cy) ** 2,
        )
    return max(candidates, key=lambda item: float(item["size"][0]) * float(item["size"][1]))


def _face_bbox_info(face: Any, image_w: int, image_h: int) -> dict[str, Any]:
    bbox = getattr(face, "bbox", None)
    if bbox is None or len(bbox) != 4:
        raise RuntimeError("InsightFace face did not include a bbox.")
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 = max(0.0, min(float(image_w), x0))
    y0 = max(0.0, min(float(image_h), y0))
    x1 = max(x0 + 1.0, min(float(image_w), x1))
    y1 = max(y0 + 1.0, min(float(image_h), y1))
    score = getattr(face, "det_score", None)
    return {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "bbox_int": [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))],
        "center": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
        "size": [float(x1 - x0), float(y1 - y0)],
        "score": float(score) if score is not None else None,
    }


def _detect_face_info_insightface(
    image_rgb: Any,
    select_mode: str,
    name: str,
    insightface_model: str,
    provider: str,
    det_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_h, image_w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    app, model_info = _get_insightface_app(str(insightface_model), str(provider), int(det_size))
    faces = list(app.get(image_rgb[:, :, ::-1].copy()))
    candidates = [_face_bbox_info(face, int(image_w), int(image_h)) for face in faces]
    selected = _select_face_info(candidates, int(image_w), int(image_h), str(select_mode), str(name), "InsightFace")
    return selected, {"backend": "insightface", **model_info, "candidate_count": int(len(candidates))}


def _get_mediapipe_face_detector(model_selection: str, min_detection_confidence: float):
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe fallback requires mediapipe. Install it with: python -m pip install mediapipe"
        ) from exc

    normalized = str(model_selection or "full_range").strip()
    model_selection_id = 0 if normalized == "short_range" else 1
    confidence = max(0.01, min(0.99, float(min_detection_confidence)))
    cache_key = (int(model_selection_id), round(confidence, 4))
    detector = _MEDIAPIPE_FACE_DETECTION_CACHE.get(cache_key)
    if detector is None:
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=int(model_selection_id),
            min_detection_confidence=float(confidence),
        )
        _MEDIAPIPE_FACE_DETECTION_CACHE[cache_key] = detector
    return detector, {
        "backend": "mediapipe",
        "model_selection": "short_range" if model_selection_id == 0 else "full_range",
        "min_detection_confidence": float(confidence),
    }


def _detect_face_info_mediapipe(
    image_rgb: Any,
    select_mode: str,
    name: str,
    model_selection: str,
    min_detection_confidence: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_h, image_w = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    detector, model_info = _get_mediapipe_face_detector(str(model_selection), float(min_detection_confidence))
    result = detector.process(image_rgb)
    detections = list(getattr(result, "detections", None) or [])
    candidates: list[dict[str, Any]] = []
    for detection in detections:
        location_data = getattr(detection, "location_data", None)
        relative_bbox = getattr(location_data, "relative_bounding_box", None)
        if relative_bbox is None:
            continue
        x0 = float(relative_bbox.xmin) * float(image_w)
        y0 = float(relative_bbox.ymin) * float(image_h)
        x1 = x0 + float(relative_bbox.width) * float(image_w)
        y1 = y0 + float(relative_bbox.height) * float(image_h)
        x0 = max(0.0, min(float(image_w), x0))
        y0 = max(0.0, min(float(image_h), y0))
        x1 = max(x0 + 1.0, min(float(image_w), x1))
        y1 = max(y0 + 1.0, min(float(image_h), y1))
        score_values = list(getattr(detection, "score", None) or [])
        keypoints: list[list[float]] = []
        for keypoint in list(getattr(location_data, "relative_keypoints", None) or []):
            keypoints.append([float(keypoint.x) * float(image_w), float(keypoint.y) * float(image_h)])
        candidates.append(
            {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "bbox_int": [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))],
                "center": [float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)],
                "size": [float(x1 - x0), float(y1 - y0)],
                "score": float(score_values[0]) if score_values else None,
                "keypoints": keypoints,
            }
        )
    selected = _select_face_info(candidates, int(image_w), int(image_h), str(select_mode), str(name), "MediaPipe")
    return selected, {**model_info, "candidate_count": int(len(candidates))}


def _detect_face_pair(
    target_rgb: Any,
    reference_rgb: Any,
    backend: str,
    target_select: str,
    reference_select: str,
    insightface_model: str,
    provider: str,
    det_size: int,
    mediapipe_model_selection: str,
    mediapipe_min_detection_confidence: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    requested_backend = str(backend or "auto").strip()
    if requested_backend not in {"auto", "insightface", "mediapipe"}:
        requested_backend = "auto"

    def run_insightface() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        target_info, target_model = _detect_face_info_insightface(
            target_rgb,
            str(target_select),
            "face_crop_video target frame",
            str(insightface_model),
            str(provider),
            int(det_size),
        )
        reference_info, reference_model = _detect_face_info_insightface(
            reference_rgb,
            str(reference_select),
            "reference_image",
            str(insightface_model),
            str(provider),
            int(det_size),
        )
        return target_info, reference_info, {
            "requested_backend": requested_backend,
            "backend_used": "insightface",
            "target": target_model,
            "reference": reference_model,
        }

    def run_mediapipe(auto_fallback_reason: Optional[str] = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        target_info, target_model = _detect_face_info_mediapipe(
            target_rgb,
            str(target_select),
            "face_crop_video target frame",
            str(mediapipe_model_selection),
            float(mediapipe_min_detection_confidence),
        )
        reference_info, reference_model = _detect_face_info_mediapipe(
            reference_rgb,
            str(reference_select),
            "reference_image",
            str(mediapipe_model_selection),
            float(mediapipe_min_detection_confidence),
        )
        detector_info = {
            "requested_backend": requested_backend,
            "backend_used": "mediapipe",
            "target": target_model,
            "reference": reference_model,
        }
        if auto_fallback_reason:
            detector_info["auto_fallback_from_insightface"] = auto_fallback_reason
        return target_info, reference_info, detector_info

    if requested_backend == "insightface":
        return run_insightface()
    if requested_backend == "mediapipe":
        return run_mediapipe()

    try:
        return run_insightface()
    except Exception as insightface_exc:
        try:
            return run_mediapipe(str(insightface_exc))
        except Exception as mediapipe_exc:
            raise RuntimeError(
                "Auto face detector failed with both InsightFace and MediaPipe. "
                f"InsightFace error: {insightface_exc}; MediaPipe error: {mediapipe_exc}"
            ) from mediapipe_exc


def _extract_reference_window_no_resize(
    image_rgb: Any,
    x0: int,
    y0: int,
    canvas_w: int,
    canvas_h: int,
    padding_mode: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    x1 = int(x0) + canvas_w
    y1 = int(y0) + canvas_h
    crop_x0 = max(0, int(x0))
    crop_y0 = max(0, int(y0))
    crop_x1 = min(width, int(x1))
    crop_y1 = min(height, int(y1))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise RuntimeError("reference crop window did not overlap the reference image.")
    crop = image_rgb[crop_y0:crop_y1, crop_x0:crop_x1, :3]
    pad_left = max(0, -int(x0))
    pad_top = max(0, -int(y0))
    pad_right = max(0, int(x1) - width)
    pad_bottom = max(0, int(y1) - height)
    normalized = str(padding_mode or "edge").strip()
    if normalized not in {"edge", "reflect", "black", "white", "mean"}:
        normalized = "edge"

    if pad_left or pad_top or pad_right or pad_bottom:
        if normalized in {"black", "white", "mean"}:
            if normalized == "white":
                fill = np.array([255, 255, 255], dtype=np.uint8)
            elif normalized == "mean":
                fill = image_rgb[:, :, :3].reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
            else:
                fill = np.array([0, 0, 0], dtype=np.uint8)
            output = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
            output[:, :, :] = fill.reshape(1, 1, 3)
            output[pad_top : pad_top + int(crop.shape[0]), pad_left : pad_left + int(crop.shape[1]), :] = crop
        else:
            try:
                np_mode = "edge" if normalized == "edge" else "reflect"
                output = np.pad(
                    crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode=np_mode,
                )
            except Exception:
                output = np.pad(
                    crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode="edge",
                )
    else:
        output = crop
    output = output[:canvas_h, :canvas_w, :3]
    if int(output.shape[0]) != canvas_h or int(output.shape[1]) != canvas_w:
        fixed = np.empty((canvas_h, canvas_w, 3), dtype=np.uint8)
        fixed[:, :, :] = output[-1:, -1:, :]
        fixed[: int(output.shape[0]), : int(output.shape[1]), :] = output
        output = fixed
    return output.astype(np.uint8), {
        "window_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "source_overlap_xyxy": [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)],
        "padding": {
            "left": int(pad_left),
            "top": int(pad_top),
            "right": int(pad_right),
            "bottom": int(pad_bottom),
            "mode": normalized,
        },
    }


def _clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = max(1, int(width))
    height = max(1, int(height))
    x0 = max(0, min(width - 1, int(x0)))
    y0 = max(0, min(height - 1, int(y0)))
    x1 = max(x0 + 1, min(width, int(x1)))
    y1 = max(y0 + 1, min(height, int(y1)))
    return x0, y0, x1, y1


def _offset_bbox_within_bounds(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clamp_bbox(bbox, int(width), int(height))
    box_w = max(1, int(x1 - x0))
    box_h = max(1, int(y1 - y0))
    max_x0 = max(0, int(width) - box_w)
    max_y0 = max(0, int(height) - box_h)
    shifted_x0 = max(0, min(max_x0, int(x0) + int(offset_x)))
    shifted_y0 = max(0, min(max_y0, int(y0) + int(offset_y)))
    return shifted_x0, shifted_y0, shifted_x0 + box_w, shifted_y0 + box_h


def _draw_rect(image: torch.Tensor, bbox: tuple[int, int, int, int], color: tuple[float, float, float]) -> None:
    _batch, height, width, _channels = image.shape
    x0, y0, x1, y1 = _clamp_bbox(bbox, int(width), int(height))
    thickness = max(1, min(int(width), int(height)) // 160)
    rgb = torch.tensor(color, dtype=image.dtype, device=image.device).view(1, 1, 3)
    image[:, y0 : min(y0 + thickness, y1), x0:x1, :3] = rgb
    image[:, max(y1 - thickness, y0) : y1, x0:x1, :3] = rgb
    image[:, y0:y1, x0 : min(x0 + thickness, x1), :3] = rgb
    image[:, y0:y1, max(x1 - thickness, x0) : x1, :3] = rgb


def _interpolate_missing_bboxes(raw: list[Optional[tuple[int, int, int, int]]]) -> list[tuple[int, int, int, int]]:
    if not raw:
        return []
    valid = [index for index, bbox in enumerate(raw) if bbox is not None]
    if not valid:
        raise ValueError("head mask did not contain any detectable head region.")
    out: list[tuple[int, int, int, int]] = [raw[valid[0]]] * len(raw)  # type: ignore[list-item]
    for left_pos, left_index in enumerate(valid):
        left_bbox = raw[left_index]
        next_index = valid[left_pos + 1] if left_pos + 1 < len(valid) else None
        if left_bbox is None:
            continue
        out[left_index] = left_bbox
        end = next_index if next_index is not None else len(raw)
        if next_index is None or raw[next_index] is None:
            for index in range(left_index + 1, end):
                out[index] = left_bbox
            continue
        right_bbox = raw[next_index]
        gap = max(1, next_index - left_index)
        for index in range(left_index + 1, next_index):
            alpha = (index - left_index) / gap
            out[index] = tuple(
                int(round(float(left_bbox[channel]) * (1.0 - alpha) + float(right_bbox[channel]) * alpha))
                for channel in range(4)
            )
    for index in range(0, valid[0]):
        out[index] = raw[valid[0]]  # type: ignore[assignment]
    return out


def _smooth_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    smoothing: float,
) -> list[tuple[int, int, int, int]]:
    if not bboxes:
        return []
    alpha = max(0.0, min(0.98, float(smoothing)))
    previous = [float(value) for value in bboxes[0]]
    smoothed: list[tuple[int, int, int, int]] = []
    for bbox in bboxes:
        current = [float(value) for value in bbox]
        previous = [previous[index] * alpha + current[index] * (1.0 - alpha) for index in range(4)]
        smoothed.append(_clamp_bbox(tuple(int(round(value)) for value in previous), width, height))
    return smoothed


def _max_aligned_square_side(width: int, height: int, align: int) -> int:
    max_side = max(1, min(int(width), int(height)))
    align = max(1, int(align))
    if align <= 1 or max_side < align:
        return max_side
    return max(align, (max_side // align) * align)


def _square_bbox_from_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
    align: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clamp_bbox(bbox, width, height)
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    side = int(math.ceil(max(box_w, box_h) * (1.0 + max(0.0, float(padding_ratio)) * 2.0)))
    align = max(1, int(align))
    side = max(align, int(math.ceil(side / align) * align))
    side = min(side, _max_aligned_square_side(int(width), int(height), align))
    return _fixed_square_bbox_from_bbox(bbox, width, height, side)


def _union_bbox_from_bboxes(
    bboxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if not bboxes:
        raise ValueError("cannot build a fixed canvas without at least one bbox.")
    x0 = min(int(bbox[0]) for bbox in bboxes)
    y0 = min(int(bbox[1]) for bbox in bboxes)
    x1 = max(int(bbox[2]) for bbox in bboxes)
    y1 = max(int(bbox[3]) for bbox in bboxes)
    return _clamp_bbox((x0, y0, x1, y1), int(width), int(height))


def _fixed_square_bbox_from_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    side: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clamp_bbox(bbox, width, height)
    side = max(1, min(int(side), int(width), int(height)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    sx0 = int(round(cx - side / 2.0))
    sy0 = int(round(cy - side / 2.0))
    sx0 = max(0, min(int(width) - side, sx0))
    sy0 = max(0, min(int(height) - side, sy0))
    sx1 = sx0 + side
    sy1 = sy0 + side
    return _clamp_bbox((sx0, sy0, sx1, sy1), width, height)


def _crop_video_by_manifest(
    video: torch.Tensor,
    masks: torch.Tensor,
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    _frame_count, height, width, channels = video.shape
    if not bboxes:
        raise ValueError("crop manifest requires at least one bbox.")
    crop_side = int(bboxes[0][2] - bboxes[0][0])
    if crop_side <= 0 or int(bboxes[0][3] - bboxes[0][1]) != crop_side:
        raise ValueError("crop manifest bboxes must be square.")
    crops: list[torch.Tensor] = []
    crop_masks: list[torch.Tensor] = []
    frames: list[dict[str, Any]] = []
    for index, bbox in enumerate(bboxes):
        x0, y0, x1, y1 = bbox
        if int(x1 - x0) != crop_side or int(y1 - y0) != crop_side:
            raise ValueError("all crop manifest bboxes must have the same fixed square size.")
        crop = video[index : index + 1, y0:y1, x0:x1, :]
        crop_mask = masks[index : index + 1, y0:y1, x0:x1]
        crops.append(crop.cpu())
        crop_masks.append(crop_mask.cpu())
        frames.append(
            {
                "frame": int(index),
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "crop_to_canvas_bbox": [int(x0), int(y0), int(x1), int(y1)],
                "canvas_to_original_bbox": [0, 0, int(width), int(height)],
                "crop_size": int(crop_side),
                "original_crop_size": [int(crop_side), int(crop_side)],
                "source_size": [int(width), int(height)],
                "channels": int(channels),
            }
        )
    return torch.cat(crops, dim=0).clamp(0, 1), torch.cat(crop_masks, dim=0).clamp(0, 1), frames


def _parse_crop_manifest(crop_manifest: str) -> dict[str, Any]:
    try:
        manifest = json.loads(str(crop_manifest or "").strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"crop_manifest must be valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("frames"), list):
        raise ValueError("crop_manifest must contain a frames array.")
    return manifest


def _local_mean_std_color_match(source: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    weight = mask.clamp(0, 1).to(dtype=torch.float32)
    if weight.ndim == 3:
        weight = weight.unsqueeze(-1)
    total = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1e-4)
    src = source[:, :, :, :3].to(torch.float32)
    dst = target[:, :, :, :3].to(torch.float32)
    src_mean = (src * weight).sum(dim=(1, 2), keepdim=True) / total
    dst_mean = (dst * weight).sum(dim=(1, 2), keepdim=True) / total
    src_var = (((src - src_mean) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / total
    dst_var = (((dst - dst_mean) ** 2) * weight).sum(dim=(1, 2), keepdim=True) / total
    scale = (dst_var.sqrt() / src_var.sqrt().clamp_min(1e-4)).clamp(0.82, 1.22)
    shift = (dst_mean - src_mean * scale).clamp(-0.12, 0.12)
    corrected = source.clone()
    corrected[:, :, :, :3] = (src * scale + shift).clamp(0, 1).to(dtype=source.dtype)
    return corrected, {"method": "local_mean_std", "applied": True}


def _draw_manifest_debug(video: torch.Tensor, frames: list[dict[str, Any]], max_frames: int = 12) -> torch.Tensor:
    count = min(int(max_frames), int(video.shape[0]), len(frames))
    previews: list[torch.Tensor] = []
    for index in range(count):
        preview = video[index : index + 1, :, :, :3].detach().cpu().clone()
        bbox = frames[index].get(
            "crop_to_canvas_bbox",
            frames[index].get("bbox", [0, 0, int(video.shape[2]), int(video.shape[1])]),
        )
        _draw_rect(preview, tuple(int(value) for value in bbox), (0.1, 0.9, 0.25))
        previews.append(preview)
    return torch.cat(previews, dim=0).contiguous() if previews else video[:1, :, :, :3].detach().cpu().contiguous()


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


class SCAIL2ChunkKeyframeExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "mode": (["planner_summary", "standard_long_video"], {"default": "planner_summary"}),
                "max_chunk_frames": ("INT", {"default": 81, "min": 17, "max": 81, "step": 4}),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 33, "step": 1}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "include_final_anchor": ("BOOLEAN", {"default": False}),
                "contact_sheet_columns": ("INT", {"default": 4, "min": 1, "max": 12, "step": 1}),
                "contact_sheet_thumbnail_width": ("INT", {"default": 256, "min": 96, "max": 1024, "step": 16}),
                "boundary_anchor_mode": (
                    ["overlap_last_frame", "overlap_first_frame", ""],
                    {"default": "overlap_last_frame"},
                ),
            },
            "optional": {
                "planner_summary": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = (
        "boundary_anchor_frames",
        "new_chunk_start_frames",
        "paired_keyframes",
        "contact_sheet",
        "summary",
    )
    FUNCTION = "extract"
    CATEGORY = f"{CATEGORY}/Chunk Plan"

    def extract(
        self,
        video: torch.Tensor,
        mode: str,
        max_chunk_frames: int = 81,
        overlap_frames: int = 5,
        max_frames: int = 0,
        include_final_anchor: bool = False,
        contact_sheet_columns: int = 4,
        contact_sheet_thumbnail_width: int = 256,
        boundary_anchor_mode: str = "overlap_last_frame",
        planner_summary: str = "",
    ):
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            raise ValueError("video must be a ComfyUI IMAGE tensor.")
        video_frame_count = int(video.shape[0])
        if video_frame_count <= 0:
            raise ValueError("video has no frames.")

        frame_cap = min(video_frame_count, int(max_frames)) if int(max_frames) > 0 else video_frame_count
        if frame_cap <= 0:
            raise ValueError("max_frames leaves no frames to extract.")

        normalized_max_chunk_frames = min(81, _ceil_to_4n_plus_1(max(17, int(max_chunk_frames))))
        overlap = _normalize_overlap(overlap_frames, normalized_max_chunk_frames)

        if str(mode) == "planner_summary":
            chunks, source = _parse_planner_chunks(
                planner_summary,
                normalized_max_chunk_frames,
                overlap,
                frame_cap,
            )
            if chunks[-1]["output_end"] > frame_cap:
                raise ValueError(
                    "planner_summary requests frames beyond the selected video range: "
                    f"last_output_end={chunks[-1]['output_end']}, available_frames={frame_cap}."
                )
        elif str(mode) == "standard_long_video":
            segments = [
                {
                    "index": 0,
                    "start": 0,
                    "end": frame_cap,
                    "frames": frame_cap,
                    "reference": 1,
                    "boundary_overlap": None,
                    "prompt": "",
                    "negative": "",
                }
            ]
            chunks = _build_chunk_plan(segments, normalized_max_chunk_frames, overlap)
            source = "standard_long_video"
        else:
            raise ValueError(f"Unsupported keyframe extraction mode: {mode}")

        normalized_boundary_anchor_mode = (
            "overlap_first_frame"
            if str(boundary_anchor_mode or "").strip() == "overlap_first_frame"
            else "overlap_last_frame"
        )
        anchor_indices, start_indices, rows = _chunk_boundary_indices(
            chunks,
            frame_cap,
            bool(include_final_anchor),
            normalized_boundary_anchor_mode,
        )
        boundary_anchor_frames = _extract_frame_batch(video[:frame_cap], anchor_indices, "boundary_anchor_frames")
        new_chunk_start_frames = _extract_frame_batch(video[:frame_cap], start_indices, "new_chunk_start_frames")
        paired_keyframes, paired_keyframes_manifest = _build_paired_keyframes(
            boundary_anchor_frames,
            new_chunk_start_frames,
            rows,
            bool(include_final_anchor),
        )
        contact_sheet = _build_keyframe_contact_sheet(
            boundary_anchor_frames,
            new_chunk_start_frames,
            rows,
            bool(include_final_anchor),
            int(contact_sheet_columns),
            int(contact_sheet_thumbnail_width),
        )

        safe_continued_keep = (
            normalized_max_chunk_frames - overlap
            if overlap > 0
            else normalized_max_chunk_frames
        )
        summary = {
            "mode": source,
            "video_frames": int(video_frame_count),
            "used_frames": int(frame_cap),
            "max_chunk_frames": int(normalized_max_chunk_frames),
            "overlap_frames": int(overlap),
            "safe_continued_keep_frames": int(safe_continued_keep),
            "include_final_anchor": bool(include_final_anchor),
            "boundary_anchor_mode": normalized_boundary_anchor_mode,
            "contact_sheet_columns": int(contact_sheet_columns),
            "contact_sheet_thumbnail_width": int(contact_sheet_thumbnail_width),
            "boundary_anchor_frame_indices_0_based": anchor_indices,
            "boundary_anchor_frame_numbers_1_based": [index + 1 for index in anchor_indices],
            "new_chunk_start_frame_indices_0_based": start_indices,
            "new_chunk_start_frame_numbers_1_based": [index + 1 for index in start_indices],
            "paired_keyframes_manifest": paired_keyframes_manifest,
            "chunks": rows,
            "note": (
                "boundary_anchor_frames and new_chunk_start_frames keep original frame size. "
                "paired_keyframes keeps original frame size in the same visual order as contact_sheet. "
                "contact_sheet is a labeled preview image only."
            ),
        }
        return (
            boundary_anchor_frames,
            new_chunk_start_frames,
            paired_keyframes,
            contact_sheet,
            json.dumps(summary, indent=2),
        )


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
                {"default": 5, "min": -1, "max": 33, "step": 1},
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
                "cache_mode": (["disk", "off"], {"default": "disk"}),
            },
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("frames", "used_pose_video_mask", "used_reference_mask_timeline", "summary")
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

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
        cache_mode: str = "disk",
        pose_video_mask=None,
        prompt=None,
        unique_id=None,
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
                "prompt_graph": _prompt_upstream_fingerprint(prompt, unique_id),
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
        cache_mode = str(cache_mode or "disk").strip().lower()
        use_disk_cache = cache_mode == "disk"
        cached_key = getattr(self, "_last_generate_cache_key", None)
        cached_result = getattr(self, "_last_generate_cache_result", None)
        if cached_key == call_cache_key and cached_result is not None:
            print("[SCAIL2ScheduledLongVideo] semantic cache hit; returning previous result without sampling.")
            return _clone_cached_result(cached_result)
        if use_disk_cache:
            disk_result = _load_single_slot_disk_cache("SCAIL2ScheduledLongVideo", unique_id, call_cache_key)
            if disk_result is not None:
                print("[SCAIL2ScheduledLongVideo] disk cache hit; returning previous result without sampling.")
                self._last_generate_cache_key = call_cache_key
                self._last_generate_cache_result = _clone_cached_result(disk_result)
                return _clone_cached_result(disk_result)

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
        if use_disk_cache:
            _save_single_slot_disk_cache("SCAIL2ScheduledLongVideo", unique_id, call_cache_key, result)
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
                "cache_mode": (["disk", "off"], {"default": "disk"}),
            },
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
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
        cache_mode: str = "disk",
        sam_model=None,
        sam_conditioning=None,
        prompt=None,
        unique_id=None,
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
                "prompt_graph": _prompt_upstream_fingerprint(prompt, unique_id),
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
        cache_mode = str(cache_mode or "disk").strip().lower()
        use_disk_cache = cache_mode == "disk"
        cached_key = getattr(self, "_last_internal_sam_cache_key", None)
        cached_result = getattr(self, "_last_internal_sam_cache_result", None)
        if cached_key == call_cache_key and cached_result is not None:
            print(
                "[SCAIL2ScheduledLongVideoWithSAM] semantic cache hit; "
                "returning previous result without SAM tracking/sampling."
            )
            return _clone_cached_result(cached_result)
        if use_disk_cache:
            disk_result = _load_single_slot_disk_cache("SCAIL2ScheduledLongVideoWithSAM", unique_id, call_cache_key)
            if disk_result is not None:
                print(
                    "[SCAIL2ScheduledLongVideoWithSAM] disk cache hit; "
                    "returning previous result without SAM tracking/sampling."
                )
                self._last_internal_sam_cache_key = call_cache_key
                self._last_internal_sam_cache_result = _clone_cached_result(disk_result)
                return _clone_cached_result(disk_result)

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
                cache_mode="off",
                prompt=prompt,
                unique_id=unique_id,
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
            if use_disk_cache:
                _save_single_slot_disk_cache("SCAIL2ScheduledLongVideoWithSAM", unique_id, call_cache_key, result)
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
            cache_mode="off",
            prompt=prompt,
            unique_id=unique_id,
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
        if use_disk_cache:
            _save_single_slot_disk_cache("SCAIL2ScheduledLongVideoWithSAM", unique_id, call_cache_key, result)
        return result


def _try_track_data_to_mask(track_data, frame_count: int, height: int, width: int) -> Optional[tuple[torch.Tensor, dict[str, Any]]]:
    try:
        import nodes
    except Exception:
        return None
    mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    candidates = []
    for name, node_cls in mappings.items():
        haystack = f"{name} {getattr(node_cls, '__name__', '')}".lower()
        if "sam3" not in haystack or "mask" not in haystack:
            continue
        return_types = tuple(getattr(node_cls, "RETURN_TYPES", ()) or ())
        if return_types and "MASK" not in return_types:
            continue
        candidates.append((name, node_cls))
    for name, node_cls in candidates:
        try:
            node = node_cls()
            function_name = getattr(node, "FUNCTION", getattr(node_cls, "FUNCTION", None))
            if not function_name or not hasattr(node, function_name):
                continue
            fn = getattr(node, function_name)
            kwargs: dict[str, Any] = {}
            unsupported = False
            for param_name, param in inspect.signature(fn).parameters.items():
                if param_name == "self":
                    continue
                lower = param_name.lower()
                if "track" in lower:
                    kwargs[param_name] = track_data
                elif lower in {"width", "w"}:
                    kwargs[param_name] = int(width)
                elif lower in {"height", "h"}:
                    kwargs[param_name] = int(height)
                elif lower in {"frame_count", "frames", "length"}:
                    kwargs[param_name] = int(frame_count)
                elif param.default is inspect._empty and param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    unsupported = True
                    break
            if unsupported:
                continue
            result = _node_result(fn(**kwargs))
            mask = next((value for value in result if isinstance(value, torch.Tensor)), None)
            if mask is None:
                continue
            normalized = _normalize_video_mask(mask, frame_count, height, width, name=f"{name}_mask")
            return normalized, {"node": name, "function": function_name}
        except Exception:
            continue
    return None


def _coerce_track_mask_tensor(
    value: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
) -> Optional[torch.Tensor]:
    tensor = value.detach().cpu().float()
    if int(tensor.numel()) <= 0:
        return None
    if float(tensor.min().item()) < 0.0 or float(tensor.max().item()) > 1.0:
        tensor = torch.sigmoid(tensor)

    if tensor.ndim == 2 and int(tensor.shape[0]) == int(height) and int(tensor.shape[1]) == int(width):
        tensor = tensor.unsqueeze(0).repeat(int(frame_count), 1, 1)
    elif tensor.ndim == 3:
        if int(tensor.shape[0]) == int(frame_count) and int(tensor.shape[1]) == int(height) and int(tensor.shape[2]) == int(width):
            pass
        elif int(tensor.shape[-2]) == int(height) and int(tensor.shape[-1]) == int(width):
            tensor = tensor.amax(dim=0, keepdim=True).repeat(int(frame_count), 1, 1)
        else:
            return None
    elif tensor.ndim == 4:
        if int(tensor.shape[0]) == int(frame_count) and int(tensor.shape[-2]) == int(height) and int(tensor.shape[-1]) == int(width):
            tensor = tensor.amax(dim=1)
        elif int(tensor.shape[1]) == int(frame_count) and int(tensor.shape[-2]) == int(height) and int(tensor.shape[-1]) == int(width):
            tensor = tensor.amax(dim=0)
        elif int(tensor.shape[0]) == int(frame_count) and int(tensor.shape[1]) == int(height) and int(tensor.shape[2]) == int(width):
            tensor = tensor.amax(dim=-1)
        else:
            return None
    else:
        return None

    if tensor.ndim != 3:
        return None
    try:
        normalized = _normalize_video_mask(tensor, int(frame_count), int(height), int(width), name="sam3_track_tensor")
    except Exception:
        return None
    if float(normalized.max().item()) <= 0.05:
        return None
    return normalized


def _extract_packed_sam3_track_mask(track_data, frame_count: int, height: int, width: int) -> Optional[tuple[torch.Tensor, dict[str, Any]]]:
    if not isinstance(track_data, dict) or "packed_masks" not in track_data:
        return None
    packed = track_data.get("packed_masks")
    if packed is None:
        return None
    try:
        from comfy.ldm.sam3.tracker import unpack_masks
        import torch.nn.functional as F

        unpacked = unpack_masks(packed.detach().cpu())
    except Exception:
        return None
    if not isinstance(unpacked, torch.Tensor) or unpacked.ndim != 4:
        return None

    if int(unpacked.shape[0]) != int(frame_count):
        return None
    union = unpacked.float().amax(dim=1, keepdim=True)
    if int(union.shape[-2]) != int(height) or int(union.shape[-1]) != int(width):
        union = F.interpolate(union, size=(int(height), int(width)), mode="nearest")
    mask = union[:, 0].float().clamp(0, 1).contiguous()
    if float(mask.max().item()) <= 0.05:
        return None
    return mask, {
        "source": "sam3_packed_masks",
        "objects": int(unpacked.shape[1]),
        "packed_shape": _shape(packed),
    }


def _extract_track_data_mask(track_data, frame_count: int, height: int, width: int) -> Optional[tuple[torch.Tensor, dict[str, Any]]]:
    packed = _extract_packed_sam3_track_mask(track_data, int(frame_count), int(height), int(width))
    if packed is not None:
        return packed

    seen: set[int] = set()
    candidates: list[tuple[float, str, torch.Tensor]] = []
    preferred_tokens = ("mask", "seg", "pred", "logit")
    blocked_tokens = ("image", "frame", "video", "rgb")

    def visit(value, path: str, depth: int = 0) -> None:
        if depth > 8:
            return
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)

        if isinstance(value, torch.Tensor):
            mask = _coerce_track_mask_tensor(value, int(frame_count), int(height), int(width))
            if mask is None:
                return
            density = float((mask > 0.05).float().mean().item())
            if density <= 0.0 or density >= 0.95:
                return
            path_lower = path.lower()
            if any(token in path_lower for token in blocked_tokens):
                return
            preferred = any(token in path_lower for token in preferred_tokens)
            if not preferred and density > 0.35:
                return
            score = abs(density - 0.045) + (0.0 if preferred else 0.25)
            candidates.append((score, path, mask))
            return

        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}", depth + 1)
            return

        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
            return

        for attr in ("masks", "mask", "pred_masks", "mask_logits", "segments", "outputs"):
            if hasattr(value, attr):
                try:
                    visit(getattr(value, attr), f"{path}.{attr}", depth + 1)
                except Exception:
                    continue

    visit(track_data, "track_data")
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    score, path, mask = candidates[0]
    return mask, {"source": "sam3_track_data_tensor", "path": path, "score": float(score)}


class SCAIL2HeadTrackCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_body_video": ("IMAGE",),
                "crop_padding_ratio": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 3.0, "step": 0.05}),
                "square_align": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "temporal_smoothing": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.98, "step": 0.05}),
                "mask_expand_px": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "mask_blur_px": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "sam_detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_max_objects": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "sam_detect_interval": ("INT", {"default": 2, "min": 1, "max": 999, "step": 1}),
                "crop_mode": (["center_follow", "fixed_canvas"], {"default": "center_follow"}),
                "mask_component_mode": (["largest", "all"], {"default": "largest"}),
            },
            "optional": {
                "head_masks": ("MASK",),
                "sam_model": ("MODEL",),
                "head_conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("face_crop_video", "crop_masks", "crop_manifest", "debug_preview")
    FUNCTION = "crop"
    CATEGORY = f"{CATEGORY}/Face Detail"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def crop(
        self,
        full_body_video: torch.Tensor,
        crop_padding_ratio: float,
        square_align: int,
        temporal_smoothing: float,
        mask_expand_px: int,
        mask_blur_px: int,
        sam_detection_threshold: float,
        sam_max_objects: int,
        sam_detect_interval: int,
        crop_mode: str = "center_follow",
        mask_component_mode: str = "largest",
        head_masks: Optional[torch.Tensor] = None,
        sam_model=None,
        head_conditioning=None,
    ):
        if not isinstance(full_body_video, torch.Tensor) or full_body_video.ndim != 4:
            raise ValueError("full_body_video must be a ComfyUI IMAGE tensor.")
        video = full_body_video.detach().cpu().float().clamp(0, 1).contiguous()
        frame_count, height, width, _channels = video.shape
        mask_source = {"source": "head_masks"}
        if head_masks is not None:
            masks = _normalize_video_mask(head_masks, int(frame_count), int(height), int(width), name="head_masks")
        elif sam_model is not None and head_conditioning is not None:
            track_data = _run_sam3_track(
                video,
                sam_model,
                head_conditioning,
                detection_threshold=float(sam_detection_threshold),
                max_objects=int(sam_max_objects),
                detect_interval=int(sam_detect_interval),
            )
            conversion_errors: list[str] = []
            extracted = _extract_track_data_mask(track_data, int(frame_count), int(height), int(width))
            if extracted is not None:
                masks, extractor = extracted
                mask_source = extractor
            else:
                conversion_errors.append("SAM3TrackDataTensor:no_matching_mask_tensor")
                converted = _try_track_data_to_mask(track_data, int(frame_count), int(height), int(width))
                if converted is not None:
                    masks, converter = converted
                    mask_source = {"source": "sam3_dynamic_converter", "converter": converter}
                else:
                    conversion_errors.append("SAM3DynamicConverter:no_converter_matched")
                    detail = "; ".join(conversion_errors) if conversion_errors else "no converter matched"
                    raise RuntimeError(
                        "SAM3 track data could not be converted to a regular face/head MASK in this ComfyUI build. "
                        "Connect a real face/head MASK to head_masks, or use a ComfyUI SAM3 node that outputs MASK. "
                        f"Conversion detail: {detail}"
                    )
            if float(masks.max().item()) <= 0.05:
                raise RuntimeError(
                    "SAM3 tracking produced an empty head mask. Try a more specific prompt such as 'face' or 'head', "
                    "increase sam_max_objects, lower sam_detection_threshold, or connect an explicit head_masks MASK."
                )
        else:
            raise ValueError("Connect head_masks, or connect sam_model and head_conditioning if your ComfyUI exposes SAM3 mask conversion.")

        normalized_mask_component_mode = str(mask_component_mode or "largest").strip()
        if normalized_mask_component_mode not in {"largest", "all"}:
            normalized_mask_component_mode = "largest"
        component_summary: Optional[dict[str, Any]] = None
        if normalized_mask_component_mode == "largest":
            masks, component_summary = _keep_largest_mask_components(masks)

        masks = _binary_mask_morph(masks, expand_px=int(mask_expand_px), blur_px=int(mask_blur_px))
        raw_bboxes = [_bbox_from_mask_frame(masks[index]) for index in range(int(frame_count))]
        filled_bboxes = _interpolate_missing_bboxes(raw_bboxes)
        smoothed = _smooth_bboxes(filled_bboxes, int(width), int(height), float(temporal_smoothing))
        normalized_crop_mode = str(crop_mode or "center_follow").strip() or "center_follow"
        if normalized_crop_mode not in {"center_follow", "fixed_canvas"}:
            normalized_crop_mode = "center_follow"
        canvas_bbox_source_count = len(filled_bboxes)
        if normalized_crop_mode == "fixed_canvas":
            canvas_source_bbox = _union_bbox_from_bboxes(filled_bboxes, int(width), int(height))
            canvas_bbox = _square_bbox_from_bbox(
                canvas_source_bbox,
                int(width),
                int(height),
                float(crop_padding_ratio),
                int(square_align),
            )
            fixed_crop_size = int(canvas_bbox[2] - canvas_bbox[0])
            square_bboxes = [canvas_bbox for _index in range(int(frame_count))]
            bbox_strategy = "fixed_canvas_union_from_mask_bboxes"
            fixed_crop_anchor_frame: Optional[int] = None
        else:
            first_square_bbox = _square_bbox_from_bbox(
                smoothed[0],
                int(width),
                int(height),
                float(crop_padding_ratio),
                int(square_align),
            )
            fixed_crop_size = int(first_square_bbox[2] - first_square_bbox[0])
            square_bboxes = [
                _fixed_square_bbox_from_bbox(bbox, int(width), int(height), fixed_crop_size)
                for bbox in smoothed
            ]
            canvas_source_bbox = smoothed[0]
            bbox_strategy = "first_frame_fixed_square_from_mask_bbox"
            fixed_crop_anchor_frame = 0
        face_crop_video, crop_masks, frames = _crop_video_by_manifest(video, masks, square_bboxes)
        for index, (frame, mask_bbox, tracking_bbox) in enumerate(zip(frames, filled_bboxes, smoothed)):
            frame["mask_bbox"] = [int(value) for value in mask_bbox]
            frame["tracking_bbox"] = [int(value) for value in tracking_bbox]
            if raw_bboxes[index] is not None:
                frame["detected_mask_bbox"] = [int(value) for value in raw_bboxes[index]]
        manifest = {
            "version": 1,
            "source_shape": _shape(video),
            "crop_shape": _shape(face_crop_video),
            "mask_source": mask_source,
            "crop_mode": normalized_crop_mode,
            "crop_padding_ratio": float(crop_padding_ratio),
            "square_align": int(square_align),
            "crop_size_aligned": bool(int(fixed_crop_size) % max(1, int(square_align)) == 0),
            "fixed_crop_size": int(fixed_crop_size),
            "fixed_crop_anchor_frame": fixed_crop_anchor_frame,
            "canvas_source_bbox": [int(value) for value in canvas_source_bbox],
            "canvas_bbox_source_count": int(canvas_bbox_source_count),
            "canvas_bbox_filtered_count": 0,
            "temporal_smoothing": float(temporal_smoothing),
            "bbox_strategy": bbox_strategy,
            "crop_bbox_field": "bbox",
            "mask_bbox_field": "mask_bbox",
            "tracking_bbox_field": "tracking_bbox",
            "detected_mask_bbox_field": "detected_mask_bbox",
            "mask_handling": "crop mask directly after optional largest-component filtering; no inner head-box estimation or bbox clipping",
            "mask_component_mode": normalized_mask_component_mode,
            "mask_component_summary": component_summary,
            "mask_expand_px": int(mask_expand_px),
            "mask_blur_px": int(mask_blur_px),
            "missing_mask_frames": [int(index) for index, bbox in enumerate(raw_bboxes) if bbox is None],
            "frames": frames,
        }
        debug_preview = _draw_manifest_debug(video, frames)
        return (face_crop_video, crop_masks, json.dumps(manifest, indent=2), debug_preview)


class SCAIL2AlignReferenceFaceToCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_crop_video": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "target_frame_index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
                "face_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01}),
                "x_offset_ratio": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005}),
                "y_offset_ratio": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.005}),
                "face_size_basis": (["bbox_width", "bbox_height"], {"default": "bbox_width"}),
                "target_face_select": (["center", "largest"], {"default": "center"}),
                "reference_face_select": (["largest", "center"], {"default": "largest"}),
                "padding_mode": (["edge", "reflect", "black", "white", "mean"], {"default": "edge"}),
                "insightface_model": (["buffalo_l", "buffalo_s"], {"default": "buffalo_l"}),
                "provider": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "det_size": ("INT", {"default": 640, "min": 160, "max": 2048, "step": 32}),
                "face_detector_backend": (["auto", "insightface", "mediapipe"], {"default": "auto"}),
                "mediapipe_model_selection": (["full_range", "short_range"], {"default": "full_range"}),
                "mediapipe_min_detection_confidence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("aligned_reference_image", "debug_preview", "summary")
    FUNCTION = "align"
    CATEGORY = f"{CATEGORY}/Face Detail"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def align(
        self,
        face_crop_video: torch.Tensor,
        reference_image: torch.Tensor,
        target_frame_index: int = 0,
        face_scale: float = 1.0,
        x_offset_ratio: float = 0.0,
        y_offset_ratio: float = 0.0,
        face_size_basis: str = "bbox_width",
        target_face_select: str = "center",
        reference_face_select: str = "largest",
        padding_mode: str = "edge",
        insightface_model: str = "buffalo_l",
        provider: str = "auto",
        det_size: int = 640,
        face_detector_backend: str = "auto",
        mediapipe_model_selection: str = "full_range",
        mediapipe_min_detection_confidence: float = 0.5,
    ):
        target_rgb = _image_batch_frame_to_rgb_uint8(face_crop_video, int(target_frame_index), "face_crop_video")
        reference_rgb = _image_batch_frame_to_rgb_uint8(reference_image, 0, "reference_image")
        target_h, target_w = int(target_rgb.shape[0]), int(target_rgb.shape[1])
        ref_h, ref_w = int(reference_rgb.shape[0]), int(reference_rgb.shape[1])
        if target_h <= 0 or target_w <= 0 or ref_h <= 0 or ref_w <= 0:
            raise ValueError("face_crop_video and reference_image must contain non-empty images.")

        target_info, reference_info, detector_info = _detect_face_pair(
            target_rgb,
            reference_rgb,
            str(face_detector_backend),
            str(target_face_select),
            str(reference_face_select),
            str(insightface_model),
            str(provider),
            int(det_size),
            str(mediapipe_model_selection),
            float(mediapipe_min_detection_confidence),
        )

        target_aspect = float(target_w) / max(1.0, float(target_h))
        normalized_basis = str(face_size_basis or "bbox_width").strip()
        if normalized_basis not in {"bbox_width", "bbox_height"}:
            normalized_basis = "bbox_width"
        scale = max(0.25, min(4.0, float(face_scale)))
        if normalized_basis == "bbox_height":
            target_relative_size = float(target_info["size"][1]) / max(1.0, float(target_h))
            reference_size = float(reference_info["size"][1])
            canvas_h = int(round(reference_size / max(1e-6, target_relative_size * scale)))
            canvas_w = int(round(float(canvas_h) * target_aspect))
        else:
            target_relative_size = float(target_info["size"][0]) / max(1.0, float(target_w))
            reference_size = float(reference_info["size"][0])
            canvas_w = int(round(reference_size / max(1e-6, target_relative_size * scale)))
            canvas_h = int(round(float(canvas_w) / max(1e-6, target_aspect)))
        canvas_w = max(1, canvas_w)
        canvas_h = max(1, canvas_h)

        target_relative_center_x = float(target_info["center"][0]) / max(1.0, float(target_w)) + float(x_offset_ratio)
        target_relative_center_y = float(target_info["center"][1]) / max(1.0, float(target_h)) + float(y_offset_ratio)
        ref_center_x = float(reference_info["center"][0])
        ref_center_y = float(reference_info["center"][1])
        window_x0 = int(round(ref_center_x - target_relative_center_x * float(canvas_w)))
        window_y0 = int(round(ref_center_y - target_relative_center_y * float(canvas_h)))
        aligned_rgb, window_info = _extract_reference_window_no_resize(
            reference_rgb,
            window_x0,
            window_y0,
            int(canvas_w),
            int(canvas_h),
            str(padding_mode),
        )
        aligned_reference = _rgb_uint8_to_image_tensor(aligned_rgb)

        target_preview = _rgb_uint8_to_image_tensor(target_rgb)
        target_marked = target_preview.clone()
        _draw_rect(target_marked, tuple(int(value) for value in target_info["bbox_int"]), (0.1, 0.9, 0.25))
        aligned_preview = _resize_image_tensor_like(
            aligned_reference,
            torch.empty((1, int(target_h), int(target_w), 3)),
            mode="bilinear",
        )
        sx = float(target_w) / max(1.0, float(canvas_w))
        sy = float(target_h) / max(1.0, float(canvas_h))
        ref_bbox = reference_info["bbox"]
        aligned_bbox = (
            int(round((float(ref_bbox[0]) - float(window_x0)) * sx)),
            int(round((float(ref_bbox[1]) - float(window_y0)) * sy)),
            int(round((float(ref_bbox[2]) - float(window_x0)) * sx)),
            int(round((float(ref_bbox[3]) - float(window_y0)) * sy)),
        )
        aligned_marked = aligned_preview.clone()
        _draw_rect(aligned_marked, aligned_bbox, (0.95, 0.25, 0.15))
        debug_preview = torch.cat([target_marked, aligned_marked], dim=2).contiguous().clamp(0, 1)

        aligned_face_bbox = [
            float(ref_bbox[0]) - float(window_x0),
            float(ref_bbox[1]) - float(window_y0),
            float(ref_bbox[2]) - float(window_x0),
            float(ref_bbox[3]) - float(window_y0),
        ]
        summary = {
            "method": "face_detector_bbox_ratio_crop_pad_no_resize",
            "face_detector": detector_info,
            "insightface": detector_info if detector_info.get("backend_used") == "insightface" else None,
            "target_frame_index": int(max(0, min(int(face_crop_video.shape[0]) - 1, int(target_frame_index)))),
            "target_crop_shape": [1, int(target_h), int(target_w), 3],
            "reference_shape": [1, int(ref_h), int(ref_w), 3],
            "aligned_reference_shape": _shape(aligned_reference),
            "reference_pixels_resized": False,
            "output_aspect": float(canvas_w / max(1, canvas_h)),
            "target_aspect": float(target_aspect),
            "face_scale": float(scale),
            "x_offset_ratio": float(x_offset_ratio),
            "y_offset_ratio": float(y_offset_ratio),
            "face_size_basis": normalized_basis,
            "target_face": target_info,
            "reference_face": reference_info,
            "aligned_reference_face_bbox": [float(value) for value in aligned_face_bbox],
            "target_relative_center": [float(target_relative_center_x), float(target_relative_center_y)],
            "target_relative_face_size": float(target_relative_size),
            "reference_face_size_px": float(reference_size),
            "reference_window": window_info,
            "debug_preview": "left is target crop frame, right is aligned reference resized only for visual comparison",
            "connect_to": "Use aligned_reference_image as the face-detail pass reference_N input.",
        }
        return (aligned_reference.contiguous().clamp(0, 1), debug_preview, json.dumps(summary, indent=2))


class SCAIL2FaceCompositeBack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_body_video": ("IMAGE",),
                "refined_face_video": ("IMAGE",),
                "crop_masks": ("MASK",),
                "crop_manifest": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "feather_px": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),
                "mask_contract_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "color_correction": ("BOOLEAN", {"default": True}),
                "color_match_method": (["local_mean_std", "none"], {"default": "local_mean_std"}),
                "face_fit_mode": (["center_crop", "pad", "stretch"], {"default": "center_crop"}),
                "frame_mismatch_mode": (["trim_to_shortest", "error"], {"default": "trim_to_shortest"}),
                "stitch_mask_expand_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "stitch_mask_resize_mode": (["bilinear", "nearest"], {"default": "bilinear"}),
                "stitch_offset_x_px": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
                "stitch_offset_y_px": ("INT", {"default": 0, "min": -128, "max": 128, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "summary", "debug_preview")
    FUNCTION = "composite"
    CATEGORY = f"{CATEGORY}/Face Detail"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def composite(
        self,
        full_body_video: torch.Tensor,
        refined_face_video: torch.Tensor,
        crop_masks: torch.Tensor,
        crop_manifest: str,
        feather_px: int,
        mask_contract_px: int,
        color_correction: bool,
        color_match_method: str,
        face_fit_mode: str = "center_crop",
        frame_mismatch_mode: str = "trim_to_shortest",
        stitch_mask_expand_px: int = 0,
        stitch_mask_resize_mode: str = "bilinear",
        stitch_offset_x_px: int = 0,
        stitch_offset_y_px: int = 0,
    ):
        if not isinstance(full_body_video, torch.Tensor) or full_body_video.ndim != 4:
            raise ValueError("full_body_video must be a ComfyUI IMAGE tensor.")
        if not isinstance(refined_face_video, torch.Tensor) or refined_face_video.ndim != 4:
            raise ValueError("refined_face_video must be a ComfyUI IMAGE tensor.")
        manifest = _parse_crop_manifest(crop_manifest)
        frames_manifest = manifest["frames"]
        full_frame_count = int(full_body_video.shape[0])
        refined_frame_count = int(refined_face_video.shape[0])
        manifest_frame_count = len(frames_manifest)
        if not isinstance(crop_masks, torch.Tensor) or crop_masks.ndim not in {3, 4}:
            raise ValueError("crop_masks must be a ComfyUI MASK tensor.")
        mask_frame_count = int(crop_masks.shape[0])
        normalized_frame_mismatch_mode = str(frame_mismatch_mode or "trim_to_shortest").strip()
        if normalized_frame_mismatch_mode not in {"trim_to_shortest", "error"}:
            normalized_frame_mismatch_mode = "trim_to_shortest"
        if normalized_frame_mismatch_mode == "error":
            if (
                refined_frame_count != full_frame_count
                or manifest_frame_count != full_frame_count
                or mask_frame_count not in {1, full_frame_count}
            ):
                raise ValueError("full_body_video, refined_face_video, crop_masks, and crop_manifest must have the same frame count.")
            frame_count = full_frame_count
        else:
            frame_count_candidates = [full_frame_count, refined_frame_count, manifest_frame_count]
            if mask_frame_count > 1:
                frame_count_candidates.append(mask_frame_count)
            frame_count = min(frame_count_candidates)
            if frame_count <= 0:
                raise ValueError("composite inputs produced no usable frames after trimming to the shortest input.")
            frames_manifest = frames_manifest[:frame_count]
            crop_masks = crop_masks[:frame_count] if mask_frame_count > 1 else crop_masks
        full = full_body_video[:frame_count].detach().cpu().float().clamp(0, 1).contiguous()
        refined = refined_face_video[:frame_count].detach().cpu().float().clamp(0, 1).contiguous()
        first_bbox = frames_manifest[0].get("crop_to_canvas_bbox", frames_manifest[0].get("bbox"))
        if not isinstance(first_bbox, list) or len(first_bbox) != 4:
            raise ValueError("crop_manifest frame 0 is missing crop_to_canvas_bbox/bbox.")
        fx0, fy0, fx1, fy1 = [int(value) for value in first_bbox]
        fx0, fy0, fx1, fy1 = _clamp_bbox((fx0, fy0, fx1, fy1), int(full.shape[2]), int(full.shape[1]))
        crop_canvas_h = max(1, int(fy1 - fy0))
        crop_canvas_w = max(1, int(fx1 - fx0))
        masks = _normalize_video_mask(crop_masks, frame_count, crop_canvas_h, crop_canvas_w, name="crop_masks")
        masks = _binary_mask_morph(
            masks,
            expand_px=int(stitch_mask_expand_px),
            contract_px=int(mask_contract_px),
            blur_px=int(feather_px),
        )
        normalized_face_fit_mode = str(face_fit_mode or "center_crop").strip()
        if normalized_face_fit_mode not in {"center_crop", "pad", "stretch"}:
            normalized_face_fit_mode = "center_crop"
        normalized_stitch_mask_resize_mode = str(stitch_mask_resize_mode or "bilinear").strip()
        if normalized_stitch_mask_resize_mode not in {"bilinear", "nearest"}:
            normalized_stitch_mask_resize_mode = "bilinear"
        output = full.clone()
        color_events: list[dict[str, Any]] = []
        for index, item in enumerate(frames_manifest):
            bbox = item.get("crop_to_canvas_bbox", item.get("bbox"))
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"crop_manifest frame {index} is missing crop_to_canvas_bbox/bbox.")
            x0, y0, x1, y1 = [int(value) for value in bbox]
            x0, y0, x1, y1 = _offset_bbox_within_bounds(
                (x0, y0, x1, y1),
                int(full.shape[2]),
                int(full.shape[1]),
                int(stitch_offset_x_px),
                int(stitch_offset_y_px),
            )
            target_h = max(1, y1 - y0)
            target_w = max(1, x1 - x0)
            face = _fit_image_to_size(
                refined[index : index + 1, :, :, :3],
                target_h,
                target_w,
                fit_mode=normalized_face_fit_mode,
                mode="bilinear",
            )
            mask = masks[index : index + 1].unsqueeze(-1)
            if int(mask.shape[1]) != target_h or int(mask.shape[2]) != target_w:
                mask = _resize_image_tensor_like(
                    mask,
                    torch.empty((1, target_h, target_w, 1)),
                    mode=normalized_stitch_mask_resize_mode,
                )
            mask = mask.clamp(0, 1)
            target = output[index : index + 1, y0:y1, x0:x1, :3]
            if bool(color_correction) and str(color_match_method) == "local_mean_std":
                face, color_info = _local_mean_std_color_match(face, target, mask)
                if index < 12:
                    color_events.append({"frame": int(index), **color_info})
            mixed = torch.lerp(target, face[:, :, :, :3], mask).clamp(0, 1)
            output[index : index + 1, y0:y1, x0:x1, :3] = mixed
        debug_preview = _draw_manifest_debug(output, frames_manifest)
        summary = {
            "source_shape": _shape(full),
            "refined_face_shape": _shape(refined),
            "output_shape": _shape(output),
            "crop_canvas_shape": [int(frame_count), int(crop_canvas_h), int(crop_canvas_w), 3],
            "crop_mask_shape": _shape(masks),
            "input_frame_counts": {
                "full_body_video": int(full_frame_count),
                "refined_face_video": int(refined_frame_count),
                "crop_masks": int(mask_frame_count),
                "crop_manifest": int(manifest_frame_count),
            },
            "used_frame_count": int(frame_count),
            "trimmed_tail_frames": {
                "full_body_video": max(0, int(full_frame_count) - int(frame_count)),
                "refined_face_video": max(0, int(refined_frame_count) - int(frame_count)),
                "crop_masks": 0 if int(mask_frame_count) == 1 else max(0, int(mask_frame_count) - int(frame_count)),
                "crop_manifest": max(0, int(manifest_frame_count) - int(frame_count)),
            },
            "frame_mismatch_mode": normalized_frame_mismatch_mode,
            "feather_px": int(feather_px),
            "mask_contract_px": int(mask_contract_px),
            "stitch_mask_expand_px": int(stitch_mask_expand_px),
            "stitch_mask_resize_mode": normalized_stitch_mask_resize_mode,
            "stitch_offset_x_px": int(stitch_offset_x_px),
            "stitch_offset_y_px": int(stitch_offset_y_px),
            "face_fit_mode": normalized_face_fit_mode,
            "color_correction": bool(color_correction),
            "color_match_method": str(color_match_method) if bool(color_correction) else "none",
            "sample_color_events": color_events,
        }
        return (output.contiguous().clamp(0, 1), json.dumps(summary, indent=2), debug_preview)


class SCAIL2KeyframeMatrixViewer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "paired_keyframes": ("IMAGE",),
                "summary": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "filename_prefix": ("STRING", {"default": "scail_keyframe"}),
                "save_location": (
                    ["temp", "output", "", "both", "overlap_boundary_only", "new_chunk_start_only"],
                    {"default": "temp"},
                ),
                "display_group": (
                    ["both", "overlap_boundary_only", "new_chunk_start_only", "", "temp", "output"],
                    {"default": "both"},
                ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = f"{CATEGORY}/Chunk Plan"

    def view(
        self,
        paired_keyframes: torch.Tensor,
        summary: str,
        filename_prefix: str = "scail_keyframe",
        save_location: str = "temp",
        display_group: str = "both",
    ):
        display_choices = {"both", "overlap_boundary_only", "new_chunk_start_only"}
        save_choices = {"temp", "output"}
        save_text = str(save_location or "").strip()
        group_text = str(display_group or "").strip()
        if save_text in display_choices:
            display_group = save_text
            save_location = group_text if group_text in save_choices else "temp"
        elif group_text in save_choices and save_text not in save_choices:
            save_location = group_text
            display_group = "both"
        matrix = _save_keyframe_matrix_images(
            paired_keyframes,
            summary,
            filename_prefix,
            save_location,
            display_group,
        )
        return {
            "ui": {
                "images": matrix["items"],
                "scail_keyframe_matrix": [matrix],
                "scail_keyframe_matrix_list": [matrix],
                "scail_keyframe_matrix_json": [json.dumps(matrix)],
            }
        }


NODE_CLASS_MAPPINGS = {
    "SCAIL2ChunkKeyframeExtractor": SCAIL2ChunkKeyframeExtractor,
    "SCAIL2KeyframeMatrixViewer": SCAIL2KeyframeMatrixViewer,
    "SCAIL2SegmentPlanBuilder": SCAIL2SegmentPlanBuilder,
    "SCAIL2SegmentPlanner": SCAIL2SegmentPlanner,
    "SCAIL2MultiReferenceColoredMask": SCAIL2MultiReferenceColoredMask,
    "SCAIL2ScheduledLongVideo": SCAIL2ScheduledLongVideo,
    "SCAIL2ScheduledLongVideoWithSAM": SCAIL2ScheduledLongVideoWithSAM,
    "SCAIL2HeadTrackCrop": SCAIL2HeadTrackCrop,
    "SCAIL2AlignReferenceFaceToCrop": SCAIL2AlignReferenceFaceToCrop,
    "SCAIL2FaceCompositeBack": SCAIL2FaceCompositeBack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAIL2ChunkKeyframeExtractor": "SCAIL-2 Chunk Keyframe Extractor",
    "SCAIL2KeyframeMatrixViewer": "SCAIL-2 Keyframe Matrix Viewer",
    "SCAIL2SegmentPlanBuilder": "SCAIL-2 Segment Plan Builder",
    "SCAIL2SegmentPlanner": "SCAIL-2 Segment Planner",
    "SCAIL2MultiReferenceColoredMask": "SCAIL-2 Multi Reference Colored Mask",
    "SCAIL2ScheduledLongVideo": "SCAIL-2 Scheduled Long Video",
    "SCAIL2ScheduledLongVideoWithSAM": "SCAIL-2 Scheduled Long Video (Internal SAM)",
    "SCAIL2HeadTrackCrop": "SCAIL-2 Head Track Crop",
    "SCAIL2AlignReferenceFaceToCrop": "SCAIL-2 Align Reference Face To Crop",
    "SCAIL2FaceCompositeBack": "SCAIL-2 Face Composite Back",
}

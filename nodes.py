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
MAX_TILES = 8
DEFAULT_MAX_TILE_PIXELS = 1280 * 720
LONG_VIDEO_STATUS_EVENT = "scail2_long_video_status"
DEFAULT_PLAN = """# frames | reference | prompt | negative | boundary_overlap
49 | 1 | first segment prompt | | 5
121 | 2 | second segment prompt | | 5
73 | 3 | third segment prompt | | 5
157 | 4 | fourth segment prompt | | 5
"""
_INSIGHTFACE_APP_CACHE: dict[tuple[str, str, int], Any] = {}
_MEDIAPIPE_FACE_DETECTION_CACHE: dict[tuple[int, float], Any] = {}


def _send_long_video_status(
    unique_id: Any,
    node_name: str,
    stage: str,
    message: str,
    *,
    progress: Optional[dict[str, Any]] = None,
    **extra: Any,
) -> None:
    if unique_id is None:
        return
    payload: dict[str, Any] = {
        "node_id": str(unique_id),
        "node": str(node_name),
        "stage": str(stage),
        "message": str(message),
        "timestamp": float(time.time()),
    }
    if progress is not None:
        payload["progress"] = progress
    payload.update(extra)
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(LONG_VIDEO_STATUS_EVENT, payload)
    except Exception:
        pass


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


def _free_tail_window_input_spec() -> tuple[str, dict[str, Any]]:
    return (
        "BOOLEAN",
        {
            "default": False,
            "tooltip": (
                "Fill the final sampling window with blank, unreferenced tail frames, "
                "then discard those tail frames from the returned outputs."
            ),
        },
    )


def _pad_image_sequence_tail_with_zeros(sequence: torch.Tensor, frame_count: int) -> torch.Tensor:
    if not isinstance(sequence, torch.Tensor) or sequence.ndim != 4:
        raise ValueError("sequence must be a ComfyUI IMAGE tensor.")
    wanted = max(0, int(frame_count))
    current = int(sequence.shape[0])
    if current >= wanted:
        return sequence[:wanted].detach().contiguous()
    padding = torch.zeros(
        (wanted - current, int(sequence.shape[1]), int(sequence.shape[2]), int(sequence.shape[3])),
        dtype=sequence.dtype,
        device=sequence.device,
    )
    return torch.cat((sequence.detach(), padding), dim=0).contiguous()


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
    window_fit_mode: str = "shift_inside_reference",
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    height, width = int(image_rgb.shape[0]), int(image_rgb.shape[1])
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    requested_x0 = int(x0)
    requested_y0 = int(y0)
    requested_x1 = requested_x0 + canvas_w
    requested_y1 = requested_y0 + canvas_h
    normalized_fit_mode = str(window_fit_mode or "shift_inside_reference").strip()
    if normalized_fit_mode not in {"shift_inside_reference", "strict_alignment"}:
        normalized_fit_mode = "shift_inside_reference"
    x0 = requested_x0
    y0 = requested_y0
    if normalized_fit_mode == "shift_inside_reference":
        if canvas_w <= width:
            x0 = max(0, min(int(x0), width - canvas_w))
        if canvas_h <= height:
            y0 = max(0, min(int(y0), height - canvas_h))
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
        "fit_mode": normalized_fit_mode,
        "requested_window_xyxy": [int(requested_x0), int(requested_y0), int(requested_x1), int(requested_y1)],
        "window_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "window_shift_xy": [int(x0) - int(requested_x0), int(y0) - int(requested_y0)],
        "source_overlap_xyxy": [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)],
        "padding": {
            "left": int(pad_left),
            "top": int(pad_top),
            "right": int(pad_right),
            "bottom": int(pad_bottom),
            "mode": normalized,
            "unavoidable": bool(
                ((pad_left or pad_right) and canvas_w > width)
                or ((pad_top or pad_bottom) and canvas_h > height)
            ),
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


def _align_up(value: int, align: int) -> int:
    value = max(1, int(value))
    align = max(1, int(align))
    return int(math.ceil(value / align) * align)


def _align_to_step(value: int, align: int, mode: str = "nearest") -> int:
    value = max(1, int(value))
    align = max(1, int(align))
    normalized = str(mode or "nearest").strip()
    if normalized == "floor":
        return max(align, int(math.floor(value / align) * align))
    if normalized == "ceil":
        return int(math.ceil(value / align) * align)
    lower = max(align, int(math.floor(value / align) * align))
    upper = int(math.ceil(value / align) * align)
    return lower if abs(value - lower) <= abs(upper - value) else upper


def _resize_image_batch(image: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    return _resize_image_tensor_like(
        image,
        torch.empty((1, max(1, int(height)), max(1, int(width)), int(image.shape[-1]))),
        mode=mode,
    )


def _parse_tile_manifest(tile_manifest: str) -> dict[str, Any]:
    try:
        manifest = json.loads(str(tile_manifest or "").strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"tile_manifest must be valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tiles"), list):
        raise ValueError("tile_manifest must contain a tiles array.")
    return manifest


def _tile_preview_from_manifest(video: torch.Tensor, manifest: dict[str, Any], max_frames: int = 1) -> torch.Tensor:
    frames = video[: max(1, min(int(max_frames), int(video.shape[0])))].detach().cpu().float().clamp(0, 1).clone()
    colors = [
        (0.1, 0.9, 0.25),
        (0.95, 0.25, 0.15),
        (0.15, 0.45, 0.95),
        (0.95, 0.75, 0.15),
    ]
    for tile in manifest.get("tiles", []):
        bbox = tile.get("source_crop_bbox")
        core = tile.get("source_core_bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            _draw_rect(frames, tuple(int(value) for value in bbox), colors[int(tile.get("index", 0)) % len(colors)])
        if isinstance(core, list) and len(core) == 4:
            _draw_rect(frames, tuple(int(value) for value in core), (1.0, 1.0, 1.0))
    protected = manifest.get("protected_region")
    if isinstance(protected, dict):
        bbox = protected.get("source_bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            _draw_rect(frames, tuple(int(value) for value in bbox), (1.0, 0.0, 1.0))
    return frames.contiguous()


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    padding_ratio: float,
    padding_px: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clamp_bbox(bbox, int(width), int(height))
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    pad_x = int(math.ceil(box_w * max(0.0, float(padding_ratio)) + max(0, int(padding_px))))
    pad_y = int(math.ceil(box_h * max(0.0, float(padding_ratio)) + max(0, int(padding_px))))
    return _clamp_bbox((x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), int(width), int(height))


def _protected_bbox_from_masks(
    masks: torch.Tensor,
    frame_count: int,
    height: int,
    width: int,
    padding_ratio: float,
    padding_px: int,
) -> Optional[dict[str, Any]]:
    normalized = _normalize_video_mask(masks, int(frame_count), int(height), int(width), name="protected_masks")
    union = normalized.amax(dim=0)
    raw = _bbox_from_mask_frame(union)
    if raw is None:
        return None
    padded = _expand_bbox(raw, int(width), int(height), float(padding_ratio), int(padding_px))
    area = max(1, (padded[2] - padded[0]) * (padded[3] - padded[1]))
    return {
        "source": "protected_masks",
        "raw_source_bbox": [int(value) for value in raw],
        "source_bbox": [int(value) for value in padded],
        "padding_ratio": float(padding_ratio),
        "padding_px": int(padding_px),
        "area_ratio": float(area / max(1, int(width) * int(height))),
    }


def _choose_protected_split(
    length: int,
    protected_interval: Optional[tuple[int, int]],
    min_tile_ratio: float,
) -> dict[str, Any]:
    length = max(2, int(length))
    center = int(round(length / 2.0))
    min_ratio = max(0.05, min(0.45, float(min_tile_ratio)))
    min_pos = max(1, int(math.ceil(length * min_ratio)))
    max_pos = min(length - 1, int(math.floor(length * (1.0 - min_ratio))))
    if min_pos > max_pos:
        min_pos = max_pos = max(1, min(length - 1, center))

    def clamp_position(value: int) -> int:
        return max(min_pos, min(max_pos, int(value)))

    if protected_interval is None:
        return {
            "position": clamp_position(center),
            "mode": "center",
            "avoids_protected_region": True,
            "min_position": int(min_pos),
            "max_position": int(max_pos),
        }

    p0, p1 = protected_interval
    p0 = max(0, min(length, int(p0)))
    p1 = max(0, min(length, int(p1)))
    if p1 < p0:
        p0, p1 = p1, p0
    center_pos = clamp_position(center)
    if not (p0 < center_pos < p1):
        return {
            "position": center_pos,
            "mode": "center_outside_protected_region",
            "avoids_protected_region": True,
            "protected_interval": [int(p0), int(p1)],
            "min_position": int(min_pos),
            "max_position": int(max_pos),
        }

    candidates = []
    for raw_position, side in ((p0, "before_protected_region"), (p1, "after_protected_region")):
        position = clamp_position(raw_position)
        avoids = not (p0 < position < p1)
        if avoids:
            candidates.append((abs(position - center), position, side))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1]))
        _distance, position, side = candidates[0]
        return {
            "position": int(position),
            "mode": side,
            "avoids_protected_region": True,
            "protected_interval": [int(p0), int(p1)],
            "min_position": int(min_pos),
            "max_position": int(max_pos),
        }

    return {
        "position": center_pos,
        "mode": "fallback_center_protected_region_too_large",
        "avoids_protected_region": False,
        "protected_interval": [int(p0), int(p1)],
        "min_position": int(min_pos),
        "max_position": int(max_pos),
    }


def _parse_manual_tile_layout(layout_json: str) -> dict[str, Any]:
    text = str(layout_json or "").strip()
    if not text:
        return {"split_x": 0.5, "split_y": 0.5, "source": "default_center"}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"layout_json must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("layout_json must be a JSON object.")
    split_x = float(value.get("split_x", 0.5))
    split_y = float(value.get("split_y", 0.5))
    return {
        **value,
        "split_x": max(0.01, min(0.99, split_x)),
        "split_y": max(0.01, min(0.99, split_y)),
    }


def _manual_core_bboxes_from_layout(
    layout: dict[str, Any],
    source_w: int,
    source_h: int,
    min_tile_ratio: float,
) -> tuple[list[list[int]], dict[str, Any]]:
    source_w = max(1, int(source_w))
    source_h = max(1, int(source_h))
    min_ratio = max(0.01, min(0.45, float(min_tile_ratio)))
    min_w = max(1, int(round(source_w * min_ratio)))
    min_h = max(1, int(round(source_h * min_ratio)))
    raw_tiles = layout.get("tiles")
    core_bboxes: list[list[int]] = []
    normalized_tiles: list[dict[str, Any]] = []
    if isinstance(raw_tiles, list) and raw_tiles:
        if len(raw_tiles) > MAX_TILES:
            raise ValueError(f"manual tile layout supports at most {MAX_TILES} tiles.")
        for index, raw in enumerate(raw_tiles):
            if not isinstance(raw, dict):
                raise ValueError(f"manual tile {index + 1} must be a JSON object.")
            if all(key in raw for key in ("x0", "y0", "x1", "y1")):
                x0 = float(raw.get("x0", 0.0))
                y0 = float(raw.get("y0", 0.0))
                x1 = float(raw.get("x1", 1.0))
                y1 = float(raw.get("y1", 1.0))
            else:
                x0 = float(raw.get("x", raw.get("left", 0.0)))
                y0 = float(raw.get("y", raw.get("top", 0.0)))
                width = float(raw.get("w", raw.get("width", 0.5)))
                height = float(raw.get("h", raw.get("height", 0.5)))
                x1 = x0 + width
                y1 = y0 + height
            x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
            y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
            px0 = max(0, min(source_w - 1, int(round(x0 * source_w))))
            py0 = max(0, min(source_h - 1, int(round(y0 * source_h))))
            px1 = max(px0 + 1, min(source_w, int(round(x1 * source_w))))
            py1 = max(py0 + 1, min(source_h, int(round(y1 * source_h))))
            if px1 - px0 < min_w or py1 - py0 < min_h:
                raise ValueError(
                    f"manual tile {index + 1} is smaller than min_tile_ratio={min_ratio}: "
                    f"{px1 - px0}x{py1 - py0}, minimum {min_w}x{min_h}."
                )
            core_bboxes.append([int(px0), int(py0), int(px1), int(py1)])
            normalized_tiles.append(
                {
                    "tile_number": len(core_bboxes),
                    "x0": float(px0 / source_w),
                    "y0": float(py0 / source_h),
                    "x1": float(px1 / source_w),
                    "y1": float(py1 / source_h),
                    "source_core_bbox": [int(px0), int(py0), int(px1), int(py1)],
                }
            )
    if core_bboxes:
        return core_bboxes, {"tiles": normalized_tiles, "tile_count": len(core_bboxes), "layout_mode": "rectangles"}

    split_x = max(min_ratio, min(1.0 - min_ratio, float(layout.get("split_x", 0.5))))
    split_y = max(min_ratio, min(1.0 - min_ratio, float(layout.get("split_y", 0.5))))
    split_x_px = max(1, min(source_w - 1, int(round(float(source_w) * split_x))))
    split_y_px = max(1, min(source_h - 1, int(round(float(source_h) * split_y))))
    core_bboxes = [
        [0, 0, int(split_x_px), int(split_y_px)],
        [int(split_x_px), 0, int(source_w), int(split_y_px)],
        [0, int(split_y_px), int(split_x_px), int(source_h)],
        [int(split_x_px), int(split_y_px), int(source_w), int(source_h)],
    ]
    return core_bboxes, {
        **layout,
        "split_x": float(split_x),
        "split_y": float(split_y),
        "split_x_px": int(split_x_px),
        "split_y_px": int(split_y_px),
        "min_tile_ratio": float(min_ratio),
        "tile_count": 4,
        "layout_mode": "split_2x2",
    }


def _manual_tile_entries_from_bboxes(
    core_bboxes: list[list[int]],
    source_w: int,
    source_h: int,
    auto_filled_from: Optional[int] = None,
) -> list[dict[str, Any]]:
    source_w = max(1, int(source_w))
    source_h = max(1, int(source_h))
    entries: list[dict[str, Any]] = []
    for index, bbox in enumerate(core_bboxes):
        x0, y0, x1, y1 = _clamp_bbox(tuple(int(value) for value in bbox), source_w, source_h)
        item = {
            "tile_number": int(index + 1),
            "x0": float(x0 / source_w),
            "y0": float(y0 / source_h),
            "x1": float(x1 / source_w),
            "y1": float(y1 / source_h),
            "source_core_bbox": [int(x0), int(y0), int(x1), int(y1)],
        }
        if auto_filled_from is not None and index >= int(auto_filled_from):
            item["auto_filled"] = True
        entries.append(item)
    return entries


def _manual_tile_coverage_gaps(
    core_bboxes: list[list[int]],
    source_w: int,
    source_h: int,
) -> dict[str, Any]:
    source_w = max(1, int(source_w))
    source_h = max(1, int(source_h))
    clamped_bboxes = [
        list(_clamp_bbox(tuple(int(value) for value in bbox), source_w, source_h))
        for bbox in core_bboxes
    ]
    x_edges = sorted({0, source_w, *(edge for bbox in clamped_bboxes for edge in (bbox[0], bbox[2]))})
    y_edges = sorted({0, source_h, *(edge for bbox in clamped_bboxes for edge in (bbox[1], bbox[3]))})
    row_runs: list[list[int]] = []
    uncovered_pixels = 0

    for y_index in range(len(y_edges) - 1):
        y0 = int(y_edges[y_index])
        y1 = int(y_edges[y_index + 1])
        if y1 <= y0:
            continue
        run: Optional[list[int]] = None
        for x_index in range(len(x_edges) - 1):
            x0 = int(x_edges[x_index])
            x1 = int(x_edges[x_index + 1])
            if x1 <= x0:
                continue
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            covered = any(
                cx >= bbox[0] and cx <= bbox[2] and cy >= bbox[1] and cy <= bbox[3]
                for bbox in clamped_bboxes
            )
            if covered:
                if run is not None:
                    row_runs.append(run)
                    run = None
                continue
            uncovered_pixels += int((x1 - x0) * (y1 - y0))
            if run is not None and run[2] == x0:
                run[2] = x1
            else:
                if run is not None:
                    row_runs.append(run)
                run = [x0, y0, x1, y1]
        if run is not None:
            row_runs.append(run)

    gaps: list[list[int]] = []
    active_by_span: dict[tuple[int, int], list[int]] = {}
    for run in sorted(row_runs, key=lambda item: (item[0], item[2], item[1], item[3])):
        key = (int(run[0]), int(run[2]))
        previous = active_by_span.get(key)
        if previous is not None and previous[3] == run[1]:
            previous[3] = int(run[3])
        else:
            gap = [int(value) for value in run]
            gaps.append(gap)
            active_by_span[key] = gap

    gaps.sort(key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), reverse=True)
    total_pixels = max(1, int(source_w * source_h))
    return {
        "gaps": gaps,
        "uncovered_pixels": int(uncovered_pixels),
        "uncovered_ratio": float(uncovered_pixels / total_pixels),
    }


def _manual_tile_coverage_summary(coverage: dict[str, Any]) -> dict[str, Any]:
    gaps = coverage.get("gaps", [])
    normalized_gaps = [[int(value) for value in gap] for gap in gaps if isinstance(gap, list) and len(gap) == 4]
    return {
        "uncovered_gap_count": int(len(normalized_gaps)),
        "uncovered_pixels": int(coverage.get("uncovered_pixels", 0)),
        "uncovered_ratio": float(coverage.get("uncovered_ratio", 0.0)),
        "uncovered_source_bboxes": normalized_gaps,
    }


def _expand_manual_gap_to_min_tile(
    bbox: list[int],
    source_w: int,
    source_h: int,
    min_w: int,
    min_h: int,
) -> list[int]:
    source_w = max(1, int(source_w))
    source_h = max(1, int(source_h))
    x0, y0, x1, y1 = _clamp_bbox(tuple(int(value) for value in bbox), source_w, source_h)

    def expand_axis(start: int, end: int, size: int, minimum: int) -> tuple[int, int]:
        size = max(1, int(size))
        minimum = max(1, min(size, int(minimum)))
        if end - start >= minimum:
            return int(start), int(end)
        center = (start + end) / 2.0
        new_start = int(round(center - minimum / 2.0))
        new_end = int(new_start + minimum)
        if new_start < 0:
            new_end -= new_start
            new_start = 0
        if new_end > size:
            new_start -= new_end - size
            new_end = size
        return int(max(0, new_start)), int(min(size, new_end))

    x0, x1 = expand_axis(int(x0), int(x1), source_w, min_w)
    y0, y1 = expand_axis(int(y0), int(y1), source_h, min_h)
    return [int(x0), int(y0), int(x1), int(y1)]


def _fill_manual_tile_gaps(
    core_bboxes: list[list[int]],
    source_w: int,
    source_h: int,
    min_tile_ratio: float,
) -> tuple[list[list[int]], dict[str, Any]]:
    source_w = max(1, int(source_w))
    source_h = max(1, int(source_h))
    min_ratio = max(0.01, min(0.45, float(min_tile_ratio)))
    min_w = max(1, int(round(source_w * min_ratio)))
    min_h = max(1, int(round(source_h * min_ratio)))
    filled = [
        list(_clamp_bbox(tuple(int(value) for value in bbox), source_w, source_h))
        for bbox in core_bboxes
    ]
    before = _manual_tile_coverage_gaps(filled, source_w, source_h)
    added: list[list[int]] = []
    current = before

    while current["gaps"] and len(filled) < MAX_TILES:
        gap = current["gaps"][0]
        addition = _expand_manual_gap_to_min_tile(gap, source_w, source_h, min_w, min_h)
        if addition in filled:
            break
        filled.append(addition)
        added.append(addition)
        current = _manual_tile_coverage_gaps(filled, source_w, source_h)

    after = _manual_tile_coverage_gaps(filled, source_w, source_h)
    return filled, {
        "coverage_policy": "auto_fill",
        "uncovered_before": _manual_tile_coverage_summary(before),
        "uncovered_after": _manual_tile_coverage_summary(after),
        "auto_filled_tile_count": int(len(added)),
        "auto_filled_source_core_bboxes": added,
    }


def _apply_manual_tile_coverage_policy(
    core_bboxes: list[list[int]],
    source_w: int,
    source_h: int,
    min_tile_ratio: float,
    coverage_policy: str,
) -> tuple[list[list[int]], dict[str, Any], Optional[int]]:
    policy = str(coverage_policy or "auto_fill").strip().lower()
    if policy not in {"auto_fill", "error", "ignore"}:
        policy = "auto_fill"

    original = [
        list(_clamp_bbox(tuple(int(value) for value in bbox), int(source_w), int(source_h)))
        for bbox in core_bboxes
    ]
    before = _manual_tile_coverage_gaps(original, int(source_w), int(source_h))
    if not before["gaps"]:
        summary = _manual_tile_coverage_summary(before)
        return original, {
            "coverage_policy": policy,
            "uncovered_before": summary,
            "uncovered_after": summary,
            "auto_filled_tile_count": 0,
            "auto_filled_source_core_bboxes": [],
        }, None

    if policy == "error":
        summary = _manual_tile_coverage_summary(before)
        raise ValueError(
            "Manual tile layout leaves uncovered source areas: "
            f"{summary['uncovered_gap_count']} gap(s), {summary['uncovered_pixels']} px. "
            "Move/resize tiles, click Fill gaps in the editor, or set coverage_policy to auto_fill/ignore."
        )

    if policy == "ignore":
        summary = _manual_tile_coverage_summary(before)
        return original, {
            "coverage_policy": policy,
            "uncovered_before": summary,
            "uncovered_after": summary,
            "auto_filled_tile_count": 0,
            "auto_filled_source_core_bboxes": [],
        }, None

    filled, fill_info = _fill_manual_tile_gaps(original, int(source_w), int(source_h), float(min_tile_ratio))
    after = fill_info["uncovered_after"]
    if int(after["uncovered_gap_count"]) > 0:
        raise ValueError(
            "Manual tile layout still leaves uncovered source areas after auto_fill: "
            f"{after['uncovered_gap_count']} gap(s), {after['uncovered_pixels']} px. "
            f"The planner supports at most {MAX_TILES} tiles; cover larger regions or delete extra small tiles."
        )
    auto_filled_from = len(original) if int(fill_info["auto_filled_tile_count"]) > 0 else None
    return filled, fill_info, auto_filled_from


def _validate_uniform_tile_scale(source_w: int, source_h: int, target_w: int, target_h: int) -> tuple[float, float]:
    scale_x = float(target_w) / max(1.0, float(source_w))
    scale_y = float(target_h) / max(1.0, float(source_h))
    if abs(scale_x - scale_y) > 1e-4:
        raise ValueError(
            "Tile upscale requires uniform final-canvas scale so each tile can be repainted proportionally. "
            f"source={source_w}x{source_h}, target={target_w}x{target_h}, "
            f"scale_x={scale_x:.6f}, scale_y={scale_y:.6f}. "
            "Set output_width/output_height to the same aspect ratio as source_video, or leave them at 0 and use scale_factor."
        )
    return scale_x, scale_y


def _resolve_tile_target_size(
    source_w: int,
    source_h: int,
    output_width: int,
    output_height: int,
    scale_factor: float,
) -> tuple[int, int, dict[str, Any]]:
    source_w = max(1, int(source_w))
    source_h = max(1, int(source_h))
    requested_w = max(0, int(output_width))
    requested_h = max(0, int(output_height))
    requested_scale = max(1.0, float(scale_factor))
    source_aspect = float(source_w / max(1, source_h))

    basis = "scale_factor"
    if requested_w > 0 and requested_h > 0:
        width_scale = float(requested_w / source_w)
        height_scale = float(requested_h / source_h)
        if abs(width_scale - height_scale) <= 1e-6:
            resolved_scale = max(1.0, (width_scale + height_scale) / 2.0)
            basis = "exact_output_size"
        else:
            width_delta = abs(width_scale - requested_scale)
            height_delta = abs(height_scale - requested_scale)
            if height_delta <= width_delta:
                resolved_scale = max(1.0, height_scale)
                basis = "output_height_preserve_aspect"
            else:
                resolved_scale = max(1.0, width_scale)
                basis = "output_width_preserve_aspect"
    elif requested_w > 0:
        resolved_scale = max(1.0, float(requested_w / source_w))
        basis = "output_width_preserve_aspect"
    elif requested_h > 0:
        resolved_scale = max(1.0, float(requested_h / source_h))
        basis = "output_height_preserve_aspect"
    else:
        resolved_scale = requested_scale

    target_w = max(1, int(round(source_w * resolved_scale)))
    target_h = max(1, int(round(source_h * resolved_scale)))
    resolved_scale_x, resolved_scale_y = _validate_uniform_tile_scale(source_w, source_h, target_w, target_h)
    requested_aspect = float(requested_w / requested_h) if requested_w > 0 and requested_h > 0 else None
    return target_w, target_h, {
        "requested_output_size": [int(requested_w), int(requested_h)],
        "resolved_target_size": [int(target_w), int(target_h)],
        "requested_scale_factor": float(requested_scale),
        "resolved_scale": [float(resolved_scale_x), float(resolved_scale_y)],
        "source_aspect_ratio": float(source_aspect),
        "requested_aspect_ratio": requested_aspect,
        "preserve_source_aspect": True,
        "resolution_basis": basis,
        "adjusted_output_size": bool(requested_w > 0 and requested_h > 0 and [requested_w, requested_h] != [target_w, target_h]),
    }


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return min(int(a1), int(b1)) > max(int(a0), int(b0))


def _manual_neighbor_overlap_edges(
    core_bboxes: list[list[int]],
    tile_index: int,
    overlap_x: int,
    overlap_y: int,
) -> dict[str, int]:
    bbox = [int(value) for value in core_bboxes[int(tile_index)]]
    x0, y0, x1, y1 = bbox
    tolerance = 1
    edges = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    for index, other_raw in enumerate(core_bboxes):
        if index == int(tile_index):
            continue
        ox0, oy0, ox1, oy1 = [int(value) for value in other_raw]
        vertical_overlap = _intervals_overlap(y0, y1, oy0, oy1)
        horizontal_overlap = _intervals_overlap(x0, x1, ox0, ox1)
        if vertical_overlap and (abs(ox1 - x0) <= tolerance or ox0 < x0 < ox1):
            edges["left"] = int(overlap_x)
        if vertical_overlap and (abs(ox0 - x1) <= tolerance or ox0 < x1 < ox1):
            edges["right"] = int(overlap_x)
        if horizontal_overlap and (abs(oy1 - y0) <= tolerance or oy0 < y0 < oy1):
            edges["top"] = int(overlap_y)
        if horizontal_overlap and (abs(oy0 - y1) <= tolerance or oy0 < y1 < oy1):
            edges["bottom"] = int(overlap_y)
    return edges


def _format_tile_resolution_report(manifest: dict[str, Any]) -> str:
    constraints = manifest.get("tile_output_constraints", {})
    lines = [
        "# tile | repaint_resolution | pixels | source_crop | target_crop | repaint_scale | composite_scale | overlap_edges",
    ]
    for tile in manifest.get("tiles", []):
        tile_number = int(tile.get("tile_number", int(tile.get("index", 0)) + 1))
        generate_w, generate_h = [int(value) for value in tile.get("tile_generate_size", [0, 0])]
        source_w, source_h = [int(value) for value in tile.get("source_crop_size", [0, 0])]
        target_w, target_h = [int(value) for value in tile.get("target_crop_size", [0, 0])]
        pixels = int(tile.get("tile_generate_pixels", generate_w * generate_h))
        repaint_scale = tile.get("tile_repaint_scale", [0, 0])
        composite_scale = tile.get("tile_to_target_scale", [0, 0])
        overlap_edges = tile.get("overlap_edges_px_source", {})
        overlap_summary = (
            f"L{int(overlap_edges.get('left', 0))}/R{int(overlap_edges.get('right', 0))}/"
            f"T{int(overlap_edges.get('top', 0))}/B{int(overlap_edges.get('bottom', 0))}"
        )
        lines.append(
            " | ".join(
                [
                    str(tile_number),
                    f"{generate_w}x{generate_h}",
                    str(pixels),
                    f"{source_w}x{source_h}",
                    f"{target_w}x{target_h}",
                    f"{float(repaint_scale[0]):.4f}x,{float(repaint_scale[1]):.4f}y",
                    f"{float(composite_scale[0]):.4f}x,{float(composite_scale[1]):.4f}y",
                    overlap_summary,
                ]
            )
        )
    lines.append("")
    lines.append(
        "max_tile_pixels="
        f"{int(constraints.get('max_tile_pixels', 0))} "
        f"enforce={bool(constraints.get('enforce_tile_pixel_limit', False))} "
        f"tile_align={int(manifest.get('tile_align', 1))}"
    )
    lines.append(
        "Use repaint_resolution for each tile's SCAIL pass. Composite uses tile_to_target_scale when fitting generated tiles back."
    )
    return "\n".join(lines)


def _tile_repaint_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# tile | actual_resolution | pixels | planned_resolution | target_crop | actual_repaint_scale | actual_to_target_scale | status",
    ]
    for tile in manifest.get("tiles", []):
        tile_number = int(tile.get("tile_number", int(tile.get("index", 0)) + 1))
        actual_w, actual_h = [int(value) for value in tile.get("actual_tile_output_size", [0, 0])]
        planned_w, planned_h = [int(value) for value in tile.get("tile_generate_size", [0, 0])]
        target_w, target_h = [int(value) for value in tile.get("target_crop_size", [0, 0])]
        repaint_scale = tile.get("actual_repaint_scale", [0, 0])
        target_scale = tile.get("actual_tile_to_target_scale", [0, 0])
        notes = tile.get("actual_repaint_notes", [])
        status = "ok" if not notes else ",".join(str(item) for item in notes)
        lines.append(
            " | ".join(
                [
                    str(tile_number),
                    f"{actual_w}x{actual_h}",
                    str(int(tile.get("actual_tile_output_pixels", actual_w * actual_h))),
                    f"{planned_w}x{planned_h}",
                    f"{target_w}x{target_h}",
                    f"{float(repaint_scale[0]):.4f}x,{float(repaint_scale[1]):.4f}y",
                    f"{float(target_scale[0]):.4f}x,{float(target_scale[1]):.4f}y",
                    status,
                ]
            )
        )
    constraints = manifest.get("actual_tile_output_constraints", manifest.get("tile_output_constraints", {}))
    lines.append("")
    lines.append(
        "max_tile_pixels="
        f"{int(constraints.get('max_tile_pixels', 0))} "
        f"enforce={bool(constraints.get('enforce_tile_pixel_limit', False))}"
    )
    lines.append("Use actual_tile_manifest as Tile Composite Video.tile_manifest.")
    return "\n".join(lines)


def _augment_manifest_with_actual_tile_outputs(
    manifest: dict[str, Any],
    tile_videos: list[torch.Tensor],
    max_tile_pixels: int,
    enforce_tile_pixel_limit: bool,
    expected_size_mismatch_mode: str,
    aspect_mismatch_mode: str,
    aspect_tolerance: float,
) -> dict[str, Any]:
    tiles = manifest.get("tiles")
    if not isinstance(tiles, list) or len(tiles) != len(tile_videos):
        raise ValueError(f"tile_manifest must contain {len(tile_videos)} tile entries.")

    normalized_expected_mode = str(expected_size_mismatch_mode or "warn").strip()
    if normalized_expected_mode not in {"ignore", "warn", "error"}:
        normalized_expected_mode = "warn"
    normalized_aspect_mode = str(aspect_mismatch_mode or "warn").strip()
    if normalized_aspect_mode not in {"ignore", "warn", "error"}:
        normalized_aspect_mode = "warn"
    tolerance = max(0.0, min(0.25, float(aspect_tolerance)))
    max_pixels = max(0, int(max_tile_pixels))

    updated = dict(manifest)
    updated_tiles: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for index, (tile, video) in enumerate(zip(tiles, tile_videos), start=1):
        if not isinstance(video, torch.Tensor) or video.ndim != 4:
            raise ValueError(f"tile_{index}_video must be a ComfyUI IMAGE tensor.")
        item = dict(tile)
        actual_h = int(video.shape[1])
        actual_w = int(video.shape[2])
        actual_pixels = int(actual_w * actual_h)
        source_w, source_h = [int(value) for value in item.get("source_crop_size", [0, 0])]
        target_w, target_h = [int(value) for value in item.get("target_crop_size", [0, 0])]
        planned_w, planned_h = [int(value) for value in item.get("tile_generate_size", [0, 0])]
        notes: list[str] = []

        if max_pixels > 0 and actual_pixels > max_pixels:
            message = (
                f"tile {index} actual output {actual_w}x{actual_h} has {actual_pixels} pixels, "
                f"over max_tile_pixels={max_pixels}"
            )
            notes.append("over_pixel_budget")
            if bool(enforce_tile_pixel_limit):
                errors.append(message)
            else:
                warnings.append(message)

        if planned_w > 0 and planned_h > 0 and [actual_w, actual_h] != [planned_w, planned_h]:
            message = f"tile {index} actual output {actual_w}x{actual_h} differs from planned {planned_w}x{planned_h}"
            notes.append("size_differs_from_plan")
            if normalized_expected_mode == "error":
                errors.append(message)
            elif normalized_expected_mode == "warn":
                warnings.append(message)

        actual_aspect = float(actual_w / max(1, actual_h))
        target_aspect = float(target_w / max(1, target_h)) if target_w > 0 and target_h > 0 else actual_aspect
        aspect_delta = abs(actual_aspect - target_aspect) / max(1e-6, target_aspect)
        if normalized_aspect_mode != "ignore" and aspect_delta > tolerance:
            message = (
                f"tile {index} actual aspect {actual_aspect:.6f} differs from target crop aspect "
                f"{target_aspect:.6f} by {aspect_delta:.4f}"
            )
            notes.append("aspect_differs_from_target")
            if normalized_aspect_mode == "error":
                errors.append(message)
            else:
                warnings.append(message)

        item.update(
            {
                "actual_tile_output_size": [int(actual_w), int(actual_h)],
                "actual_tile_output_pixels": int(actual_pixels),
                "actual_tile_output_frames": int(video.shape[0]),
                "actual_repaint_scale": [
                    float(actual_w / max(1, source_w)),
                    float(actual_h / max(1, source_h)),
                ],
                "actual_tile_to_target_scale": [
                    float(target_w / max(1, actual_w)),
                    float(target_h / max(1, actual_h)),
                ],
                "actual_aspect_ratio": float(actual_aspect),
                "target_crop_aspect_ratio": float(target_aspect),
                "actual_aspect_delta_ratio": float(aspect_delta),
                "actual_resolution_matches_plan": bool([actual_w, actual_h] == [planned_w, planned_h]),
                "actual_within_tile_pixel_limit": bool(max_pixels <= 0 or actual_pixels <= max_pixels),
                "actual_repaint_notes": notes,
            }
        )
        updated_tiles.append(item)

    if errors:
        raise ValueError("Tile repaint collector rejected actual tile outputs: " + "; ".join(errors))

    updated["tiles"] = updated_tiles
    updated["actual_tile_output_constraints"] = {
        "max_tile_pixels": int(max_pixels),
        "enforce_tile_pixel_limit": bool(enforce_tile_pixel_limit),
        "expected_size_mismatch_mode": normalized_expected_mode,
        "aspect_mismatch_mode": normalized_aspect_mode,
        "aspect_tolerance": float(tolerance),
    }
    updated["actual_tile_output_warnings"] = warnings
    updated["actual_tile_outputs_collected"] = True
    return updated


def _build_2x2_tile_manifest(
    source_video: torch.Tensor,
    target_w: int,
    target_h: int,
    overlap_ratio: float,
    tile_align: int,
    feather_px: int,
    x_edges: list[int],
    y_edges: list[int],
    *,
    mode: str,
    max_tile_pixels: int = DEFAULT_MAX_TILE_PIXELS,
    enforce_tile_pixel_limit: bool = True,
    resolution_snap_mode: str = "nearest",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    _frames, source_h, source_w, channels = source_video.shape
    rows = 2
    cols = 2
    target_w = max(cols, int(target_w))
    target_h = max(rows, int(target_h))
    overlap_ratio = max(0.0, min(0.45, float(overlap_ratio)))
    tile_align = max(1, int(tile_align))
    normalized_snap_mode = str(resolution_snap_mode or "nearest").strip()
    if normalized_snap_mode not in {"nearest", "ceil", "floor"}:
        normalized_snap_mode = "nearest"
    scale_x, scale_y = _validate_uniform_tile_scale(int(source_w), int(source_h), int(target_w), int(target_h))
    max_tile_pixels = max(0, int(max_tile_pixels))
    if len(x_edges) != 3 or len(y_edges) != 3:
        raise ValueError("2x2 tile manifest requires exactly three x_edges and three y_edges.")
    x_edges = [max(0, min(int(source_w), int(value))) for value in x_edges]
    y_edges = [max(0, min(int(source_h), int(value))) for value in y_edges]
    x_edges[0], x_edges[-1] = 0, int(source_w)
    y_edges[0], y_edges[-1] = 0, int(source_h)
    if not (x_edges[0] < x_edges[1] < x_edges[2]) or not (y_edges[0] < y_edges[1] < y_edges[2]):
        raise ValueError(f"Invalid tile edges: x_edges={x_edges}, y_edges={y_edges}")

    tiles: list[dict[str, Any]] = []
    oversized_tiles: list[dict[str, Any]] = []
    for row in range(rows):
        core_y0 = int(y_edges[row])
        core_y1 = int(y_edges[row + 1])
        core_h = max(1, core_y1 - core_y0)
        overlap_y = int(math.ceil(core_h * overlap_ratio))
        overlap_top = int(overlap_y if row > 0 else 0)
        overlap_bottom = int(overlap_y if row < rows - 1 else 0)
        crop_y0 = max(0, core_y0 - overlap_top)
        crop_y1 = min(int(source_h), core_y1 + overlap_bottom)
        for col in range(cols):
            core_x0 = int(x_edges[col])
            core_x1 = int(x_edges[col + 1])
            core_w = max(1, core_x1 - core_x0)
            overlap_x = int(math.ceil(core_w * overlap_ratio))
            overlap_left = int(overlap_x if col > 0 else 0)
            overlap_right = int(overlap_x if col < cols - 1 else 0)
            edge_overlaps = {
                "left": overlap_left,
                "right": overlap_right,
                "top": overlap_top,
                "bottom": overlap_bottom,
            }
            crop_x0 = max(0, core_x0 - overlap_left)
            crop_x1 = min(int(source_w), core_x1 + overlap_right)
            target_crop_bbox = [
                int(round(crop_x0 * scale_x)),
                int(round(crop_y0 * scale_y)),
                int(round(crop_x1 * scale_x)),
                int(round(crop_y1 * scale_y)),
            ]
            target_core_bbox = [
                int(round(core_x0 * scale_x)),
                int(round(core_y0 * scale_y)),
                int(round(core_x1 * scale_x)),
                int(round(core_y1 * scale_y)),
            ]
            target_crop_w = max(1, target_crop_bbox[2] - target_crop_bbox[0])
            target_crop_h = max(1, target_crop_bbox[3] - target_crop_bbox[1])
            source_crop_w = max(1, crop_x1 - crop_x0)
            source_crop_h = max(1, crop_y1 - crop_y0)
            tile_generate_w = int(_align_to_step(target_crop_w, tile_align, normalized_snap_mode))
            tile_generate_h = int(_align_to_step(target_crop_h, tile_align, normalized_snap_mode))
            tile_generate_pixels = int(tile_generate_w * tile_generate_h)
            if bool(enforce_tile_pixel_limit) and max_tile_pixels > 0 and tile_generate_pixels > max_tile_pixels:
                oversized_tiles.append(
                    {
                        "tile_number": len(tiles) + 1,
                        "tile_generate_size": [int(tile_generate_w), int(tile_generate_h)],
                        "tile_generate_pixels": int(tile_generate_pixels),
                    }
                )
            tiles.append(
                {
                    "index": len(tiles),
                    "tile_number": len(tiles) + 1,
                    "row": int(row),
                    "col": int(col),
                    "source_core_bbox": [int(core_x0), int(core_y0), int(core_x1), int(core_y1)],
                    "source_crop_bbox": [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)],
                    "source_core_size": [int(core_w), int(core_h)],
                    "source_crop_size": [int(source_crop_w), int(source_crop_h)],
                    "target_core_bbox": target_core_bbox,
                    "target_crop_bbox": target_crop_bbox,
                    "target_crop_size": [int(target_crop_w), int(target_crop_h)],
                    "tile_generate_size": [int(tile_generate_w), int(tile_generate_h)],
                    "tile_generate_pixels": int(tile_generate_pixels),
                    "tile_repaint_scale": [
                        float(tile_generate_w / max(1, source_crop_w)),
                        float(tile_generate_h / max(1, source_crop_h)),
                    ],
                    "target_composite_scale": [
                        float(target_crop_w / max(1, source_crop_w)),
                        float(target_crop_h / max(1, source_crop_h)),
                    ],
                    "tile_to_target_scale": [
                        float(target_crop_w / max(1, tile_generate_w)),
                        float(target_crop_h / max(1, tile_generate_h)),
                    ],
                    "aspect_ratio": float(tile_generate_w / max(1, tile_generate_h)),
                    "within_tile_pixel_limit": bool(max_tile_pixels <= 0 or tile_generate_pixels <= max_tile_pixels),
                    "overlap_px_source": [int(overlap_x), int(overlap_y)],
                    "overlap_edges_px_source": edge_overlaps,
                }
            )

    if oversized_tiles:
        details = ", ".join(
            f"tile {item['tile_number']}={item['tile_generate_size'][0]}x{item['tile_generate_size'][1]}"
            f" ({item['tile_generate_pixels']} px)"
            for item in oversized_tiles
        )
        raise ValueError(
            f"Tile repaint resolution exceeds max_tile_pixels={max_tile_pixels}: {details}. "
            "Move the split lines, reduce overlap_ratio, reduce scale_factor/output size, or raise max_tile_pixels."
        )

    manifest = {
        "version": 1,
        "mode": str(mode),
        "source_shape": _shape(source_video),
        "source_size": [int(source_w), int(source_h)],
        "target_size": [int(target_w), int(target_h)],
        "rows": rows,
        "cols": cols,
        "tile_count": len(tiles),
        "scale_factor": float(target_w / max(1, int(source_w))),
        "scale": [float(scale_x), float(scale_y)],
        "overlap_ratio": float(overlap_ratio),
        "tile_align": int(tile_align),
        "resolution_snap_mode": normalized_snap_mode,
        "default_feather_px": int(feather_px),
        "tile_output_constraints": {
            "max_tile_pixels": int(max_tile_pixels),
            "max_tile_reference_resolution": "1280x720 pixels by default as total pixel budget",
            "enforce_tile_pixel_limit": bool(enforce_tile_pixel_limit),
            "uniform_final_scale_required": True,
        },
        "channels": int(channels),
        "split_plan": {
            "x_edges": [int(value) for value in x_edges],
            "y_edges": [int(value) for value in y_edges],
        },
        "tiles": tiles,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _build_rect_tile_manifest(
    source_video: torch.Tensor,
    target_w: int,
    target_h: int,
    overlap_ratio: float,
    tile_align: int,
    feather_px: int,
    core_bboxes: list[list[int]],
    *,
    mode: str,
    max_tile_pixels: int = DEFAULT_MAX_TILE_PIXELS,
    enforce_tile_pixel_limit: bool = True,
    resolution_snap_mode: str = "nearest",
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    _frames, source_h, source_w, channels = source_video.shape
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    overlap_ratio = max(0.0, min(0.45, float(overlap_ratio)))
    tile_align = max(1, int(tile_align))
    normalized_snap_mode = str(resolution_snap_mode or "nearest").strip()
    if normalized_snap_mode not in {"nearest", "ceil", "floor"}:
        normalized_snap_mode = "nearest"
    scale_x, scale_y = _validate_uniform_tile_scale(int(source_w), int(source_h), int(target_w), int(target_h))
    max_tile_pixels = max(0, int(max_tile_pixels))
    if not core_bboxes:
        raise ValueError("manual tile layout produced no tiles.")
    if len(core_bboxes) > MAX_TILES:
        raise ValueError(f"manual tile layout supports at most {MAX_TILES} tiles.")

    normalized_core_bboxes = [
        list(_clamp_bbox(tuple(int(value) for value in bbox), int(source_w), int(source_h)))
        for bbox in core_bboxes
    ]
    tiles: list[dict[str, Any]] = []
    oversized_tiles: list[dict[str, Any]] = []
    for tile_index, bbox in enumerate(normalized_core_bboxes):
        core_x0, core_y0, core_x1, core_y1 = [int(value) for value in bbox]
        core_w = max(1, core_x1 - core_x0)
        core_h = max(1, core_y1 - core_y0)
        overlap_x = int(math.ceil(core_w * overlap_ratio))
        overlap_y = int(math.ceil(core_h * overlap_ratio))
        edge_overlaps = _manual_neighbor_overlap_edges(normalized_core_bboxes, tile_index, overlap_x, overlap_y)
        crop_x0 = max(0, core_x0 - edge_overlaps["left"])
        crop_y0 = max(0, core_y0 - edge_overlaps["top"])
        crop_x1 = min(int(source_w), core_x1 + edge_overlaps["right"])
        crop_y1 = min(int(source_h), core_y1 + edge_overlaps["bottom"])
        target_crop_bbox = [
            int(round(crop_x0 * scale_x)),
            int(round(crop_y0 * scale_y)),
            int(round(crop_x1 * scale_x)),
            int(round(crop_y1 * scale_y)),
        ]
        target_core_bbox = [
            int(round(core_x0 * scale_x)),
            int(round(core_y0 * scale_y)),
            int(round(core_x1 * scale_x)),
            int(round(core_y1 * scale_y)),
        ]
        target_crop_w = max(1, target_crop_bbox[2] - target_crop_bbox[0])
        target_crop_h = max(1, target_crop_bbox[3] - target_crop_bbox[1])
        source_crop_w = max(1, crop_x1 - crop_x0)
        source_crop_h = max(1, crop_y1 - crop_y0)
        tile_generate_w = int(_align_to_step(target_crop_w, tile_align, normalized_snap_mode))
        tile_generate_h = int(_align_to_step(target_crop_h, tile_align, normalized_snap_mode))
        tile_generate_pixels = int(tile_generate_w * tile_generate_h)
        if bool(enforce_tile_pixel_limit) and max_tile_pixels > 0 and tile_generate_pixels > max_tile_pixels:
            oversized_tiles.append(
                {
                    "tile_number": tile_index + 1,
                    "tile_generate_size": [int(tile_generate_w), int(tile_generate_h)],
                    "tile_generate_pixels": int(tile_generate_pixels),
                }
            )
        tiles.append(
            {
                "index": int(tile_index),
                "tile_number": int(tile_index + 1),
                "row": None,
                "col": None,
                "source_core_bbox": [int(core_x0), int(core_y0), int(core_x1), int(core_y1)],
                "source_crop_bbox": [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)],
                "source_core_size": [int(core_w), int(core_h)],
                "source_crop_size": [int(source_crop_w), int(source_crop_h)],
                "target_core_bbox": target_core_bbox,
                "target_crop_bbox": target_crop_bbox,
                "target_crop_size": [int(target_crop_w), int(target_crop_h)],
                "tile_generate_size": [int(tile_generate_w), int(tile_generate_h)],
                "tile_generate_pixels": int(tile_generate_pixels),
                "tile_repaint_scale": [
                    float(tile_generate_w / max(1, source_crop_w)),
                    float(tile_generate_h / max(1, source_crop_h)),
                ],
                "target_composite_scale": [
                    float(target_crop_w / max(1, source_crop_w)),
                    float(target_crop_h / max(1, source_crop_h)),
                ],
                "tile_to_target_scale": [
                    float(target_crop_w / max(1, tile_generate_w)),
                    float(target_crop_h / max(1, tile_generate_h)),
                ],
                "aspect_ratio": float(tile_generate_w / max(1, tile_generate_h)),
                "within_tile_pixel_limit": bool(max_tile_pixels <= 0 or tile_generate_pixels <= max_tile_pixels),
                "overlap_px_source": [int(overlap_x), int(overlap_y)],
                "overlap_edges_px_source": edge_overlaps,
            }
        )

    if oversized_tiles:
        details = ", ".join(
            f"tile {item['tile_number']}={item['tile_generate_size'][0]}x{item['tile_generate_size'][1]}"
            f" ({item['tile_generate_pixels']} px)"
            for item in oversized_tiles
        )
        raise ValueError(
            f"Tile repaint resolution exceeds max_tile_pixels={max_tile_pixels}: {details}. "
            "Resize the tile, reduce overlap_ratio, reduce scale_factor/output size, or raise max_tile_pixels."
        )

    manifest = {
        "version": 1,
        "mode": str(mode),
        "source_shape": _shape(source_video),
        "source_size": [int(source_w), int(source_h)],
        "target_size": [int(target_w), int(target_h)],
        "rows": None,
        "cols": None,
        "tile_count": len(tiles),
        "scale_factor": float(target_w / max(1, int(source_w))),
        "scale": [float(scale_x), float(scale_y)],
        "overlap_ratio": float(overlap_ratio),
        "tile_align": int(tile_align),
        "resolution_snap_mode": normalized_snap_mode,
        "default_feather_px": int(feather_px),
        "tile_output_constraints": {
            "max_tile_pixels": int(max_tile_pixels),
            "max_tile_reference_resolution": "1280x720 pixels by default as total pixel budget",
            "enforce_tile_pixel_limit": bool(enforce_tile_pixel_limit),
            "uniform_final_scale_required": True,
            "max_tiles": int(MAX_TILES),
        },
        "channels": int(channels),
        "split_plan": {
            "manual_rectangles": [tile["source_core_bbox"] for tile in tiles],
        },
        "tiles": tiles,
    }
    if extra:
        manifest.update(extra)
    return manifest


def _save_manual_tile_preview_frames(
    video: torch.Tensor,
    manifest: dict[str, Any],
    filename_prefix: str,
    preview_frame_count: int,
) -> dict[str, Any]:
    import os

    import numpy as np
    from PIL import Image

    try:
        import folder_paths
    except Exception:
        return {
            "items": [],
            "reason": "folder_paths_unavailable",
            "source_shape": _shape(video),
        }

    if not isinstance(video, torch.Tensor) or video.ndim != 4 or int(video.shape[0]) <= 0:
        return {"items": [], "reason": "invalid_video", "source_shape": _shape(video)}
    frame_count = int(video.shape[0])
    wanted = max(1, min(24, int(preview_frame_count), frame_count))
    if wanted == 1:
        indices = [0]
    else:
        indices = [int(round(index * (frame_count - 1) / (wanted - 1))) for index in range(wanted)]

    base_dir = folder_paths.get_temp_directory()
    subfolder = "scail_manual_tile_preview"
    target_dir = os.path.join(base_dir, subfolder)
    os.makedirs(target_dir, exist_ok=True)
    prefix = _matrix_safe_filename_part(filename_prefix or "scail_manual_tile")
    fingerprint = _stable_fingerprint(
        {
            "prefix": prefix,
            "shape": _shape(video),
            "indices": indices,
            "manual_layout": manifest.get("manual_layout"),
            "target_size": manifest.get("target_size"),
        }
    )[:8]

    items: list[dict[str, Any]] = []
    for item_index, frame_index in enumerate(indices):
        frame = video[frame_index].detach().cpu().float().clamp(0, 1)
        if int(frame.shape[-1]) > 3:
            frame = frame[..., :3]
        if int(frame.shape[-1]) < 3:
            frame = frame[..., :1].repeat(1, 1, 3)
        array = (frame.numpy() * 255.0).round().astype(np.uint8)
        image = Image.fromarray(array, mode="RGB")
        filename = f"{prefix}_{fingerprint}_{item_index:03d}_frame{frame_index + 1}.png"
        image.save(os.path.join(target_dir, filename))
        items.append(
            {
                "index": int(item_index),
                "frame_0_based": int(frame_index),
                "frame_1_based": int(frame_index + 1),
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp",
                "label": f"frame {frame_index + 1}",
            }
        )

    return {
        "items": items,
        "count": len(items),
        "source_shape": _shape(video),
        "source_size": manifest.get("source_size"),
        "manual_layout": manifest.get("manual_layout"),
        "split_plan": manifest.get("split_plan"),
        "target_size": manifest.get("target_size"),
    }


def _crop_resize_image_tensor(
    image: torch.Tensor,
    source_bbox: list[int],
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("image must be a ComfyUI IMAGE tensor.")
    _, height, width, _channels = image.shape
    x0, y0, x1, y1 = _clamp_bbox(tuple(int(value) for value in source_bbox), int(width), int(height))
    cropped = image[:, y0:y1, x0:x1, :].detach().cpu().float().clamp(0, 1).contiguous()
    return _resize_image_batch(cropped, int(target_height), int(target_width), mode=mode).contiguous().clamp(0, 1)


def _scaled_bbox_for_image(source_bbox: list[int], source_size: list[int], image: torch.Tensor) -> list[int]:
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("image must be a ComfyUI IMAGE tensor.")
    source_w, source_h = int(source_size[0]), int(source_size[1])
    image_h, image_w = int(image.shape[1]), int(image.shape[2])
    x0, y0, x1, y1 = [int(value) for value in source_bbox]
    return [
        int(round(x0 * image_w / max(1, source_w))),
        int(round(y0 * image_h / max(1, source_h))),
        int(round(x1 * image_w / max(1, source_w))),
        int(round(y1 * image_h / max(1, source_h))),
    ]


def _tile_tensor_crop_bbox(source_bbox: list[int], source_size: list[int], image: torch.Tensor) -> list[int]:
    bbox = _scaled_bbox_for_image(source_bbox, source_size, image)
    return list(_clamp_bbox(tuple(int(value) for value in bbox), int(image.shape[2]), int(image.shape[1])))


def _blank_image_like_tile(tile_video: torch.Tensor, frames: int = 1) -> torch.Tensor:
    return torch.zeros(
        (max(1, int(frames)), int(tile_video.shape[1]), int(tile_video.shape[2]), int(tile_video.shape[-1])),
        dtype=tile_video.dtype,
        device=tile_video.device,
    )


def _tile_weight_mask_core_feather(height: int, width: int, crop_bbox: list[int], core_bbox: list[int], feather_px: int) -> torch.Tensor:
    height = max(1, int(height))
    width = max(1, int(width))
    feather_px = max(0, int(feather_px))
    x0, y0, _x1, _y1 = [int(value) for value in crop_bbox]
    cx0, cy0, cx1, cy1 = [int(value) for value in core_bbox]
    local_core = [
        max(0, min(width, cx0 - x0)),
        max(0, min(height, cy0 - y0)),
        max(0, min(width, cx1 - x0)),
        max(0, min(height, cy1 - y0)),
    ]
    lx0, ly0, lx1, ly1 = local_core
    core = torch.zeros((height, width), dtype=torch.float32)
    core[ly0:ly1, lx0:lx1] = 1.0
    if feather_px <= 0:
        return core.clamp(0, 1)
    return _binary_mask_morph(core.unsqueeze(0), blur_px=feather_px)[0].clamp(0, 1)


def _tile_weight_mask_ttp_seam(height: int, width: int, crop_bbox: list[int], core_bbox: list[int], feather_px: int) -> torch.Tensor:
    height = max(1, int(height))
    width = max(1, int(width))
    feather_px = max(0, int(feather_px))
    x0, y0, _x1, _y1 = [int(value) for value in crop_bbox]
    cx0, cy0, cx1, cy1 = [int(value) for value in core_bbox]
    lx0 = max(0, min(width, cx0 - x0))
    ly0 = max(0, min(height, cy0 - y0))
    lx1 = max(0, min(width, cx1 - x0))
    ly1 = max(0, min(height, cy1 - y0))
    if feather_px <= 0:
        core = torch.zeros((height, width), dtype=torch.float32)
        core[ly0:ly1, lx0:lx1] = 1.0
        return core.clamp(0, 1)

    def axis_weights(size: int, start: int, end: int) -> torch.Tensor:
        coords = torch.arange(int(size), dtype=torch.float32) + 0.5
        weights = torch.ones((int(size),), dtype=torch.float32)

        left_context = max(0, int(start))
        if left_context > 0:
            blend = max(1.0, float(min(int(feather_px), left_context * 2)))
            ramp_start = float(start) - blend / 2.0
            ramp_end = float(start) + blend / 2.0
            left_ramp = ((coords - ramp_start) / max(1e-6, ramp_end - ramp_start)).clamp(0, 1)
            weights = torch.minimum(weights, left_ramp)

        right_context = max(0, int(size) - int(end))
        if right_context > 0:
            blend = max(1.0, float(min(int(feather_px), right_context * 2)))
            ramp_start = float(end) - blend / 2.0
            ramp_end = float(end) + blend / 2.0
            right_ramp = (1.0 - ((coords - ramp_start) / max(1e-6, ramp_end - ramp_start))).clamp(0, 1)
            weights = torch.minimum(weights, right_ramp)

        return weights.clamp(0, 1)

    x_weights = axis_weights(width, lx0, lx1).view(1, width)
    y_weights = axis_weights(height, ly0, ly1).view(height, 1)
    return (y_weights * x_weights).clamp(0, 1)


def _tile_weight_mask(
    height: int,
    width: int,
    crop_bbox: list[int],
    core_bbox: list[int],
    feather_px: int,
    blend_mode: str = "core_feather",
) -> torch.Tensor:
    normalized = str(blend_mode or "core_feather").strip()
    if normalized == "ttp_seam":
        return _tile_weight_mask_ttp_seam(height, width, crop_bbox, core_bbox, feather_px)
    return _tile_weight_mask_core_feather(height, width, crop_bbox, core_bbox, feather_px)


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
                "free_tail_window": _free_tail_window_input_spec(),
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
        free_tail_window: bool = False,
        pose_video_mask=None,
        prompt=None,
        unique_id=None,
        status_unique_id=None,
        status_prefix: str = "",
        **kwargs,
    ):
        if not isinstance(pose_video, torch.Tensor) or pose_video.ndim != 4:
            raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
        free_tail_window = bool(free_tail_window)
        status_id = unique_id if status_unique_id is None else status_unique_id
        status_prefix_text = str(status_prefix or "")

        def send_status(stage: str, message: str, *, progress: Optional[dict[str, Any]] = None, **extra: Any) -> None:
            _send_long_video_status(
                status_id,
                "SCAIL2ScheduledLongVideo",
                stage,
                f"{status_prefix_text}{message}",
                progress=progress,
                **extra,
            )

        send_status("preparing", "Preparing long video")

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
                "free_tail_window": bool(free_tail_window),
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
            send_status("cache_hit", "Cache hit; using previous result", progress={"current": 1, "total": 1})
            return _clone_cached_result(cached_result)
        if use_disk_cache:
            disk_result = _load_single_slot_disk_cache("SCAIL2ScheduledLongVideo", unique_id, call_cache_key)
            if disk_result is not None:
                print("[SCAIL2ScheduledLongVideo] disk cache hit; returning previous result without sampling.")
                send_status("cache_hit", "Disk cache hit; loading previous result", progress={"current": 1, "total": 1})
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
        send_status(
            "planning",
            f"Planning {planned_frames} frame(s) across {len(planned_chunks)} chunk(s)",
            progress={"current": 0, "total": max(1, len(planned_chunks))},
        )
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
            send_status("validating_masks", "Validating replacement masks")
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
        for ref_position, index in enumerate(used_refs, start=1):
            send_status(
                "encoding_references",
                f"Encoding reference {ref_position}/{len(used_refs)}",
                progress={"current": ref_position, "total": max(1, len(used_refs))},
                reference=int(index),
            )
            get_reference(index)
        for prompt_position, (prompt, negative) in enumerate(prompt_pairs, start=1):
            send_status(
                "encoding_prompts",
                f"Encoding prompt pair {prompt_position}/{len(prompt_pairs)}",
                progress={"current": prompt_position, "total": max(1, len(prompt_pairs))},
            )
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
        free_tail_window_summary: dict[str, Any] = {
            "enabled": bool(free_tail_window),
            "applied": False,
            "strategy": "blank_conditioning_tail_in_final_window",
            "forced_pre_tail_split": False,
            "conditioning_tail_frames": 0,
            "decoded_tail_frames_discarded": 0,
            "events": [],
        }

        for segment_position, segment in enumerate(segments):
            remaining = int(segment["frames"])
            segment_kept = 0
            segment_reference = int(segment["reference"])
            is_final_segment = segment_position == len(segments) - 1
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
                forced_pre_tail_split = False
                if free_tail_window and is_final_segment and wanted_keep >= remaining and remaining > 1:
                    projected_raw_length = int(wanted_keep) if not has_previous else int(wanted_keep) + int(actual_overlap)
                    projected_length = _ceil_to_4n_plus_1(projected_raw_length)
                    if projected_length < 17 and remaining > 1:
                        projected_length = 17
                    projected_length = min(int(projected_length), int(max_chunk_frames))
                    projected_tail_frames = max(0, int(projected_length) - int(projected_raw_length))
                    if projected_tail_frames <= 0:
                        preferred_tail_keep = int(actual_overlap) if actual_overlap > 0 else (int(overlap) if overlap > 0 else 5)
                        tail_real_keep = min(int(remaining) - 1, max(1, int(preferred_tail_keep)))
                        if tail_real_keep > 0 and int(remaining) - int(tail_real_keep) > 0:
                            wanted_keep = int(remaining) - int(tail_real_keep)
                            forced_pre_tail_split = True
                            free_tail_window_summary["forced_pre_tail_split"] = True
                raw_length = wanted_keep if not has_previous else wanted_keep + actual_overlap
                length = _ceil_to_4n_plus_1(raw_length)
                if length < 17 and remaining > 1:
                    length = 17
                if length > max_chunk_frames:
                    length = max_chunk_frames
                    wanted_keep = length if not has_previous else length - actual_overlap
                if wanted_keep <= 0:
                    raise RuntimeError("overlap_frames leaves no room for new frames.")
                raw_length = wanted_keep if not has_previous else wanted_keep + actual_overlap
                is_final_output_chunk = bool(free_tail_window) and is_final_segment and int(wanted_keep) >= int(remaining)
                if is_final_output_chunk and int(length) < int(max_chunk_frames):
                    length = int(max_chunk_frames)
                tail_window_padding_frames = max(0, int(length) - int(raw_length))
                tail_window_applied = bool(is_final_output_chunk and tail_window_padding_frames > 0)
                video_frame_offset = int(produced)
                internal_window_offset = max(0, int(produced) - int(actual_overlap))
                print(
                    "[SCAIL2ScheduledLongVideo] run "
                    f"chunk={chunk_index} segment={segment['index']} ref={segment['reference']} "
                    f"gen={length} discard={'pending'} keep_target={wanted_keep} "
                    f"offset_input={video_frame_offset} internal_window_offset={internal_window_offset} "
                    f"produced_before={produced} free_tail={tail_window_applied}"
                )
                chunk_progress = {"current": int(chunk_index) + 1, "total": max(1, len(planned_chunks))}
                send_status(
                    "building_conditioning",
                    f"Chunk {chunk_index + 1}/{max(1, len(planned_chunks))}: building conditioning",
                    progress=chunk_progress,
                    chunk_index=int(chunk_index),
                    reference=int(segment["reference"]),
                )
                previous_frames_for_scail = (
                    previous_frames[-actual_overlap:].contiguous()
                    if previous_frames is not None and actual_overlap > 0
                    else None
                )
                scail_pose_video = pose_video
                scail_pose_video_mask = pose_video_mask
                tail_conditioning_frames = 0
                if tail_window_applied:
                    required_conditioning_frames = int(video_frame_offset) + int(length)
                    scail_pose_video = _pad_image_sequence_tail_with_zeros(
                        pose_video[:planned_frames],
                        required_conditioning_frames,
                    )
                    tail_conditioning_frames = max(0, int(required_conditioning_frames) - int(planned_frames))
                    if pose_video_mask is not None:
                        scail_pose_video_mask = _pad_image_sequence_tail_with_zeros(
                            pose_video_mask[:planned_frames],
                            required_conditioning_frames,
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
                        pose_video=scail_pose_video,
                        pose_video_mask=scail_pose_video_mask,
                        reference_image_mask=ref_info["reference_image_mask"],
                        previous_frames=previous_frames_for_scail,
                    )
                )
                if len(scail_out) != 4:
                    raise RuntimeError("WanSCAILToVideo returned an unexpected result.")
                chunk_positive, chunk_negative, latent, _next_offset = scail_out
                send_status(
                    "sampling",
                    f"Chunk {chunk_index + 1}/{max(1, len(planned_chunks))}: sampling",
                    progress=chunk_progress,
                    chunk_index=int(chunk_index),
                )
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
                send_status(
                    "decoding",
                    f"Chunk {chunk_index + 1}/{max(1, len(planned_chunks))}: decoding",
                    progress=chunk_progress,
                    chunk_index=int(chunk_index),
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
                    pose_mask_source = scail_pose_video_mask if tail_window_applied and scail_pose_video_mask is not None else pose_video_mask
                    pose_mask_window = pose_mask_source[
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
                decoded_tail_frames = max(0, int(decoded.shape[0]) - int(discard_head) - int(kept.shape[0]))
                if tail_window_applied:
                    free_tail_event = {
                        "chunk_index": int(chunk_index),
                        "segment_index": int(segment["index"]),
                        "generate_length": int(length),
                        "raw_conditioned_length": int(raw_length),
                        "kept_frames": int(kept.shape[0]),
                        "conditioning_tail_frames": int(tail_conditioning_frames),
                        "decoded_tail_frames_discarded": int(decoded_tail_frames),
                    }
                    free_tail_window_summary["applied"] = True
                    free_tail_window_summary["conditioning_tail_frames"] += int(tail_conditioning_frames)
                    free_tail_window_summary["decoded_tail_frames_discarded"] += int(decoded_tail_frames)
                    free_tail_window_summary["events"].append(free_tail_event)
                if overlap > 0:
                    previous_frames = torch.cat(stitched, dim=0)[-overlap:].contiguous()
                else:
                    previous_frames = None
                send_status(
                    "stitching_chunks",
                    f"Chunk {chunk_index + 1}: stitched {produced}/{planned_frames} frame(s)",
                    progress={"current": int(produced), "total": max(1, int(planned_frames))},
                    chunk_index=int(chunk_index),
                )

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
                        "output_end": int(produced),
                        "generate_length": int(length),
                        "discard_head": int(discard_head),
                        "kept_frames": int(kept.shape[0]),
                        "produced_total": int(produced),
                        "free_tail_window": {
                            "forced_pre_tail_split": bool(forced_pre_tail_split),
                            "applied": bool(tail_window_applied),
                            "conditioning_tail_frames": int(tail_conditioning_frames),
                            "decoded_tail_frames_discarded": int(decoded_tail_frames),
                        },
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
        output_frames_before_tail_trim = int(frames.shape[0])
        if int(frames.shape[0]) > int(planned_frames):
            frames = frames[:planned_frames].contiguous()
            used_pose_video_mask = used_pose_video_mask[:planned_frames].contiguous()
            used_reference_mask_timeline = used_reference_mask_timeline[:planned_frames].contiguous()
        free_tail_window_summary["output_frames_before_trim"] = int(output_frames_before_tail_trim)
        free_tail_window_summary["output_frames"] = int(frames.shape[0])
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
            "free_tail_window": free_tail_window_summary,
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
        send_status("done", f"Done: {int(frames.shape[0])} frame(s)", progress={"current": 1, "total": 1})
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
                "free_tail_window": _free_tail_window_input_spec(),
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
        cache_mode: str = "disk",
        free_tail_window: bool = False,
        object_indices: str = "",
        reference_object_indices: str = "",
        sort_by: str = "left_to_right",
        sam_detection_threshold: float = 0.5,
        sam_max_objects: int = 2,
        sam_detect_interval: int = 2,
        sam_model=None,
        sam_conditioning=None,
        prompt=None,
        unique_id=None,
        status_unique_id=None,
        status_prefix: str = "",
        **kwargs,
    ):
        if not isinstance(pose_video, torch.Tensor) or pose_video.ndim != 4:
            raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
        status_id = unique_id if status_unique_id is None else status_unique_id
        status_prefix_text = str(status_prefix or "")

        def send_status(stage: str, message: str, *, progress: Optional[dict[str, Any]] = None, **extra: Any) -> None:
            _send_long_video_status(
                status_id,
                "SCAIL2ScheduledLongVideoWithSAM",
                stage,
                f"{status_prefix_text}{message}",
                progress=progress,
                **extra,
            )

        send_status("preparing", "Preparing internal SAM long video")

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
                "free_tail_window": bool(free_tail_window),
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
            send_status("cache_hit", "Cache hit; using previous result", progress={"current": 1, "total": 1})
            return _clone_cached_result(cached_result)
        if use_disk_cache:
            disk_result = _load_single_slot_disk_cache("SCAIL2ScheduledLongVideoWithSAM", unique_id, call_cache_key)
            if disk_result is not None:
                print(
                    "[SCAIL2ScheduledLongVideoWithSAM] disk cache hit; "
                    "returning previous result without SAM tracking/sampling."
                )
                send_status("cache_hit", "Disk cache hit; loading previous result", progress={"current": 1, "total": 1})
                self._last_internal_sam_cache_key = call_cache_key
                self._last_internal_sam_cache_result = _clone_cached_result(disk_result)
                return _clone_cached_result(disk_result)

        if not replacement_mode:
            send_status("skipping_sam", "Animation mode: skipping SAM masks")
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
                free_tail_window=bool(free_tail_window),
                prompt=prompt,
                unique_id=unique_id,
                status_unique_id=status_id,
                status_prefix=status_prefix_text,
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
            send_status("done", "Done", progress={"current": 1, "total": 1})
            return result

        if sam_model is None or sam_conditioning is None:
            raise ValueError("replacement mode requires sam_model and sam_conditioning.")

        print(
            "[SCAIL2ScheduledLongVideoWithSAM] "
            f"tracking pose_video frames={int(pose_video.shape[0])} refs={used_refs} "
            f"object_indices='{object_indices}' reference_object_indices='{reference_object_indices}' "
            f"sort_by={sort_by}"
        )
        send_status("running_sam_pose", "Running SAM on pose video")
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
        for ref_position, index in enumerate(used_refs, start=1):
            send_status(
                "running_sam_reference",
                f"Running SAM on reference {ref_position}/{len(used_refs)}",
                progress={"current": ref_position, "total": max(1, len(used_refs))},
                reference=int(index),
            )
            reference_track_data = _run_sam3_track(
                references[index],
                sam_model,
                sam_conditioning,
                detection_threshold=float(sam_detection_threshold),
                max_objects=int(sam_max_objects),
                detect_interval=1,
            )

            send_status(
                "building_masks",
                f"Building replacement masks for reference {ref_position}/{len(used_refs)}",
                progress={"current": ref_position, "total": max(1, len(used_refs))},
                reference=int(index),
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
            free_tail_window=bool(free_tail_window),
            prompt=prompt,
            unique_id=unique_id,
            status_unique_id=status_id,
            status_prefix=status_prefix_text,
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
        send_status("done", "Done", progress={"current": 1, "total": 1})
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
                "window_fit_mode": (
                    ["shift_inside_reference", "strict_alignment"],
                    {"default": "shift_inside_reference"},
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
        window_fit_mode: str = "shift_inside_reference",
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
            str(window_fit_mode),
        )
        aligned_reference = _rgb_uint8_to_image_tensor(aligned_rgb)
        actual_window_x0 = int(window_info["window_xyxy"][0])
        actual_window_y0 = int(window_info["window_xyxy"][1])

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
            int(round((float(ref_bbox[0]) - float(actual_window_x0)) * sx)),
            int(round((float(ref_bbox[1]) - float(actual_window_y0)) * sy)),
            int(round((float(ref_bbox[2]) - float(actual_window_x0)) * sx)),
            int(round((float(ref_bbox[3]) - float(actual_window_y0)) * sy)),
        )
        aligned_marked = aligned_preview.clone()
        _draw_rect(aligned_marked, aligned_bbox, (0.95, 0.25, 0.15))
        debug_preview = torch.cat([target_marked, aligned_marked], dim=2).contiguous().clamp(0, 1)

        aligned_face_bbox = [
            float(ref_bbox[0]) - float(actual_window_x0),
            float(ref_bbox[1]) - float(actual_window_y0),
            float(ref_bbox[2]) - float(actual_window_x0),
            float(ref_bbox[3]) - float(actual_window_y0),
        ]
        summary = {
            "method": "face_detector_bbox_ratio_crop_fit_no_resize",
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
            "window_fit_mode": str(window_info.get("fit_mode", window_fit_mode)),
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


class SCAIL2TilePlanBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE",),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "overlap_ratio": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.45, "step": 0.01}),
                "tile_align": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "resolution_snap_mode": (["nearest", "ceil", "floor"], {"default": "nearest"}),
                "feather_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "min_tile_ratio": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.45, "step": 0.01}),
                "protected_padding_ratio": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.05}),
                "protected_padding_px": ("INT", {"default": 12, "min": 0, "max": 512, "step": 1}),
                "max_tile_pixels": ("INT", {"default": DEFAULT_MAX_TILE_PIXELS, "min": 0, "max": 4096 * 4096, "step": 1024}),
                "enforce_tile_pixel_limit": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "protected_masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("tile_manifest", "debug_preview", "tile_resolution_report")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/Tile Upscale"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def build(
        self,
        source_video: torch.Tensor,
        output_width: int = 0,
        output_height: int = 0,
        scale_factor: float = 2.0,
        overlap_ratio: float = 0.10,
        tile_align: int = 32,
        resolution_snap_mode: str = "nearest",
        feather_px: int = 48,
        min_tile_ratio: float = 0.30,
        protected_padding_ratio: float = 0.45,
        protected_padding_px: int = 12,
        max_tile_pixels: int = DEFAULT_MAX_TILE_PIXELS,
        enforce_tile_pixel_limit: bool = True,
        protected_masks: Optional[torch.Tensor] = None,
    ):
        if not isinstance(source_video, torch.Tensor) or source_video.ndim != 4:
            raise ValueError("source_video must be a ComfyUI IMAGE tensor.")
        _frames, source_h, source_w, channels = source_video.shape
        rows = 2
        cols = 2
        target_w, target_h, target_size_adjustment = _resolve_tile_target_size(
            int(source_w),
            int(source_h),
            int(output_width),
            int(output_height),
            float(scale_factor),
        )
        target_w = max(cols, int(target_w))
        target_h = max(rows, int(target_h))
        overlap_ratio = max(0.0, min(0.45, float(overlap_ratio)))
        tile_align = max(1, int(tile_align))
        scale_x = target_w / max(1, int(source_w))
        scale_y = target_h / max(1, int(source_h))
        protected_region = None
        if protected_masks is not None:
            protected_region = _protected_bbox_from_masks(
                protected_masks,
                int(source_video.shape[0]),
                int(source_h),
                int(source_w),
                float(protected_padding_ratio),
                int(protected_padding_px),
            )
        protected_bbox = protected_region.get("source_bbox") if isinstance(protected_region, dict) else None
        split_x_plan = _choose_protected_split(
            int(source_w),
            (int(protected_bbox[0]), int(protected_bbox[2])) if isinstance(protected_bbox, list) and len(protected_bbox) == 4 else None,
            float(min_tile_ratio),
        )
        split_y_plan = _choose_protected_split(
            int(source_h),
            (int(protected_bbox[1]), int(protected_bbox[3])) if isinstance(protected_bbox, list) and len(protected_bbox) == 4 else None,
            float(min_tile_ratio),
        )
        x_edges = [0, int(split_x_plan["position"]), int(source_w)]
        y_edges = [0, int(split_y_plan["position"]), int(source_h)]

        manifest = _build_2x2_tile_manifest(
            source_video,
            int(target_w),
            int(target_h),
            float(overlap_ratio),
            int(tile_align),
            int(feather_px),
            x_edges,
            y_edges,
            mode="2x2_face_safe_spatial_tile_upscale" if protected_region is not None else "2x2_spatial_tile_upscale",
            max_tile_pixels=int(max_tile_pixels),
            enforce_tile_pixel_limit=bool(enforce_tile_pixel_limit),
            resolution_snap_mode=str(resolution_snap_mode),
        )
        tiles = manifest["tiles"]

        protected_owner_tile = None
        if isinstance(protected_bbox, list) and len(protected_bbox) == 4:
            px0, py0, px1, py1 = [int(value) for value in protected_bbox]
            best_area = -1
            for tile in tiles:
                tx0, ty0, tx1, ty1 = [int(value) for value in tile["source_core_bbox"]]
                ix0 = max(px0, tx0)
                iy0 = max(py0, ty0)
                ix1 = min(px1, tx1)
                iy1 = min(py1, ty1)
                area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                tile["protected_core_intersection_area"] = int(area)
                if area > best_area:
                    best_area = area
                    protected_owner_tile = int(tile["tile_number"])
            for tile in tiles:
                tile["is_protected_owner_tile"] = int(tile["tile_number"]) == int(protected_owner_tile)

        manifest.update(
            {
                "scale_factor": float(scale_x),
                "target_size_adjustment": target_size_adjustment,
                "min_tile_ratio": float(min_tile_ratio),
                "protected_region": protected_region,
                "protected_owner_tile": protected_owner_tile,
                "split_plan": {
                    **manifest.get("split_plan", {}),
                    "x": split_x_plan,
                    "y": split_y_plan,
                    "protected_split_safe": bool(
                        split_x_plan.get("avoids_protected_region", True)
                        and split_y_plan.get("avoids_protected_region", True)
                    ),
                },
                "note": (
                    "Use one SCAIL-2 Tile Extractor per tile, then stitch generated tile videos with "
                    "SCAIL-2 Tile Composite Video. For best face quality, run the Face Detail branch after "
                    "the tile composite on the final high-resolution frames."
                ),
            }
        )
        return (json.dumps(manifest, indent=2), _tile_preview_from_manifest(source_video, manifest), _format_tile_resolution_report(manifest))


class SCAIL2ManualTilePlanBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("IMAGE",),
                "layout_json": (
                    "STRING",
                    {
                        "default": '{"split_x":0.5,"split_y":0.5}',
                        "multiline": False,
                        "tooltip": "Written by the visual tile editor. Supports normalized tiles or split_x/split_y fallback.",
                    },
                ),
                "output_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "output_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "overlap_ratio": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 0.45, "step": 0.01}),
                "tile_align": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "resolution_snap_mode": (["nearest", "ceil", "floor"], {"default": "nearest"}),
                "feather_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "min_tile_ratio": ("FLOAT", {"default": 0.20, "min": 0.05, "max": 0.45, "step": 0.01}),
                "max_tile_pixels": ("INT", {"default": DEFAULT_MAX_TILE_PIXELS, "min": 0, "max": 4096 * 4096, "step": 1024}),
                "enforce_tile_pixel_limit": ("BOOLEAN", {"default": True}),
                "preview_frame_count": ("INT", {"default": 8, "min": 1, "max": 24, "step": 1}),
                "preview_filename_prefix": ("STRING", {"default": "scail_manual_tile"}),
                "coverage_policy": (["auto_fill", "error", "ignore"], {"default": "auto_fill"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("tile_manifest", "debug_preview", "tile_resolution_report")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/Tile Upscale"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def build(
        self,
        source_video: torch.Tensor,
        layout_json: str = '{"split_x":0.5,"split_y":0.5}',
        output_width: int = 0,
        output_height: int = 0,
        scale_factor: float = 2.0,
        overlap_ratio: float = 0.10,
        tile_align: int = 32,
        resolution_snap_mode: str = "nearest",
        feather_px: int = 48,
        min_tile_ratio: float = 0.20,
        max_tile_pixels: int = DEFAULT_MAX_TILE_PIXELS,
        enforce_tile_pixel_limit: bool = True,
        preview_frame_count: int = 8,
        preview_filename_prefix: str = "scail_manual_tile",
        coverage_policy: str = "auto_fill",
    ):
        if not isinstance(source_video, torch.Tensor) or source_video.ndim != 4:
            raise ValueError("source_video must be a ComfyUI IMAGE tensor.")
        _frames, source_h, source_w, _channels = source_video.shape
        target_w, target_h, target_size_adjustment = _resolve_tile_target_size(
            int(source_w),
            int(source_h),
            int(output_width),
            int(output_height),
            float(scale_factor),
        )
        layout = _parse_manual_tile_layout(layout_json)
        core_bboxes, normalized_layout = _manual_core_bboxes_from_layout(
            layout,
            int(source_w),
            int(source_h),
            float(min_tile_ratio),
        )
        core_bboxes, coverage_info, auto_filled_from = _apply_manual_tile_coverage_policy(
            core_bboxes,
            int(source_w),
            int(source_h),
            float(min_tile_ratio),
            str(coverage_policy),
        )
        normalized_layout = {
            **normalized_layout,
            "tiles": _manual_tile_entries_from_bboxes(core_bboxes, int(source_w), int(source_h), auto_filled_from),
            "tile_count": int(len(core_bboxes)),
            "coverage": coverage_info,
        }
        manifest = _build_rect_tile_manifest(
            source_video,
            int(target_w),
            int(target_h),
            float(overlap_ratio),
            int(tile_align),
            int(feather_px),
            core_bboxes,
            mode="manual_rect_spatial_tile_upscale",
            max_tile_pixels=int(max_tile_pixels),
            enforce_tile_pixel_limit=bool(enforce_tile_pixel_limit),
            resolution_snap_mode=str(resolution_snap_mode),
            extra={
                "target_size_adjustment": target_size_adjustment,
                "manual_layout": {
                    **normalized_layout,
                    "min_tile_ratio": float(max(0.01, min(0.45, float(min_tile_ratio)))),
                    "coverage_policy": str(coverage_policy or "auto_fill"),
                },
                "note": (
                    "Manual tile layout: user chooses core rectangles in the front-end editor; "
                    "the node adds overlap and aligned generation sizes automatically."
                ),
            },
        )
        manifest_text = json.dumps(manifest, indent=2)
        debug_preview = _tile_preview_from_manifest(source_video, manifest)
        preview = _save_manual_tile_preview_frames(
            source_video,
            manifest,
            str(preview_filename_prefix or "scail_manual_tile"),
            int(preview_frame_count),
        )
        return {
            "ui": {
                "scail_manual_tile_preview": [preview],
                "scail_manual_tile_preview_json": [json.dumps(preview)],
            },
            "result": (manifest_text, debug_preview, _format_tile_resolution_report(manifest)),
        }


class SCAIL2TileExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        optional: dict[str, Any] = {
            "pose_video_mask": ("IMAGE",),
        }
        for index in range(1, MAX_REFERENCES + 1):
            optional[f"reference_{index}"] = ("IMAGE",)
            optional[f"reference_{index}_mask"] = ("IMAGE",)
        return {
            "required": {
                "source_video": ("IMAGE",),
                "tile_manifest": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "tile_index": ("INT", {"default": 1, "min": 1, "max": MAX_TILES, "step": 1}),
                "reference_count": ("INT", {"default": 1, "min": 1, "max": MAX_REFERENCES, "step": 1}),
                "image_resize_mode": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
                "mask_resize_mode": (["nearest", "bilinear"], {"default": "nearest"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "tile_pose_video",
        "tile_pose_video_mask",
        "tile_reference_1",
        "tile_reference_2",
        "tile_reference_3",
        "tile_reference_4",
        "tile_reference_5",
        "tile_reference_6",
        "tile_reference_7",
        "tile_reference_8",
        "tile_reference_1_mask",
        "tile_reference_2_mask",
        "tile_reference_3_mask",
        "tile_reference_4_mask",
        "tile_reference_5_mask",
        "tile_reference_6_mask",
        "tile_reference_7_mask",
        "tile_reference_8_mask",
        "summary",
        "debug_preview",
    )
    FUNCTION = "extract"
    CATEGORY = f"{CATEGORY}/Tile Upscale"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def extract(
        self,
        source_video: torch.Tensor,
        tile_manifest: str,
        tile_index: int = 1,
        reference_count: int = 1,
        image_resize_mode: str = "bilinear",
        mask_resize_mode: str = "nearest",
        pose_video_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not isinstance(source_video, torch.Tensor) or source_video.ndim != 4:
            raise ValueError("source_video must be a ComfyUI IMAGE tensor.")
        manifest = _parse_tile_manifest(tile_manifest)
        tiles = manifest["tiles"]
        index = int(tile_index) - 1
        if index < 0 or index >= len(tiles):
            raise ValueError(f"tile_index {tile_index} is outside manifest tile count {len(tiles)}.")
        tile = tiles[index]
        target_w, target_h = [int(value) for value in tile["tile_generate_size"]]
        source_bbox = [int(value) for value in tile["source_crop_bbox"]]
        source_size = manifest.get("source_size", [int(source_video.shape[2]), int(source_video.shape[1])])
        pose_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, source_video)
        tile_pose_video = _crop_resize_image_tensor(
            source_video,
            pose_bbox,
            target_h,
            target_w,
            mode=str(image_resize_mode),
        )

        if pose_video_mask is not None:
            pose_mask_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, pose_video_mask)
            tile_pose_mask = _crop_resize_image_tensor(
                pose_video_mask,
                pose_mask_bbox,
                target_h,
                target_w,
                mode=str(mask_resize_mode),
            )
        else:
            pose_mask_bbox = None
            tile_pose_mask = torch.zeros_like(tile_pose_video)

        active_reference_count = max(1, min(MAX_REFERENCES, int(reference_count)))
        tile_references: list[torch.Tensor] = []
        tile_reference_masks: list[torch.Tensor] = []
        reference_summary: list[dict[str, Any]] = []
        blank = _blank_image_like_tile(tile_pose_video, frames=1)
        for ref_index in range(1, MAX_REFERENCES + 1):
            reference = kwargs.get(f"reference_{ref_index}")
            reference_mask = kwargs.get(f"reference_{ref_index}_mask")
            if reference is not None and ref_index <= active_reference_count:
                ref_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, reference)
                tile_ref = _crop_resize_image_tensor(reference[:1], ref_bbox, target_h, target_w, mode=str(image_resize_mode))
                reference_summary.append(
                    {
                        "reference": int(ref_index),
                        "status": "ok",
                        "source_shape": _shape(reference),
                        "source_region_bbox": source_bbox,
                        "source_crop_bbox": ref_bbox,
                        "tile_shape": _shape(tile_ref),
                    }
                )
            else:
                tile_ref = blank.clone()
                reference_summary.append({"reference": int(ref_index), "status": "missing_or_inactive"})
            if reference_mask is not None and ref_index <= active_reference_count:
                mask_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, reference_mask)
                tile_mask = _crop_resize_image_tensor(reference_mask[:1], mask_bbox, target_h, target_w, mode=str(mask_resize_mode))
                reference_summary[-1]["mask_source_crop_bbox"] = mask_bbox
                reference_summary[-1]["tile_mask_shape"] = _shape(tile_mask)
            else:
                tile_mask = torch.zeros_like(tile_ref)
            tile_references.append(tile_ref.contiguous())
            tile_reference_masks.append(tile_mask.contiguous())

        debug_preview = tile_pose_video[: min(4, int(tile_pose_video.shape[0]))].detach().cpu().contiguous()
        summary = {
            "tile_index": int(tile_index),
            "tile": tile,
            "source_video_shape": _shape(source_video),
            "source_region_bbox": source_bbox,
            "pose_video_source_crop_bbox": pose_bbox,
            "pose_video_mask_source_crop_bbox": pose_mask_bbox,
            "tile_pose_video_shape": _shape(tile_pose_video),
            "tile_pose_video_mask_shape": _shape(tile_pose_mask),
            "tile_crop_contract": "same source_region_bbox scaled per input tensor, then resized to tile_generate_size",
            "reference_count": int(active_reference_count),
            "references": reference_summary,
            "image_resize_mode": str(image_resize_mode),
            "mask_resize_mode": str(mask_resize_mode),
            "connect_to": "SCAIL-2 Scheduled Long Video pose_video/reference_N/reference_N_mask inputs.",
        }
        return (
            tile_pose_video,
            tile_pose_mask,
            *tile_references,
            *tile_reference_masks,
            json.dumps(summary, indent=2),
            debug_preview,
        )


class SCAIL2TileRepaintCollector:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"tile_{index}_video": ("IMAGE",) for index in range(2, MAX_TILES + 1)}
        return {
            "required": {
                "tile_manifest": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "tile_1_video": ("IMAGE",),
                "max_tile_pixels": ("INT", {"default": DEFAULT_MAX_TILE_PIXELS, "min": 0, "max": 4096 * 4096, "step": 1024}),
                "enforce_tile_pixel_limit": ("BOOLEAN", {"default": True}),
                "expected_size_mismatch_mode": (["warn", "error", "ignore"], {"default": "warn"}),
                "aspect_mismatch_mode": (["warn", "error", "ignore"], {"default": "warn"}),
                "aspect_tolerance": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.25, "step": 0.005}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",) + ("IMAGE",) * MAX_TILES + ("STRING",)
    RETURN_NAMES = (
        "actual_tile_manifest",
        "tile_1_video",
        "tile_2_video",
        "tile_3_video",
        "tile_4_video",
        "tile_5_video",
        "tile_6_video",
        "tile_7_video",
        "tile_8_video",
        "tile_repaint_report",
    )
    FUNCTION = "collect"
    CATEGORY = f"{CATEGORY}/Tile Upscale"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def collect(
        self,
        tile_manifest: str,
        tile_1_video: torch.Tensor,
        max_tile_pixels: int = DEFAULT_MAX_TILE_PIXELS,
        enforce_tile_pixel_limit: bool = True,
        expected_size_mismatch_mode: str = "warn",
        aspect_mismatch_mode: str = "warn",
        aspect_tolerance: float = 0.03,
        **kwargs,
    ):
        manifest = _parse_tile_manifest(tile_manifest)
        tile_count = int(manifest.get("tile_count", len(manifest.get("tiles", []))))
        if tile_count < 1 or tile_count > MAX_TILES:
            raise ValueError(f"tile_manifest tile_count must be between 1 and {MAX_TILES}.")
        tile_videos = [tile_1_video]
        for index in range(2, tile_count + 1):
            video = kwargs.get(f"tile_{index}_video")
            if video is None:
                raise ValueError(f"tile_manifest requires tile_{index}_video, but it is not connected.")
            tile_videos.append(video)
        actual_manifest = _augment_manifest_with_actual_tile_outputs(
            manifest,
            tile_videos,
            int(max_tile_pixels),
            bool(enforce_tile_pixel_limit),
            str(expected_size_mismatch_mode),
            str(aspect_mismatch_mode),
            float(aspect_tolerance),
        )
        passthrough = list(tile_videos)
        while len(passthrough) < MAX_TILES:
            passthrough.append(tile_videos[-1])
        return (
            json.dumps(actual_manifest, indent=2),
            *passthrough,
            _tile_repaint_report(actual_manifest),
        )


class SCAIL2TileCompositeVideo:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "base_video": ("IMAGE",),
        }
        for index in range(2, MAX_TILES + 1):
            optional[f"tile_{index}_video"] = ("IMAGE",)
        return {
            "required": {
                "tile_manifest": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "tile_1_video": ("IMAGE",),
                "feather_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "tile_fit_mode": (["stretch", "center_crop", "pad"], {"default": "stretch"}),
                "frame_mismatch_mode": (["trim_to_shortest", "error"], {"default": "trim_to_shortest"}),
                "color_correction": ("BOOLEAN", {"default": False}),
                "blend_mode": (["core_feather", "ttp_seam"], {"default": "core_feather"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "summary", "debug_preview")
    FUNCTION = "composite"
    CATEGORY = f"{CATEGORY}/Tile Upscale"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        fingerprint_inputs: dict[str, Any] = {}
        for key in sorted(kwargs):
            value = kwargs[key]
            fingerprint_inputs[key] = value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__
        return _stable_fingerprint(fingerprint_inputs)

    def composite(
        self,
        tile_manifest: str,
        tile_1_video: torch.Tensor,
        feather_px: int = 48,
        tile_fit_mode: str = "stretch",
        frame_mismatch_mode: str = "trim_to_shortest",
        color_correction: bool = False,
        blend_mode: str = "core_feather",
        base_video: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        manifest = _parse_tile_manifest(tile_manifest)
        tiles = manifest.get("tiles", [])
        tile_count = int(manifest.get("tile_count", len(tiles)))
        if tile_count < 1 or tile_count > MAX_TILES or len(tiles) != tile_count:
            raise ValueError(f"SCAIL-2 Tile Composite Video requires a manifest with 1-{MAX_TILES} tiles.")
        tile_videos = [tile_1_video]
        for index in range(2, tile_count + 1):
            video = kwargs.get(f"tile_{index}_video")
            if video is None:
                raise ValueError(f"tile_manifest requires tile_{index}_video, but it is not connected.")
            tile_videos.append(video)
        for index, video in enumerate(tile_videos, start=1):
            if not isinstance(video, torch.Tensor) or video.ndim != 4:
                raise ValueError(f"tile_{index}_video must be a ComfyUI IMAGE tensor.")
        frame_counts = [int(video.shape[0]) for video in tile_videos]
        normalized_frame_mismatch_mode = str(frame_mismatch_mode or "trim_to_shortest").strip()
        if normalized_frame_mismatch_mode not in {"trim_to_shortest", "error"}:
            normalized_frame_mismatch_mode = "trim_to_shortest"
        if normalized_frame_mismatch_mode == "error" and len(set(frame_counts)) != 1:
            raise ValueError(f"tile video frame counts must match: {frame_counts}")
        frame_count = min(frame_counts)
        if frame_count <= 0:
            raise ValueError("tile videos contain no frames.")
        target_w, target_h = [int(value) for value in manifest.get("target_size", [0, 0])]
        if target_w <= 0 or target_h <= 0:
            raise ValueError("tile_manifest target_size is invalid.")
        normalized_tile_fit_mode = str(tile_fit_mode or "stretch").strip()
        if normalized_tile_fit_mode not in {"stretch", "center_crop", "pad"}:
            normalized_tile_fit_mode = "stretch"
        normalized_blend_mode = str(blend_mode or "core_feather").strip()
        if normalized_blend_mode not in {"core_feather", "ttp_seam"}:
            normalized_blend_mode = "core_feather"

        output = torch.zeros((frame_count, target_h, target_w, 3), dtype=torch.float32)
        weights = torch.zeros((frame_count, target_h, target_w, 1), dtype=torch.float32)
        base_target: Optional[torch.Tensor] = None
        if bool(color_correction) and base_video is not None and isinstance(base_video, torch.Tensor) and base_video.ndim == 4:
            base_target = _resize_image_batch(
                base_video[:frame_count].detach().cpu().float().clamp(0, 1),
                target_h,
                target_w,
                mode="bilinear",
            )[:, :, :, :3].contiguous()

        paste_events: list[dict[str, Any]] = []
        for tile, video in zip(tiles, tile_videos):
            crop_bbox = [int(value) for value in tile["target_crop_bbox"]]
            core_bbox = [int(value) for value in tile["target_core_bbox"]]
            x0, y0, x1, y1 = _clamp_bbox(tuple(crop_bbox), target_w, target_h)
            crop_w = max(1, x1 - x0)
            crop_h = max(1, y1 - y0)
            fitted = _fit_image_to_size(
                video[:frame_count].detach().cpu().float().clamp(0, 1)[:, :, :, :3],
                crop_h,
                crop_w,
                fit_mode=normalized_tile_fit_mode,
                mode="bilinear",
            )
            mask = _tile_weight_mask(
                crop_h,
                crop_w,
                [x0, y0, x1, y1],
                core_bbox,
                int(feather_px),
                normalized_blend_mode,
            ).view(1, crop_h, crop_w, 1)
            mask = mask.repeat(frame_count, 1, 1, 1).contiguous()
            if base_target is not None:
                target_patch = base_target[:, y0:y1, x0:x1, :]
                fitted, _color_info = _local_mean_std_color_match(fitted, target_patch, mask)
            output[:, y0:y1, x0:x1, :] += fitted[:, :, :, :3] * mask
            weights[:, y0:y1, x0:x1, :] += mask
            paste_events.append(
                {
                    "tile_number": int(tile.get("tile_number", len(paste_events) + 1)),
                    "target_crop_bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "target_core_bbox": core_bbox,
                    "local_core_bbox_in_crop": [
                        int(core_bbox[0] - x0),
                        int(core_bbox[1] - y0),
                        int(core_bbox[2] - x0),
                        int(core_bbox[3] - y0),
                    ],
                    "tile_generate_size": tile.get("tile_generate_size"),
                    "actual_tile_output_size": tile.get("actual_tile_output_size", [int(video.shape[2]), int(video.shape[1])]),
                    "tile_repaint_scale": tile.get("tile_repaint_scale"),
                    "actual_repaint_scale": tile.get("actual_repaint_scale"),
                    "target_composite_scale": tile.get("target_composite_scale"),
                    "tile_to_target_scale": tile.get("tile_to_target_scale"),
                    "actual_tile_to_target_scale": tile.get("actual_tile_to_target_scale"),
                    "blend_mode": normalized_blend_mode,
                    "input_shape": _shape(video),
                    "fitted_shape": _shape(fitted),
                }
            )

        output = output / weights.clamp_min(1e-6)
        uncovered = weights <= 1e-6
        if bool(uncovered.any()):
            output = output.masked_fill(uncovered.expand_as(output), 0.0)
        debug_preview = output[: min(4, frame_count)].detach().cpu().contiguous().clamp(0, 1)
        summary = {
            "target_size": [int(target_w), int(target_h)],
            "output_shape": _shape(output),
            "input_frame_counts": {
                f"tile_{index + 1}_video": int(count)
                for index, count in enumerate(frame_counts)
            },
            "used_frame_count": int(frame_count),
            "frame_mismatch_mode": normalized_frame_mismatch_mode,
            "feather_px": int(feather_px),
            "tile_weight_mode": normalized_blend_mode,
            "tile_fit_mode": normalized_tile_fit_mode,
            "color_correction": bool(color_correction and base_target is not None),
            "paste_events": paste_events,
            "recommended_face_detail_next_step": (
                "Connect frames to SCAIL-2 Head Track Crop, run a face-detail SCAIL pass on the crop, "
                "then use SCAIL-2 Face Composite Back."
            ),
        }
        return (output.contiguous().clamp(0, 1), json.dumps(summary, indent=2), debug_preview)


def _parse_summary_text(summary_text: Any) -> Any:
    if isinstance(summary_text, str):
        try:
            return json.loads(summary_text)
        except Exception:
            return {"raw_summary": summary_text}
    return summary_text


def _tile_child_unique_id(unique_id: Any, tile_number: int, suffix: str) -> str:
    base = "anonymous" if unique_id is None else str(unique_id)
    return f"{base}:tile:{int(tile_number)}:{suffix}"


def _tile_seed(seed: int, tile_number: int, tile_seed_mode: str) -> int:
    base = int(seed)
    if str(tile_seed_mode or "offset_by_tile").strip() == "same_seed":
        return base
    return int((base + (int(tile_number) - 1) * 1000003) % (0xFFFFFFFFFFFFFFFF + 1))


def _validate_tiled_long_video_manifest(
    manifest: dict[str, Any],
    pose_video: torch.Tensor,
    max_tile_pixels: int,
    enforce_tile_pixel_limit: bool,
) -> list[str]:
    if not isinstance(pose_video, torch.Tensor) or pose_video.ndim != 4:
        raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
    tiles = manifest.get("tiles")
    tile_count = int(manifest.get("tile_count", len(tiles) if isinstance(tiles, list) else 0))
    if not isinstance(tiles, list) or tile_count < 1 or tile_count > MAX_TILES or len(tiles) != tile_count:
        raise ValueError(f"tile_manifest must contain 1-{MAX_TILES} tiles and matching tile_count.")

    source_size = manifest.get("source_size")
    if isinstance(source_size, list) and len(source_size) == 2:
        source_w, source_h = [int(value) for value in source_size]
        if [source_w, source_h] != [int(pose_video.shape[2]), int(pose_video.shape[1])]:
            raise ValueError(
                "tile_manifest source_size must match pose_video size. "
                f"manifest={source_w}x{source_h}, pose_video={int(pose_video.shape[2])}x{int(pose_video.shape[1])}."
            )

    max_pixels = max(0, int(max_tile_pixels))
    warnings: list[str] = []
    oversized: list[str] = []
    for tile in tiles:
        tile_number = int(tile.get("tile_number", int(tile.get("index", 0)) + 1))
        generate_size = tile.get("tile_generate_size")
        if not isinstance(generate_size, list) or len(generate_size) != 2:
            raise ValueError(f"tile {tile_number} is missing tile_generate_size.")
        generate_w, generate_h = [max(1, int(value)) for value in generate_size]
        pixels = int(generate_w * generate_h)
        if max_pixels > 0 and pixels > max_pixels:
            message = f"tile {tile_number} planned {generate_w}x{generate_h} has {pixels} pixels"
            if bool(enforce_tile_pixel_limit):
                oversized.append(message)
            else:
                warnings.append(message)
    if oversized:
        raise ValueError(
            f"Tiled long video rejected tile(s) over max_tile_pixels={max_pixels}: "
            + "; ".join(oversized)
            + ". Resize or split the tile, reduce overlap/target size, or raise max_tile_pixels."
        )
    return warnings


def _collect_reference_inputs(
    kwargs: dict[str, Any],
    reference_count: int,
    *,
    include_masks: bool,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    active_reference_count = max(1, min(MAX_REFERENCES, int(reference_count)))
    references: dict[int, torch.Tensor] = {}
    reference_masks: dict[int, torch.Tensor] = {}
    for index in range(1, active_reference_count + 1):
        reference = kwargs.get(f"reference_{index}")
        if reference is not None:
            references[index] = _first_image(reference, f"reference_{index}")
        if include_masks:
            reference_mask = kwargs.get(f"reference_{index}_mask")
            if reference_mask is not None:
                if not isinstance(reference_mask, torch.Tensor) or reference_mask.ndim != 4:
                    raise ValueError(f"reference_{index}_mask must be a ComfyUI IMAGE tensor.")
                reference_masks[index] = reference_mask[:1].detach().contiguous()
    return references, reference_masks


def _validate_tiled_reference_inputs(
    segments: list[dict[str, Any]],
    references: dict[int, torch.Tensor],
    reference_masks: dict[int, torch.Tensor],
    reference_count: int,
    mode: str,
    *,
    pose_video_mask: Optional[torch.Tensor],
    use_internal_sam: bool,
    sam_model: Any = None,
    sam_conditioning: Any = None,
) -> list[int]:
    active_reference_count = max(1, min(MAX_REFERENCES, int(reference_count)))
    used_refs = sorted({int(segment["reference"]) for segment in segments})
    outside_active = [index for index in used_refs if index > active_reference_count]
    if outside_active:
        raise ValueError(f"segment_plan references {outside_active}, but reference_count is {active_reference_count}.")
    missing = [index for index in used_refs if index not in references]
    if missing:
        raise ValueError(f"segment_plan references missing image input(s): {missing}")
    if str(mode or "replacement") == "replacement":
        if use_internal_sam:
            if sam_model is None or sam_conditioning is None:
                raise ValueError("Tiled Long Video (Internal SAM) replacement mode requires sam_model and sam_conditioning.")
        else:
            if pose_video_mask is None:
                raise ValueError("Tiled Long Video replacement mode requires pose_video_mask.")
            missing_masks = [index for index in used_refs if index not in reference_masks]
            if missing_masks:
                raise ValueError(f"Tiled Long Video replacement mode requires reference_N_mask for used refs: {missing_masks}.")
    return used_refs


def _build_tiled_global_sam_masks(
    pose_video: torch.Tensor,
    references: dict[int, torch.Tensor],
    used_refs: list[int],
    sam_options: dict[str, Any],
    *,
    status_unique_id: Any = None,
    status_node_name: str = "SCAIL2TiledLongVideoWithSAM",
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, Any]]:
    sam_model = sam_options.get("sam_model")
    sam_conditioning = sam_options.get("sam_conditioning")
    if sam_model is None or sam_conditioning is None:
        raise ValueError("Tiled Long Video (Internal SAM) replacement mode requires sam_model and sam_conditioning.")

    object_indices = str(sam_options.get("object_indices", ""))
    reference_object_indices = str(sam_options.get("reference_object_indices", ""))
    sort_by = str(sam_options.get("sort_by", "left_to_right"))
    if sort_by not in {"none", "left_to_right", "area"}:
        sort_by = "left_to_right"
    detection_threshold = float(sam_options.get("sam_detection_threshold", 0.5))
    max_objects = int(sam_options.get("sam_max_objects", 2))
    detect_interval = int(sam_options.get("sam_detect_interval", 2))

    print(
        "[SCAIL2TiledLongVideoWithSAM] global SAM tracking "
        f"pose_frames={int(pose_video.shape[0])} refs={used_refs} "
        f"object_indices='{object_indices}' reference_object_indices='{reference_object_indices}' sort_by={sort_by}"
    )
    _send_long_video_status(status_unique_id, status_node_name, "running_global_sam_pose", "Running global SAM on pose video")
    driving_track_data = _run_sam3_track(
        pose_video,
        sam_model,
        sam_conditioning,
        detection_threshold=detection_threshold,
        max_objects=max_objects,
        detect_interval=detect_interval,
    )

    pose_video_mask = None
    reference_masks: dict[int, torch.Tensor] = {}
    track_summary: list[dict[str, Any]] = []
    for ref_position, index in enumerate(used_refs, start=1):
        _send_long_video_status(
            status_unique_id,
            status_node_name,
            "running_global_sam_reference",
            f"Running global SAM on reference {ref_position}/{len(used_refs)}",
            progress={"current": ref_position, "total": max(1, len(used_refs))},
            reference=int(index),
        )
        reference_track_data = _run_sam3_track(
            references[index],
            sam_model,
            sam_conditioning,
            detection_threshold=detection_threshold,
            max_objects=max_objects,
            detect_interval=1,
        )
        _send_long_video_status(
            status_unique_id,
            status_node_name,
            "building_global_masks",
            f"Building global masks for reference {ref_position}/{len(used_refs)}",
            progress={"current": ref_position, "total": max(1, len(used_refs))},
            reference=int(index),
        )
        current_pose_mask, _ = _create_scail_masks(
            driving_track_data,
            reference_track_data,
            object_indices,
            sort_by,
            True,
        )
        _, reference_mask = _create_scail_masks(
            driving_track_data,
            reference_track_data,
            reference_object_indices,
            sort_by,
            True,
        )
        if pose_video_mask is None:
            pose_video_mask = current_pose_mask.detach().contiguous()
        reference_masks[index] = reference_mask.detach().contiguous()
        track_summary.append(
            {
                "reference": int(index),
                "reference_shape": _shape(references[index]),
                "reference_mask_shape": _shape(reference_mask),
            }
        )

    if pose_video_mask is None:
        raise RuntimeError("Global internal SAM tracking produced no pose_video_mask.")
    _send_long_video_status(status_unique_id, status_node_name, "global_masks_ready", "Global SAM masks ready")

    return (
        pose_video_mask.detach().contiguous(),
        reference_masks,
        {
            "enabled": True,
            "mask_strategy": "global_once_then_tile_crop",
            "object_indices": object_indices,
            "reference_object_indices": reference_object_indices,
            "sort_by": sort_by,
            "sam_detection_threshold": float(detection_threshold),
            "sam_max_objects": int(max_objects),
            "sam_detect_interval": int(detect_interval),
            "pose_video_mask_shape": _shape(pose_video_mask),
            "references_tracked": track_summary,
        },
    )


def _extract_tiled_long_video_inputs(
    pose_video: torch.Tensor,
    pose_video_mask: Optional[torch.Tensor],
    references: dict[int, torch.Tensor],
    reference_masks: dict[int, torch.Tensor],
    manifest: dict[str, Any],
    tile: dict[str, Any],
    reference_count: int,
    image_resize_mode: str,
    mask_resize_mode: str,
    *,
    include_masks: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    source_bbox = [int(value) for value in tile["source_crop_bbox"]]
    target_w, target_h = [int(value) for value in tile["tile_generate_size"]]
    source_size = manifest.get("source_size", [int(pose_video.shape[2]), int(pose_video.shape[1])])
    pose_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, pose_video)
    tile_pose_video = _crop_resize_image_tensor(
        pose_video,
        pose_bbox,
        int(target_h),
        int(target_w),
        mode=str(image_resize_mode),
    )
    tile_kwargs: dict[str, torch.Tensor] = {}
    extraction_summary: dict[str, Any] = {
        "tile_number": int(tile.get("tile_number", int(tile.get("index", 0)) + 1)),
        "source_crop_bbox": source_bbox,
        "source_region_bbox": source_bbox,
        "pose_video_source_crop_bbox": pose_bbox,
        "tile_pose_video_shape": _shape(tile_pose_video),
        "tile_crop_contract": "same source_region_bbox scaled per input tensor, then resized to tile_generate_size",
        "references": [],
    }

    if include_masks and pose_video_mask is not None:
        pose_mask_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, pose_video_mask)
        tile_pose_video_mask = _crop_resize_image_tensor(
            pose_video_mask,
            pose_mask_bbox,
            int(target_h),
            int(target_w),
            mode=str(mask_resize_mode),
        )
        tile_kwargs["pose_video_mask"] = tile_pose_video_mask
        extraction_summary["pose_video_mask_source_crop_bbox"] = pose_mask_bbox
        extraction_summary["tile_pose_video_mask_shape"] = _shape(tile_pose_video_mask)

    active_reference_count = max(1, min(MAX_REFERENCES, int(reference_count)))
    for index in range(1, active_reference_count + 1):
        reference = references.get(index)
        if reference is not None:
            ref_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, reference)
            tile_reference = _crop_resize_image_tensor(
                reference,
                ref_bbox,
                int(target_h),
                int(target_w),
                mode=str(image_resize_mode),
            )
            tile_kwargs[f"reference_{index}"] = tile_reference
            ref_summary = {
                "reference": int(index),
                "source_shape": _shape(reference),
                "source_region_bbox": source_bbox,
                "source_crop_bbox": ref_bbox,
                "tile_shape": _shape(tile_reference),
            }
        else:
            ref_summary = {"reference": int(index), "status": "missing_or_inactive"}

        if include_masks and index in reference_masks:
            reference_mask = reference_masks[index]
            mask_bbox = _tile_tensor_crop_bbox(source_bbox, source_size, reference_mask)
            tile_reference_mask = _crop_resize_image_tensor(
                reference_mask,
                mask_bbox,
                int(target_h),
                int(target_w),
                mode=str(mask_resize_mode),
            )
            tile_kwargs[f"reference_{index}_mask"] = tile_reference_mask
            ref_summary["mask_source_crop_bbox"] = mask_bbox
            ref_summary["tile_mask_shape"] = _shape(tile_reference_mask)
        extraction_summary["references"].append(ref_summary)

    return tile_pose_video, tile_kwargs, extraction_summary


def _run_tiled_long_video(
    *,
    use_internal_sam: bool,
    model,
    clip,
    vae,
    sampler,
    sigmas,
    clip_vision,
    pose_video: torch.Tensor,
    tile_manifest: str,
    segment_plan: str,
    seed: int,
    cfg: float,
    mode: str,
    max_frames: int,
    max_chunk_frames: int,
    overlap_frames: int,
    reference_count: int,
    color_correction: bool,
    cache_mode: str,
    max_tile_pixels: int,
    enforce_tile_pixel_limit: bool,
    expected_size_mismatch_mode: str,
    aspect_mismatch_mode: str,
    aspect_tolerance: float,
    image_resize_mode: str,
    mask_resize_mode: str,
    composite_feather_px: int,
    composite_blend_mode: str,
    tile_fit_mode: str,
    frame_mismatch_mode: str,
    composite_color_correction: bool,
    tile_seed_mode: str,
    free_tail_window: bool,
    pose_video_mask: Optional[torch.Tensor],
    sam_options: Optional[dict[str, Any]],
    prompt=None,
    unique_id=None,
    **kwargs,
):
    if not isinstance(pose_video, torch.Tensor) or pose_video.ndim != 4:
        raise ValueError("pose_video must be a ComfyUI IMAGE tensor.")
    status_node_name = "SCAIL2TiledLongVideoWithSAM" if bool(use_internal_sam) else "SCAIL2TiledLongVideo"

    def send_status(stage: str, message: str, *, progress: Optional[dict[str, Any]] = None, **extra: Any) -> None:
        _send_long_video_status(
            unique_id,
            status_node_name,
            stage,
            message,
            progress=progress,
            **extra,
        )

    send_status("validating_manifest", "Validating tile manifest")
    manifest = _parse_tile_manifest(tile_manifest)
    preflight_warnings = _validate_tiled_long_video_manifest(
        manifest,
        pose_video,
        int(max_tile_pixels),
        bool(enforce_tile_pixel_limit),
    )
    send_status("planning_tiles", f"Planning {len(manifest.get('tiles', []))} tile(s)")
    segments = _parse_plan(segment_plan, pose_frame_count=int(pose_video.shape[0]), max_frames=max_frames)
    send_status("collecting_references", "Collecting tiled references and masks")
    references, reference_masks = _collect_reference_inputs(
        kwargs,
        int(reference_count),
        include_masks=not bool(use_internal_sam),
    )
    sam_options = dict(sam_options or {})
    used_refs = _validate_tiled_reference_inputs(
        segments,
        references,
        reference_masks,
        int(reference_count),
        str(mode),
        pose_video_mask=pose_video_mask,
        use_internal_sam=bool(use_internal_sam),
        sam_model=sam_options.get("sam_model"),
        sam_conditioning=sam_options.get("sam_conditioning"),
    )

    tiles = manifest["tiles"]
    tile_videos: list[torch.Tensor] = []
    tile_summaries: list[dict[str, Any]] = []
    replacement_mode = str(mode or "replacement") == "replacement"
    internal_sam_summary: dict[str, Any] = {
        "enabled": bool(use_internal_sam),
        "mask_strategy": "not_used",
    }
    if bool(use_internal_sam) and replacement_mode:
        send_status("running_global_sam", "Running global SAM before tile cropping")
        pose_video_mask, reference_masks, internal_sam_summary = _build_tiled_global_sam_masks(
            pose_video,
            references,
            used_refs,
            sam_options,
            status_unique_id=unique_id,
            status_node_name=status_node_name,
        )
    elif bool(use_internal_sam):
        internal_sam_summary = {
            "enabled": True,
            "skipped": True,
            "reason": "animation mode does not use SCAIL replacement masks",
            "mask_strategy": "not_used",
        }

    include_masks = bool(replacement_mode)
    generator_suffix = "global_sam_masks" if bool(use_internal_sam) else "external_masks"
    print(
        f"[SCAIL2TiledLongVideo] tiles={len(tiles)} internal_sam={bool(use_internal_sam)} "
        f"mode={mode} used_refs={used_refs} target={manifest.get('target_size')}"
    )

    for tile_position, tile in enumerate(tiles, start=1):
        tile_number = int(tile.get("tile_number", int(tile.get("index", 0)) + 1))
        current_seed = _tile_seed(int(seed), tile_number, str(tile_seed_mode))
        try:
            tile_progress = {"current": tile_position, "total": max(1, len(tiles))}
            send_status(
                "cropping_tile_inputs",
                f"Tile {tile_position}/{len(tiles)}: cropping inputs",
                progress=tile_progress,
                tile_number=int(tile_number),
            )
            tile_pose_video, tile_kwargs, extraction_summary = _extract_tiled_long_video_inputs(
                pose_video,
                pose_video_mask,
                references,
                reference_masks,
                manifest,
                tile,
                int(reference_count),
                str(image_resize_mode),
                str(mask_resize_mode),
                include_masks=include_masks,
            )
            child_unique_id = _tile_child_unique_id(unique_id, tile_number, generator_suffix)
            generator = SCAIL2ScheduledLongVideo()
            send_status(
                "running_tile",
                f"Tile {tile_position}/{len(tiles)}: running long video",
                progress=tile_progress,
                tile_number=int(tile_number),
            )
            frames, used_pose_mask, used_reference_mask_timeline, scheduled_summary_text = generator.generate(
                model,
                clip,
                vae,
                sampler,
                sigmas,
                clip_vision,
                tile_pose_video,
                segment_plan,
                current_seed,
                cfg,
                mode,
                max_frames,
                max_chunk_frames,
                overlap_frames,
                reference_count,
                color_correction,
                cache_mode=str(cache_mode),
                free_tail_window=bool(free_tail_window),
                prompt=prompt,
                unique_id=child_unique_id,
                status_unique_id=unique_id,
                status_prefix=f"Tile {tile_position}/{len(tiles)}: ",
                **tile_kwargs,
            )
        except Exception as exc:
            send_status(
                "error",
                f"Tile {tile_position}/{len(tiles)} failed: {exc}",
                progress={"current": tile_position, "total": max(1, len(tiles))},
                tile_number=int(tile_number),
            )
            raise RuntimeError(f"Tiled long video failed on tile {tile_number}: {exc}") from exc

        tile_videos.append(frames)
        tile_summaries.append(
            {
                "tile_number": int(tile_number),
                "seed": int(current_seed),
                "planned_tile_generate_size": tile.get("tile_generate_size"),
                "source_crop_bbox": tile.get("source_crop_bbox"),
                "target_crop_bbox": tile.get("target_crop_bbox"),
                "extraction": extraction_summary,
                "generated_shape": _shape(frames),
                "used_pose_video_mask_shape": _shape(used_pose_mask),
                "used_reference_mask_timeline_shape": _shape(used_reference_mask_timeline),
                "scheduled_summary": _parse_summary_text(scheduled_summary_text),
            }
        )
        _empty_cache(force=True)
        send_status(
            "tile_complete",
            f"Tile {tile_position}/{len(tiles)} complete",
            progress={"current": tile_position, "total": max(1, len(tiles))},
            tile_number=int(tile_number),
        )

    send_status("collecting_tile_outputs", "Collecting tile outputs")
    actual_manifest = _augment_manifest_with_actual_tile_outputs(
        manifest,
        tile_videos,
        int(max_tile_pixels),
        bool(enforce_tile_pixel_limit),
        str(expected_size_mismatch_mode),
        str(aspect_mismatch_mode),
        float(aspect_tolerance),
    )
    actual_manifest_text = json.dumps(actual_manifest, indent=2)
    composite_kwargs = {
        f"tile_{index + 1}_video": video
        for index, video in enumerate(tile_videos[1:], start=1)
    }
    send_status("compositing_tiles", "Compositing tiles")
    frames, composite_summary_text, debug_preview = SCAIL2TileCompositeVideo().composite(
        actual_manifest_text,
        tile_videos[0],
        int(composite_feather_px),
        str(tile_fit_mode),
        str(frame_mismatch_mode),
        bool(composite_color_correction),
        str(composite_blend_mode),
        base_video=pose_video if bool(composite_color_correction) else None,
        **composite_kwargs,
    )
    summary = {
        "node": "SCAIL2TiledLongVideoWithSAM" if bool(use_internal_sam) else "SCAIL2TiledLongVideo",
        "tile_count": int(len(tile_videos)),
        "target_size": actual_manifest.get("target_size"),
        "source_size": actual_manifest.get("source_size"),
        "mode": str(mode),
        "segment_plan": segment_plan,
        "used_refs": used_refs,
        "tile_seed_mode": str(tile_seed_mode),
        "seed_start": int(seed),
        "free_tail_window": bool(free_tail_window),
        "max_tile_pixels": int(max_tile_pixels),
        "composite_blend_mode": str(composite_blend_mode),
        "internal_sam": internal_sam_summary,
        "preflight_warnings": preflight_warnings,
        "actual_tile_output_warnings": actual_manifest.get("actual_tile_output_warnings", []),
        "tile_summaries": tile_summaries,
        "composite_summary": _parse_summary_text(composite_summary_text),
    }
    send_status("done", f"Done: {int(frames.shape[0])} frame(s)", progress={"current": 1, "total": 1})
    return (
        frames,
        actual_manifest_text,
        _tile_repaint_report(actual_manifest),
        json.dumps(summary, indent=2),
        debug_preview,
    )


class SCAIL2TiledLongVideo:
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
                "tile_manifest": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "segment_plan": ("STRING", {"default": DEFAULT_PLAN, "multiline": True}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "mode": (["replacement", "animation"], {"default": "replacement"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "max_chunk_frames": ("INT", {"default": 81, "min": 17, "max": 81, "step": 4}),
                "overlap_frames": ("INT", {"default": 5, "min": 0, "max": 33, "step": 1}),
                "reference_count": ("INT", {"default": 2, "min": 1, "max": MAX_REFERENCES, "step": 1}),
                "color_correction": ("BOOLEAN", {"default": True}),
                "max_tile_pixels": ("INT", {"default": DEFAULT_MAX_TILE_PIXELS, "min": 0, "max": 4096 * 4096, "step": 1024}),
                "enforce_tile_pixel_limit": ("BOOLEAN", {"default": True}),
                "expected_size_mismatch_mode": (["warn", "error", "ignore"], {"default": "warn"}),
                "aspect_mismatch_mode": (["warn", "error", "ignore"], {"default": "warn"}),
                "aspect_tolerance": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.25, "step": 0.005}),
                "image_resize_mode": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
                "mask_resize_mode": (["nearest", "bilinear"], {"default": "nearest"}),
                "composite_feather_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "tile_fit_mode": (["stretch", "center_crop", "pad"], {"default": "stretch"}),
                "frame_mismatch_mode": (["trim_to_shortest", "error"], {"default": "trim_to_shortest"}),
                "composite_color_correction": ("BOOLEAN", {"default": False}),
                "tile_seed_mode": (["offset_by_tile", "same_seed"], {"default": "offset_by_tile"}),
                "cache_mode": (["disk", "off"], {"default": "disk"}),
                "composite_blend_mode": (["core_feather", "ttp_seam"], {"default": "core_feather"}),
                "free_tail_window": _free_tail_window_input_spec(),
            },
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("frames", "actual_tile_manifest", "tile_repaint_report", "summary", "debug_preview")
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY}/Tile Upscale"

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
        tile_manifest: str,
        segment_plan: str,
        seed: int,
        cfg: float,
        mode: str,
        max_frames: int,
        max_chunk_frames: int,
        overlap_frames: int,
        reference_count: int,
        color_correction: bool,
        max_tile_pixels: int = DEFAULT_MAX_TILE_PIXELS,
        enforce_tile_pixel_limit: bool = True,
        expected_size_mismatch_mode: str = "warn",
        aspect_mismatch_mode: str = "warn",
        aspect_tolerance: float = 0.03,
        image_resize_mode: str = "bilinear",
        mask_resize_mode: str = "nearest",
        composite_feather_px: int = 48,
        tile_fit_mode: str = "stretch",
        frame_mismatch_mode: str = "trim_to_shortest",
        composite_color_correction: bool = False,
        tile_seed_mode: str = "offset_by_tile",
        cache_mode: str = "disk",
        composite_blend_mode: str = "core_feather",
        free_tail_window: bool = False,
        pose_video_mask: Optional[torch.Tensor] = None,
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        return _run_tiled_long_video(
            use_internal_sam=False,
            model=model,
            clip=clip,
            vae=vae,
            sampler=sampler,
            sigmas=sigmas,
            clip_vision=clip_vision,
            pose_video=pose_video,
            tile_manifest=tile_manifest,
            segment_plan=segment_plan,
            seed=int(seed),
            cfg=float(cfg),
            mode=str(mode),
            max_frames=int(max_frames),
            max_chunk_frames=int(max_chunk_frames),
            overlap_frames=int(overlap_frames),
            reference_count=int(reference_count),
            color_correction=bool(color_correction),
            cache_mode=str(cache_mode),
            max_tile_pixels=int(max_tile_pixels),
            enforce_tile_pixel_limit=bool(enforce_tile_pixel_limit),
            expected_size_mismatch_mode=str(expected_size_mismatch_mode),
            aspect_mismatch_mode=str(aspect_mismatch_mode),
            aspect_tolerance=float(aspect_tolerance),
            image_resize_mode=str(image_resize_mode),
            mask_resize_mode=str(mask_resize_mode),
            composite_feather_px=int(composite_feather_px),
            composite_blend_mode=str(composite_blend_mode),
            tile_fit_mode=str(tile_fit_mode),
            frame_mismatch_mode=str(frame_mismatch_mode),
            composite_color_correction=bool(composite_color_correction),
            tile_seed_mode=str(tile_seed_mode),
            free_tail_window=bool(free_tail_window),
            pose_video_mask=pose_video_mask,
            sam_options=None,
            prompt=prompt,
            unique_id=unique_id,
            **kwargs,
        )


class SCAIL2TiledLongVideoWithSAM:
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
                "tile_manifest": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
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
                        "tooltip": "Comma-separated reference-image object indices. Empty = all reference objects.",
                    },
                ),
                "sort_by": (["none", "left_to_right", "area"], {"default": "left_to_right"}),
                "sam_detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sam_max_objects": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "sam_detect_interval": ("INT", {"default": 2, "min": 1, "max": 999, "step": 1}),
                "max_tile_pixels": ("INT", {"default": DEFAULT_MAX_TILE_PIXELS, "min": 0, "max": 4096 * 4096, "step": 1024}),
                "enforce_tile_pixel_limit": ("BOOLEAN", {"default": True}),
                "expected_size_mismatch_mode": (["warn", "error", "ignore"], {"default": "warn"}),
                "aspect_mismatch_mode": (["warn", "error", "ignore"], {"default": "warn"}),
                "aspect_tolerance": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 0.25, "step": 0.005}),
                "image_resize_mode": (["bilinear", "bicubic", "nearest"], {"default": "bilinear"}),
                "mask_resize_mode": (["nearest", "bilinear"], {"default": "nearest"}),
                "composite_feather_px": ("INT", {"default": 48, "min": 0, "max": 512, "step": 1}),
                "tile_fit_mode": (["stretch", "center_crop", "pad"], {"default": "stretch"}),
                "frame_mismatch_mode": (["trim_to_shortest", "error"], {"default": "trim_to_shortest"}),
                "composite_color_correction": ("BOOLEAN", {"default": False}),
                "tile_seed_mode": (["offset_by_tile", "same_seed"], {"default": "offset_by_tile"}),
                "cache_mode": (["disk", "off"], {"default": "disk"}),
                "composite_blend_mode": (["core_feather", "ttp_seam"], {"default": "core_feather"}),
                "free_tail_window": _free_tail_window_input_spec(),
            },
            "optional": optional,
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = SCAIL2TiledLongVideo.RETURN_TYPES
    RETURN_NAMES = SCAIL2TiledLongVideo.RETURN_NAMES
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY}/Tile Upscale"

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
        tile_manifest: str,
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
        max_tile_pixels: int = DEFAULT_MAX_TILE_PIXELS,
        enforce_tile_pixel_limit: bool = True,
        expected_size_mismatch_mode: str = "warn",
        aspect_mismatch_mode: str = "warn",
        aspect_tolerance: float = 0.03,
        image_resize_mode: str = "bilinear",
        mask_resize_mode: str = "nearest",
        composite_feather_px: int = 48,
        tile_fit_mode: str = "stretch",
        frame_mismatch_mode: str = "trim_to_shortest",
        composite_color_correction: bool = False,
        tile_seed_mode: str = "offset_by_tile",
        cache_mode: str = "disk",
        composite_blend_mode: str = "core_feather",
        free_tail_window: bool = False,
        sam_model=None,
        sam_conditioning=None,
        prompt=None,
        unique_id=None,
        **kwargs,
    ):
        return _run_tiled_long_video(
            use_internal_sam=True,
            model=model,
            clip=clip,
            vae=vae,
            sampler=sampler,
            sigmas=sigmas,
            clip_vision=clip_vision,
            pose_video=pose_video,
            tile_manifest=tile_manifest,
            segment_plan=segment_plan,
            seed=int(seed),
            cfg=float(cfg),
            mode=str(mode),
            max_frames=int(max_frames),
            max_chunk_frames=int(max_chunk_frames),
            overlap_frames=int(overlap_frames),
            reference_count=int(reference_count),
            color_correction=bool(color_correction),
            cache_mode=str(cache_mode),
            max_tile_pixels=int(max_tile_pixels),
            enforce_tile_pixel_limit=bool(enforce_tile_pixel_limit),
            expected_size_mismatch_mode=str(expected_size_mismatch_mode),
            aspect_mismatch_mode=str(aspect_mismatch_mode),
            aspect_tolerance=float(aspect_tolerance),
            image_resize_mode=str(image_resize_mode),
            mask_resize_mode=str(mask_resize_mode),
            composite_feather_px=int(composite_feather_px),
            composite_blend_mode=str(composite_blend_mode),
            tile_fit_mode=str(tile_fit_mode),
            frame_mismatch_mode=str(frame_mismatch_mode),
            composite_color_correction=bool(composite_color_correction),
            tile_seed_mode=str(tile_seed_mode),
            free_tail_window=bool(free_tail_window),
            pose_video_mask=None,
            sam_options={
                "sam_model": sam_model,
                "sam_conditioning": sam_conditioning,
                "object_indices": object_indices,
                "reference_object_indices": reference_object_indices,
                "sort_by": sort_by,
                "sam_detection_threshold": float(sam_detection_threshold),
                "sam_max_objects": int(sam_max_objects),
                "sam_detect_interval": int(sam_detect_interval),
            },
            prompt=prompt,
            unique_id=unique_id,
            **kwargs,
        )


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
    "SCAIL2TilePlanBuilder": SCAIL2TilePlanBuilder,
    "SCAIL2ManualTilePlanBuilder": SCAIL2ManualTilePlanBuilder,
    "SCAIL2TileExtractor": SCAIL2TileExtractor,
    "SCAIL2TileRepaintCollector": SCAIL2TileRepaintCollector,
    "SCAIL2TileCompositeVideo": SCAIL2TileCompositeVideo,
    "SCAIL2TiledLongVideo": SCAIL2TiledLongVideo,
    "SCAIL2TiledLongVideoWithSAM": SCAIL2TiledLongVideoWithSAM,
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
    "SCAIL2TilePlanBuilder": "SCAIL-2 Tile Plan Builder",
    "SCAIL2ManualTilePlanBuilder": "SCAIL-2 Manual Tile Plan Builder",
    "SCAIL2TileExtractor": "SCAIL-2 Tile Extractor",
    "SCAIL2TileRepaintCollector": "SCAIL-2 Tile Repaint Collector",
    "SCAIL2TileCompositeVideo": "SCAIL-2 Tile Composite Video",
    "SCAIL2TiledLongVideo": "SCAIL-2 Tiled Long Video",
    "SCAIL2TiledLongVideoWithSAM": "SCAIL-2 Tiled Long Video (Internal SAM)",
}

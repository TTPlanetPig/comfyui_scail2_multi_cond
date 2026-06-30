import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[SCAIL Multi Cond] dynamic UI extension loaded");

const MAX_SEGMENTS = 8;
const MAX_REFERENCES = 8;
const MAX_TILES = 8;
const LONG_VIDEO_STATUS_EVENT = "scail2_long_video_status";
const SCAIL_TOOLTIP_DELAY_MS = 3000;
const LONG_VIDEO_NODE_TYPES = new Set([
    "SCAIL2ScheduledLongVideo",
    "SCAIL2ScheduledLongVideoWithSAM",
    "SCAIL2TiledLongVideo",
    "SCAIL2TiledLongVideoWithSAM",
]);
const SCAIL_WIDGET_TOOLTIPS = new Map([
    ["segment_count", "Number of plan segments to expose in this builder."],
    ["segment_plan", "Frame plan used by the long-video scheduler. Each row selects frames, reference, prompt, negative prompt, and optional boundary overlap."],
    ["max_frames", "Optional hard cap for output frames. Use 0 to follow the input video length and segment plan."],
    ["pose_frame_count", "Frame count used for planning when no pose video is connected to this helper."],
    ["max_chunk_frames", "Maximum frames in one SCAIL sampling window. SCAIL windows must be 4n+1 and are capped at 81."],
    ["overlap_frames", "Frames reused from the previous chunk as previous_frames context. Higher values improve continuity but reduce new frames per chunk."],
    ["boundary_overlap", "Override overlap only at this segment boundary, useful when changing references."],
    ["reference_count", "Number of reference image slots to keep active."],
    ["seed", "Base random seed. Tiled nodes may offset this per tile depending on tile_seed_mode."],
    ["cfg", "Classifier-free guidance scale passed to sampling."],
    ["mode", "replacement uses pose/reference masks; animation skips replacement masks and only uses motion/reference conditioning."],
    ["color_correction", "Match chunk color against previous context after overlap removal."],
    ["cache_mode", "disk reuses the last matching node result when inputs fingerprint the same; off always recomputes."],
    ["free_tail_window", "Blank final conditioning frames to append and discard. 0 disables; 4 adds one latent step, 8 adds two, etc."],
    ["object_indices", "Comma-separated tracked object indices from the driving video. Empty keeps all objects after sorting."],
    ["reference_object_indices", "Comma-separated tracked object indices from the reference image. Empty keeps all reference objects."],
    ["sort_by", "How tracked objects are ordered before object index filtering."],
    ["sam_detection_threshold", "SAM detection confidence threshold. Higher values are stricter."],
    ["sam_max_objects", "Maximum objects SAM should track."],
    ["sam_detect_interval", "Frame interval for SAM detection refresh during tracking."],
    ["tile_manifest", "JSON tile plan describing source crops, target crops, target size, overlap, and repaint constraints."],
    ["output_width", "Requested final output width. Some builders adjust it to preserve aspect ratio and alignment."],
    ["output_height", "Requested final output height. Some builders adjust it to preserve aspect ratio and alignment."],
    ["scale_factor", "Fallback upscale multiplier when output width or height is not explicitly set."],
    ["overlap_ratio", "Tile overlap ratio applied only where tiles touch neighboring tiles."],
    ["tile_align", "Pixel alignment for tile boxes and generated tile sizes. SCAIL requires 32-pixel multiples; values are normalized to 32 steps."],
    ["resolution_snap_mode", "Controls how target output resolution is snapped to alignment constraints."],
    ["feather_px", "Blend feather width in output pixels for tile or face compositing."],
    ["composite_feather_px", "Blend feather width used when tiled long video composites generated tile videos."],
    ["min_tile_ratio", "Minimum allowed tile width/height ratio relative to the source frame."],
    ["protected_padding_ratio", "Extra protected-area padding as a ratio of source dimensions."],
    ["protected_padding_px", "Extra protected-area padding in source pixels."],
    ["max_tile_pixels", "Maximum generated pixels allowed for each tile. Use this to stay under model limits."],
    ["enforce_tile_pixel_limit", "Reject tile plans or repaint videos that exceed max_tile_pixels."],
    ["expected_size_mismatch_mode", "What to do when a generated tile video does not match the planned tile size."],
    ["aspect_mismatch_mode", "What to do when a generated tile aspect ratio differs from the planned tile aspect ratio."],
    ["aspect_tolerance", "Allowed aspect-ratio error before aspect_mismatch_mode is triggered."],
    ["image_resize_mode", "Resize filter used for reference images and pose/tile image crops."],
    ["mask_resize_mode", "Resize filter used for masks. nearest keeps mask edges crisp."],
    ["tile_fit_mode", "How generated tile video is fit into the planned tile region before compositing."],
    ["frame_mismatch_mode", "How to handle tile videos with different frame counts."],
    ["composite_color_correction", "Match tile colors during final compositing."],
    ["tile_seed_mode", "Use the same seed for every tile or offset the seed by tile number."],
    ["composite_blend_mode", "Tile seam blending method. core_feather is standard; ttp_seam uses the alternate TTP-style blend."],
    ["blend_mode", "Tile composite blend method."],
    ["layout_json", "Manual tile layout written by the visual editor. Keep this connected through the builder output."],
    ["preview_frame_count", "Number of preview frames saved for the manual tile editor."],
    ["preview_filename_prefix", "Filename prefix for manual tile preview images."],
    ["coverage_policy", "How the manual tile builder handles uncovered source areas."],
    ["tile_index", "1-based tile number to extract from the tile manifest."],
    ["include_final_anchor", "Also export the final frame as a keyframe anchor."],
    ["contact_sheet_columns", "Number of columns in the keyframe contact sheet."],
    ["contact_sheet_thumbnail_width", "Thumbnail width used in the keyframe contact sheet."],
    ["boundary_anchor_mode", "Which frame is used as the anchor around chunk boundaries."],
    ["planner_summary", "Optional planner JSON summary. When connected, chunk/keyframe helpers use it instead of rebuilding a plan."],
    ["filename_prefix", "Filename prefix used when saving viewer output images."],
    ["save_location", "Where viewer output images are saved."],
    ["display_group", "Which keyframe group the matrix viewer displays."],
    ["crop_padding_ratio", "Relative padding around the tracked head crop."],
    ["square_align", "Pixel alignment for square face crops. Keep this on 32-pixel steps so the face crop matches SCAIL generation geometry."],
    ["temporal_smoothing", "Amount of smoothing applied to tracked crop motion."],
    ["mask_expand_px", "Pixels to expand masks before blur or compositing."],
    ["mask_blur_px", "Pixels used to blur mask edges."],
    ["crop_mode", "Face/head crop shape strategy."],
    ["mask_component_mode", "Which mask components are kept for face/head tracking."],
    ["target_frame_index", "Frame index used as the alignment target."],
    ["face_scale", "Scale applied to the detected reference face before fitting."],
    ["x_offset_ratio", "Horizontal reference-face offset inside the aligned crop."],
    ["y_offset_ratio", "Vertical reference-face offset inside the aligned crop."],
    ["face_size_basis", "Which face measurement is used as the size basis for alignment."],
    ["target_face_select", "Which detected face to use in the target crop."],
    ["reference_face_select", "Which detected face to use in the reference image."],
    ["padding_mode", "How pixels outside the reference image are filled during alignment."],
    ["insightface_model", "InsightFace model name used by the face aligner."],
    ["provider", "Execution provider for face detection when available."],
    ["det_size", "Detector input size for InsightFace."],
    ["face_detector_backend", "Face detector backend used for reference alignment."],
    ["mediapipe_model_selection", "MediaPipe face detection model selection."],
    ["mediapipe_min_detection_confidence", "Minimum confidence for MediaPipe face detection."],
    ["window_fit_mode", "How the aligned reference image is fit into the crop window."],
    ["mask_contract_px", "Pixels to contract the face mask before compositing back."],
    ["color_match_method", "Color matching method for face compositing."],
    ["face_fit_mode", "How the refined face video is fit back into the original crop."],
    ["stitch_mask_expand_px", "Pixels to expand the stitch mask at paste-back time."],
    ["stitch_mask_resize_mode", "Resize filter used for stitch masks."],
    ["stitch_offset_x_px", "Horizontal paste-back offset in pixels."],
    ["stitch_offset_y_px", "Vertical paste-back offset in pixels."],
]);

function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

let scailTooltipListenersInstalled = false;
let scailTooltipElement = null;
let scailTooltipTimer = null;
let scailTooltipHoverKey = "";
let scailTooltipHover = null;

function scailTooltipForName(name) {
    const text = SCAIL_WIDGET_TOOLTIPS.get(name);
    if (text) {
        return text;
    }
    if (/^segment_\d+_frames$/.test(name)) {
        return "Number of output frames kept for this segment.";
    }
    if (/^segment_\d+_reference$/.test(name)) {
        return "Reference image slot used by this segment.";
    }
    if (/^segment_\d+_prompt$/.test(name)) {
        return "Positive prompt used only for this segment.";
    }
    if (/^segment_\d+_negative$/.test(name)) {
        return "Negative prompt used only for this segment.";
    }
    if (/^segment_\d+_boundary_overlap$/.test(name)) {
        return "Optional overlap override when entering this segment. Leave blank to use overlap_frames.";
    }
    if (/^reference_\d+$/.test(name)) {
        return "Reference image for segment_plan rows that select this reference number.";
    }
    if (/^reference_\d+_mask$/.test(name)) {
        return "Replacement mask paired with the matching reference image.";
    }
    if (/^reference_\d+_track_data$/.test(name)) {
        return "SAM track data for this reference image, used to build replacement masks.";
    }
    if (/^tile_\d+_video$/.test(name)) {
        return "Generated repaint video for this tile number.";
    }
    return "";
}

function scailTooltipForWidget(node, widget) {
    const existing =
        widget?.tooltip ??
        widget?.options?.tooltip ??
        widget?.callback?.tooltip ??
        "";
    return String(existing || scailTooltipForName(String(widget?.name ?? "")) || "").trim();
}

function installScailWidgetTooltips(node) {
    ensureScailTooltipListeners();
    for (const widget of node.widgets ?? []) {
        const tooltip = scailTooltipForWidget(node, widget);
        if (!tooltip) {
            continue;
        }
        widget.tooltip = tooltip;
        if (widget.options && typeof widget.options === "object" && !Array.isArray(widget.options)) {
            widget.options.tooltip = tooltip;
        }
    }
}

function ensureScailTooltipElement() {
    if (scailTooltipElement) {
        return scailTooltipElement;
    }
    const element = document.createElement("div");
    element.className = "scail-widget-tooltip";
    element.style.cssText = [
        "position:fixed",
        "z-index:100000",
        "display:none",
        "pointer-events:none",
        "max-width:320px",
        "padding:8px 10px",
        "border:1px solid rgba(148,163,184,.45)",
        "border-radius:6px",
        "background:rgba(15,23,42,.96)",
        "color:#e5e7eb",
        "box-shadow:0 12px 28px rgba(0,0,0,.35)",
        "font:12px/1.4 system-ui,-apple-system,BlinkMacSystemFont,sans-serif",
        "white-space:normal",
        "overflow-wrap:anywhere",
    ].join(";");
    document.body.append(element);
    scailTooltipElement = element;
    return element;
}

function positionScailTooltip(event) {
    const element = ensureScailTooltipElement();
    const margin = 12;
    const maxX = Math.max(margin, window.innerWidth - element.offsetWidth - margin);
    const maxY = Math.max(margin, window.innerHeight - element.offsetHeight - margin);
    const x = Math.min(maxX, Math.max(margin, event.clientX + 14));
    const y = Math.min(maxY, Math.max(margin, event.clientY + 16));
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;
}

function hideScailTooltip() {
    if (scailTooltipTimer) {
        clearTimeout(scailTooltipTimer);
        scailTooltipTimer = null;
    }
    scailTooltipHoverKey = "";
    scailTooltipHover = null;
    if (scailTooltipElement) {
        scailTooltipElement.style.display = "none";
    }
}

function canvasEventToGraphPoint(event) {
    const graphCanvas = app.canvas;
    const canvas = graphCanvas?.canvas;
    if (!canvas) {
        return null;
    }
    let offset = null;
    try {
        offset = graphCanvas.convertEventToCanvasOffset?.(event);
    } catch {
        offset = null;
    }
    if (!Array.isArray(offset)) {
        const rect = canvas.getBoundingClientRect();
        offset = [event.clientX - rect.left, event.clientY - rect.top];
    }
    try {
        const converted = graphCanvas.ds?.convertOffsetToCanvas?.(offset);
        if (Array.isArray(converted)) {
            return converted;
        }
    } catch {
        // Fallback below handles older LiteGraph builds.
    }
    const scale = Number(graphCanvas.ds?.scale ?? 1) || 1;
    const dsOffset = graphCanvas.ds?.offset ?? [0, 0];
    return [
        (offset[0] - Number(dsOffset[0] ?? 0)) / scale,
        (offset[1] - Number(dsOffset[1] ?? 0)) / scale,
    ];
}

function scailWidgetBounds(node, widget) {
    if (!widget || widget.hidden) {
        return null;
    }
    const y = Number(widget.last_y);
    if (!Number.isFinite(y)) {
        return null;
    }
    let height = Number(widget.computedHeight ?? widget.height ?? 0);
    if (!Number.isFinite(height) || height <= 0) {
        try {
            const computed = widget.computeSize?.(node.size?.[0] ?? 420);
            if (Array.isArray(computed)) {
                height = Number(computed[1]);
            }
        } catch {
            height = 0;
        }
    }
    if (!Number.isFinite(height) || height <= 0) {
        height = 20;
    }
    const margin = 6;
    return {
        x: Number(node.pos?.[0] ?? 0) + margin,
        y: Number(node.pos?.[1] ?? 0) + y,
        width: Math.max(1, Number(node.size?.[0] ?? 420) - margin * 2),
        height,
    };
}

function findHoveredScailWidget(event) {
    const point = canvasEventToGraphPoint(event);
    if (!point) {
        return null;
    }
    const nodes = app.graph?._nodes ?? [];
    for (let nodeIndex = nodes.length - 1; nodeIndex >= 0; nodeIndex -= 1) {
        const node = nodes[nodeIndex];
        const nodeName = String(node?.scailNodeName ?? node?.type ?? "");
        if (!nodeName.startsWith("SCAIL2")) {
            continue;
        }
        const nodeX = Number(node.pos?.[0] ?? 0);
        const nodeY = Number(node.pos?.[1] ?? 0);
        const nodeW = Number(node.size?.[0] ?? 0);
        const nodeH = Number(node.size?.[1] ?? 0);
        if (point[0] < nodeX || point[0] > nodeX + nodeW || point[1] < nodeY || point[1] > nodeY + nodeH) {
            continue;
        }
        installScailWidgetTooltips(node);
        for (const widget of node.widgets ?? []) {
            const tooltip = scailTooltipForWidget(node, widget);
            if (!tooltip) {
                continue;
            }
            const bounds = scailWidgetBounds(node, widget);
            if (!bounds) {
                continue;
            }
            if (
                point[0] >= bounds.x &&
                point[0] <= bounds.x + bounds.width &&
                point[1] >= bounds.y &&
                point[1] <= bounds.y + bounds.height
            ) {
                return { node, widget, tooltip };
            }
        }
    }
    return null;
}

function handleScailTooltipMove(event) {
    const hit = findHoveredScailWidget(event);
    if (!hit) {
        hideScailTooltip();
        return;
    }
    const key = `${hit.node.id ?? hit.node.title}:${hit.widget.name}`;
    if (key === scailTooltipHoverKey) {
        scailTooltipHover = { ...hit, event };
        if (scailTooltipElement?.style.display === "block") {
            positionScailTooltip(event);
        }
        return;
    }
    hideScailTooltip();
    scailTooltipHoverKey = key;
    scailTooltipHover = { ...hit, event };
    scailTooltipTimer = window.setTimeout(() => {
        if (!scailTooltipHover || scailTooltipHoverKey !== key) {
            return;
        }
        const element = ensureScailTooltipElement();
        element.textContent = scailTooltipHover.tooltip;
        element.style.display = "block";
        positionScailTooltip(scailTooltipHover.event);
    }, SCAIL_TOOLTIP_DELAY_MS);
}

function ensureScailTooltipListeners() {
    if (scailTooltipListenersInstalled) {
        return;
    }
    const canvas = app.canvas?.canvas;
    if (!canvas) {
        return;
    }
    scailTooltipListenersInstalled = true;
    canvas.addEventListener("mousemove", handleScailTooltipMove, { passive: true });
    canvas.addEventListener("mouseleave", hideScailTooltip, { passive: true });
    canvas.addEventListener("mousedown", hideScailTooltip, { passive: true });
    canvas.addEventListener("wheel", hideScailTooltip, { passive: true });
}

function setWidgetVisible(widget, visible) {
    if (!widget) {
        return;
    }
    widget.hidden = !visible;
    widget.computeSize = visible
        ? undefined
        : () => [0, -4];
}

function setInputVisible(node, inputName, visible) {
    const input = node.inputs?.find((slot) => slot.name === inputName);
    if (!input) {
        return;
    }
    input.hidden = !visible;
}

function scheduleCanvas(node) {
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function resizeNode(node) {
    if (node.computeSize) {
        const size = node.computeSize();
        node.size = [Math.max(node.size?.[0] ?? 420, size[0]), size[1]];
    }
    scheduleCanvas(node);
}

function findGraphNodeById(nodeId) {
    const nodeIdText = String(nodeId ?? "");
    if (!nodeIdText) {
        return null;
    }
    const numericId = Number(nodeIdText);
    if (Number.isFinite(numericId)) {
        const direct = app.graph?.getNodeById?.(numericId) ?? app.graph?._nodes_by_id?.[numericId];
        if (direct) {
            return direct;
        }
    }
    return app.graph?._nodes?.find((node) => String(node.id) === nodeIdText) ?? null;
}

function ensureLongVideoStatusWidget(node) {
    if (node.scailLongVideoStatusContainer) {
        return node.scailLongVideoStatusContainer;
    }
    const container = document.createElement("div");
    container.className = "scail-long-video-status";
    container.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "min-height:36px",
        "padding:7px 8px",
        "border:1px solid rgba(125,211,252,.32)",
        "border-left:3px solid #38bdf8",
        "border-radius:5px",
        "background:#0f172a",
        "color:#e2e8f0",
        "font:12px/1.35 system-ui,-apple-system,BlinkMacSystemFont,sans-serif",
    ].join(";");

    if (node.addDOMWidget) {
        const widget = node.addDOMWidget("scail_long_video_status", "SCAIL-2 Status", container, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.computeSize = () => [
            Math.max(420, node.size?.[0] ?? 420),
            46,
        ];
    } else {
        node.addWidget("text", "scail_long_video_status", "Idle", () => {}, { serialize: false });
    }
    node.scailLongVideoStatusContainer = container;
    renderLongVideoStatus(node, node.scailLongVideoStatus ?? { stage: "idle", message: "Idle" });
    return container;
}

function renderLongVideoStatus(node, status) {
    const container = ensureLongVideoStatusWidget(node);
    const detail = status ?? {};
    node.scailLongVideoStatus = detail;
    const stage = String(detail.stage ?? "idle");
    const message = String(detail.message ?? stage);
    const progress = detail.progress ?? {};
    const current = Number(progress.current);
    const total = Number(progress.total);
    const hasProgress = Number.isFinite(current) && Number.isFinite(total) && total > 0;
    const timestamp = Number(detail.timestamp);
    const timeText = Number.isFinite(timestamp)
        ? new Date(timestamp * 1000).toLocaleTimeString()
        : new Date().toLocaleTimeString();

    container.replaceChildren();
    const main = document.createElement("div");
    main.textContent = message;
    main.style.cssText = "font-weight:700;white-space:normal;overflow-wrap:anywhere;";
    const meta = document.createElement("div");
    meta.textContent = [
        stage,
        hasProgress ? `${Math.round(current)}/${Math.round(total)}` : "",
        timeText,
    ].filter(Boolean).join(" · ");
    meta.style.cssText = "opacity:.72;margin-top:2px;";
    container.append(main, meta);
    container.style.borderLeftColor = stage === "done" || stage === "cache_hit"
        ? "#22c55e"
        : stage === "error"
            ? "#ef4444"
            : "#38bdf8";
    scheduleCanvas(node);
}

function handleLongVideoStatus(event) {
    const detail = event?.detail ?? event;
    const node = findGraphNodeById(detail?.node_id);
    if (!node) {
        return;
    }
    ensureLongVideoStatusWidget(node);
    renderLongVideoStatus(node, detail);
}

api.addEventListener?.(LONG_VIDEO_STATUS_EVENT, handleLongVideoStatus);

function updateSegmentBuilder(node) {
    const countWidget = widgetByName(node, "segment_count");
    const count = Math.max(1, Math.min(MAX_SEGMENTS, Number(countWidget?.value ?? 1)));
    if (countWidget) {
        countWidget.value = count;
    }

    for (let index = 1; index <= MAX_SEGMENTS; index += 1) {
        const visible = index <= count;
        for (const suffix of ["frames", "reference", "prompt", "negative", "boundary_overlap"]) {
            setWidgetVisible(widgetByName(node, `segment_${index}_${suffix}`), visible);
        }
    }
    resizeNode(node);
}

function referenceInputInfo(input) {
    const match = /^reference_(\d+)(?:_mask)?$/.exec(input?.name ?? "");
    if (!match) {
        return null;
    }
    return {
        number: Number(match[1]),
        isMask: input.name.endsWith("_mask"),
    };
}

function referenceTrackInputNumber(input) {
    const match = /^reference_(\d+)_track_data$/.exec(input?.name ?? "");
    return match ? Number(match[1]) : null;
}

function referenceMaskOutputNumber(output) {
    const match = /^reference_(\d+)_mask$/.exec(output?.name ?? "");
    return match ? Number(match[1]) : null;
}

function syncOutputLinkSlots(node) {
    const graph = node.graph ?? app.graph;
    for (let outputIndex = 0; outputIndex < (node.outputs?.length ?? 0); outputIndex += 1) {
        const output = node.outputs[outputIndex];
        for (const linkId of output?.links ?? []) {
            const link = graph?.links?.[linkId];
            if (link && link.origin_id === node.id) {
                link.origin_slot = outputIndex;
            }
        }
    }
}

function updateMultiReferenceMaskOutputs(node, count) {
    const desiredNames = [
        "pose_video_mask",
        ...Array.from({ length: count }, (_, index) => `reference_${index + 1}_mask`),
    ];
    const desiredNameSet = new Set(desiredNames);

    for (let outputIndex = (node.outputs?.length ?? 0) - 1; outputIndex >= 0; outputIndex -= 1) {
        const output = node.outputs[outputIndex];
        const referenceNumber = referenceMaskOutputNumber(output);
        const shouldRemove =
            !desiredNameSet.has(output?.name) ||
            (referenceNumber !== null && referenceNumber > count);
        if (shouldRemove) {
            node.removeOutput(outputIndex);
        }
    }

    const existingNames = new Set((node.outputs ?? []).map((output) => output.name));
    for (const name of desiredNames) {
        if (!existingNames.has(name)) {
            node.addOutput(name, "IMAGE");
        }
    }

    const outputsByName = new Map((node.outputs ?? []).map((output) => [output.name, output]));
    node.outputs = desiredNames
        .map((name) => {
            const output = outputsByName.get(name);
            output.type = "IMAGE";
            return output;
        })
        .filter(Boolean);
    syncOutputLinkSlots(node);
}

function updateScheduledGenerator(node) {
    const countWidget = widgetByName(node, "reference_count");
    const count = Math.max(1, Math.min(MAX_REFERENCES, Number(countWidget?.value ?? MAX_REFERENCES)));
    if (countWidget) {
        countWidget.value = count;
    }

    for (let inputIndex = (node.inputs?.length ?? 0) - 1; inputIndex >= 0; inputIndex -= 1) {
        const referenceInfo = referenceInputInfo(node.inputs[inputIndex]);
        if (referenceInfo !== null && referenceInfo.number > count) {
            node.removeInput(inputIndex);
        }
    }

    const existingImages = new Set();
    const existingMasks = new Set();
    for (const input of node.inputs ?? []) {
        const referenceInfo = referenceInputInfo(input);
        if (!referenceInfo) {
            continue;
        }
        if (referenceInfo.isMask) {
            existingMasks.add(referenceInfo.number);
        } else {
            existingImages.add(referenceInfo.number);
        }
    }
    for (let index = 1; index <= count; index += 1) {
        if (!existingImages.has(index)) {
            node.addInput(`reference_${index}`, "IMAGE");
        }
        if (!existingMasks.has(index)) {
            node.addInput(`reference_${index}_mask`, "IMAGE");
        }
    }
    resizeNode(node);
}

function updateScheduledGeneratorWithSAM(node) {
    const countWidget = widgetByName(node, "reference_count");
    const count = Math.max(1, Math.min(MAX_REFERENCES, Number(countWidget?.value ?? MAX_REFERENCES)));
    if (countWidget) {
        countWidget.value = count;
    }

    for (let inputIndex = (node.inputs?.length ?? 0) - 1; inputIndex >= 0; inputIndex -= 1) {
        const referenceInfo = referenceInputInfo(node.inputs[inputIndex]);
        if (referenceInfo !== null && (referenceInfo.isMask || referenceInfo.number > count)) {
            node.removeInput(inputIndex);
        }
    }

    const existingImages = new Set();
    for (const input of node.inputs ?? []) {
        const referenceInfo = referenceInputInfo(input);
        if (referenceInfo && !referenceInfo.isMask) {
            existingImages.add(referenceInfo.number);
        }
    }
    for (let index = 1; index <= count; index += 1) {
        if (!existingImages.has(index)) {
            node.addInput(`reference_${index}`, "IMAGE");
        }
    }
    resizeNode(node);
}

function updateMultiReferenceMask(node) {
    const countWidget = widgetByName(node, "reference_count");
    const count = Math.max(1, Math.min(MAX_REFERENCES, Number(countWidget?.value ?? MAX_REFERENCES)));
    if (countWidget) {
        countWidget.value = count;
    }

    for (let inputIndex = (node.inputs?.length ?? 0) - 1; inputIndex >= 0; inputIndex -= 1) {
        const referenceNumber = referenceTrackInputNumber(node.inputs[inputIndex]);
        if (referenceNumber !== null && referenceNumber > count) {
            node.removeInput(inputIndex);
        }
    }

    const existingTracks = new Set(
        (node.inputs ?? [])
            .map(referenceTrackInputNumber)
            .filter((value) => value !== null)
    );
    for (let index = 1; index <= count; index += 1) {
        if (!existingTracks.has(index)) {
            node.addInput(`reference_${index}_track_data`, "SAM3_TRACK_DATA");
        }
    }
    updateMultiReferenceMaskOutputs(node, count);
    resizeNode(node);
}

function addUpdateButton(node, label, callback) {
    if (node.widgets?.some((widget) => widget.name === label)) {
        return;
    }
    node.addWidget("button", label, null, () => callback(node));
}

function matrixImageUrl(item) {
    const image = item?.image ?? item;
    if (typeof image === "string") {
        if (/^(https?:)?\/\//.test(image) || image.startsWith("/")) {
            return image;
        }
        const params = new URLSearchParams();
        params.set("filename", image);
        params.set("type", "temp");
        params.set("subfolder", "");
        return api.apiURL(`/view?${params.toString()}`);
    }
    if (image?.url) {
        return image.url;
    }
    const params = new URLSearchParams();
    params.set("filename", image?.filename ?? "");
    params.set("type", image?.type ?? "temp");
    params.set("subfolder", image?.subfolder ?? "");
    return api.apiURL(`/view?${params.toString()}`);
}

function normalizeMatrixItem(item, index) {
    const image = item?.image ?? item;
    const filename = image?.filename ?? (typeof image === "string" ? image : "");
    const frameMatch = /frame(\d+)/i.exec(filename);
    const rawChunkIndex = item?.chunk_index;
    const chunkNumber = item?.chunk_number_1_based ?? (Number.isFinite(Number(rawChunkIndex)) ? Number(rawChunkIndex) + 1 : "-");
    const frame = item?.frame_1_based ?? (frameMatch ? Number(frameMatch[1]) : null);
    const displayKind = item?.display_kind ?? item?.kind ?? "image";
    return {
        ...(typeof item === "object" && item !== null ? item : {}),
        filename,
        subfolder: image?.subfolder ?? item?.subfolder ?? "",
        type: image?.type ?? item?.type ?? "temp",
        batch_index: item?.batch_index ?? item?.index ?? index,
        chunk_index: rawChunkIndex ?? "-",
        chunk_number_1_based: chunkNumber,
        kind: item?.kind ?? displayKind,
        display_kind: displayKind,
        frame_1_based: frame,
        output_range_1_based_inclusive: item?.output_range_1_based_inclusive,
        label:
            item?.label ??
            `${String(item?.batch_index ?? item?.index ?? index).padStart(3, "0")} | chunk ${chunkNumber} | ${displayKind}${
                frame ? ` | frame ${frame}` : ""
            }`,
    };
}

function normalizeMatrixPayload(payload) {
    let value = payload;
    if (
        Array.isArray(value) &&
        value.every((item) => typeof item === "string") &&
        value.includes("items") &&
        value.includes("count")
    ) {
        return { items: [] };
    }
    if (Array.isArray(value) && value.length === 1 && Array.isArray(value[0]?.items)) {
        value = value[0];
    }
    if (typeof value === "string") {
        try {
            value = JSON.parse(value);
        } catch {
            return { items: [] };
        }
    }
    if (value?.matrix) {
        value = value.matrix;
    }
    if (Array.isArray(value?.items)) {
        return { ...value, items: value.items.map(normalizeMatrixItem) };
    }
    if (Array.isArray(value)) {
        return { items: value.map(normalizeMatrixItem) };
    }
    return { ...(value ?? {}), items: [] };
}

function ensureMatrixWidget(node) {
    if (node.scailMatrixContainer) {
        return node.scailMatrixContainer;
    }
    const container = document.createElement("div");
    container.className = "scail-keyframe-matrix";
    container.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "max-height:560px",
        "overflow:auto",
        "padding:8px",
        "border:1px solid rgba(140,148,160,.35)",
        "border-radius:6px",
        "background:#111827",
        "color:#e5e7eb",
        "font:12px/1.35 system-ui,-apple-system,BlinkMacSystemFont,sans-serif",
    ].join(";");
    container.textContent = "Run this node to build the keyframe matrix.";

    if (node.addDOMWidget) {
        const widget = node.addDOMWidget("keyframe_matrix", "Keyframe Matrix", container, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.computeSize = () => [
            Math.max(520, node.size?.[0] ?? 520),
            Math.min(620, Math.max(160, container.scrollHeight + 20)),
        ];
    } else {
        node.addWidget("text", "keyframe_matrix", "Run node to view keyframe matrix", () => {}, {
            serialize: false,
        });
    }
    node.scailMatrixContainer = container;
    return container;
}

function renderMatrix(node, matrix) {
    const container = ensureMatrixWidget(node);
    const normalizedMatrix = normalizeMatrixPayload(matrix);
    const items = Array.isArray(normalizedMatrix?.items) ? normalizedMatrix.items : [];
    container.replaceChildren();

    const header = document.createElement("div");
    header.style.cssText = "display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = `Keyframe Matrix (${items.length})`;
    title.style.cssText = "font-weight:700;font-size:13px;";
    const hint = document.createElement("div");
    hint.textContent = "Open/download links point to original PNG files.";
    hint.style.cssText = "opacity:.72;text-align:right;";
    header.append(title, hint);
    container.append(header);

    if (!items.length) {
        const empty = document.createElement("div");
        empty.textContent = "No keyframes returned.";
        empty.style.opacity = ".72";
        container.append(empty);
        resizeNode(node);
        return;
    }

    const grid = document.createElement("div");
    grid.style.cssText = [
        "display:grid",
        "grid-template-columns:repeat(auto-fill,minmax(156px,1fr))",
        "gap:8px",
    ].join(";");

    for (const item of items) {
        const url = matrixImageUrl(item);
        const card = document.createElement("div");
        card.style.cssText = [
            "background:#f8fafc",
            "color:#111827",
            "border:1px solid #cbd5e1",
            "border-radius:6px",
            "overflow:hidden",
        ].join(";");

        const link = document.createElement("a");
        link.href = url;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.title = "Open original image";

        const img = document.createElement("img");
        img.src = url;
        img.loading = "lazy";
        img.style.cssText = "display:block;width:100%;height:128px;object-fit:contain;background:#0f172a;";
        link.append(img);

        const body = document.createElement("div");
        body.style.cssText = "padding:7px;";
        const name = document.createElement("div");
        name.textContent = item.label;
        name.style.cssText = "font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
        const frame = document.createElement("div");
        frame.textContent = item.frame_1_based ? `frame ${item.frame_1_based}` : item.filename;
        frame.style.cssText = "margin-top:2px;color:#475569;";
        const range = document.createElement("div");
        const outputRange = item.output_range_1_based_inclusive;
        range.textContent = Array.isArray(outputRange) ? `out ${outputRange[0]}-${outputRange[1]}` : "";
        range.style.cssText = "color:#64748b;";

        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;gap:6px;margin-top:7px;";
        const open = document.createElement("a");
        open.href = url;
        open.target = "_blank";
        open.rel = "noreferrer";
        open.textContent = "Open";
        const download = document.createElement("a");
        download.href = url;
        download.download = item.filename || `keyframe_${item.batch_index}.png`;
        download.textContent = "Download";
        const copy = document.createElement("button");
        copy.type = "button";
        copy.textContent = "Copy URL";
        copy.onclick = async () => {
            await navigator.clipboard?.writeText(new URL(url, location.href).toString());
            copy.textContent = "Copied";
            setTimeout(() => {
                copy.textContent = "Copy URL";
            }, 900);
        };
        for (const action of [open, download, copy]) {
            action.style.cssText = [
                "font:inherit",
                "font-size:11px",
                "color:#0f172a",
                "background:#e2e8f0",
                "border:1px solid #cbd5e1",
                "border-radius:4px",
                "padding:3px 5px",
                "text-decoration:none",
                "cursor:pointer",
            ].join(";");
        }
        actions.append(open, download, copy);
        body.append(name, frame, range, actions);
        card.append(link, body);
        grid.append(card);
    }

    container.append(grid);
    resizeNode(node);
}

function clampNumber(value, min, max) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return min;
    }
    return Math.max(min, Math.min(max, number));
}

function roundRatio(value) {
    return Number(clampNumber(value, 0, 1).toFixed(6));
}

function manualTileMinRatio(node) {
    const widget = widgetByName(node, "min_tile_ratio");
    return Math.max(0.05, Math.min(0.45, Number(widget?.value ?? 0.2)));
}

function manualTileAlign(node) {
    const widget = widgetByName(node, "tile_align");
    const requested = Math.max(32, Math.min(256, Math.round(Number(widget?.value ?? 32))));
    return Math.max(32, Math.min(256, Math.ceil(requested / 32) * 32));
}

function manualTileSnapThreshold(node, axis) {
    const sourceSize = manualTileSourceSize(node);
    const dimension = axis === "x" ? sourceSize?.width : sourceSize?.height;
    if (Number(dimension) > 1) {
        return Math.max(0.002, Math.min(0.04, 12 / Number(dimension)));
    }
    return 0.015;
}

function manualTileSourceSize(node) {
    const previewSize = node.scailManualTilePreview?.source_size;
    if (Array.isArray(previewSize) && Number(previewSize[0]) > 0 && Number(previewSize[1]) > 0) {
        return { width: Number(previewSize[0]), height: Number(previewSize[1]) };
    }
    return null;
}

function manualTileAspect(node) {
    const sourceSize = manualTileSourceSize(node);
    if (sourceSize) {
        return Math.max(0.25, Math.min(4, sourceSize.height / sourceSize.width));
    }
    const outputWidth = Number(widgetByName(node, "output_width")?.value ?? 0);
    const outputHeight = Number(widgetByName(node, "output_height")?.value ?? 0);
    if (outputWidth > 0 && outputHeight > 0) {
        return Math.max(0.25, Math.min(4, outputHeight / outputWidth));
    }
    return 960 / 548;
}

function manualTileAspectRatioCss(node) {
    const sourceSize = manualTileSourceSize(node);
    if (sourceSize) {
        return `${Math.max(1, Math.round(sourceSize.width))} / ${Math.max(1, Math.round(sourceSize.height))}`;
    }
    const outputWidth = Number(widgetByName(node, "output_width")?.value ?? 0);
    const outputHeight = Number(widgetByName(node, "output_height")?.value ?? 0);
    if (outputWidth > 0 && outputHeight > 0) {
        return `${Math.max(1, Math.round(outputWidth))} / ${Math.max(1, Math.round(outputHeight))}`;
    }
    return "548 / 960";
}

function manualTilePreviewItems(node) {
    const preview = node.scailManualTilePreview ?? { items: [] };
    return Array.isArray(preview.items) ? preview.items : [];
}

function manualTileEditorInnerWidth(node) {
    return Math.max(280, Math.round(Number(node.size?.[0] ?? 460) - 18));
}

function manualTileStageHeight(node) {
    return Math.max(180, Math.round(manualTileEditorInnerWidth(node) * manualTileAspect(node)));
}

function manualTileEditorHeight(node) {
    const hasFrameSlider = manualTilePreviewItems(node).length > 1;
    return manualTileStageHeight(node) + (hasFrameSlider ? 154 : 124);
}

function normalizeManualTileAxis(start, end, dimension, minRatio, align) {
    const usePixels = Number(dimension) > 1;
    let first = clampNumber(start, 0, 1);
    let second = clampNumber(end, 0, 1);
    if (second < first) {
        [first, second] = [second, first];
    }
    if (!usePixels) {
        const minSpan = Math.max(0.01, Math.min(0.95, Number(minRatio)));
        if (second - first < minSpan) {
            const center = (first + second) / 2;
            first = center - minSpan / 2;
            second = center + minSpan / 2;
            if (first < 0) {
                second -= first;
                first = 0;
            }
            if (second > 1) {
                first -= second - 1;
                second = 1;
            }
        }
        return [roundRatio(first), roundRatio(second)];
    }

    const size = Math.max(1, Math.round(Number(dimension)));
    const step = Math.max(1, Math.round(Number(align)));
    let a = Math.round((first * size) / step) * step;
    let b = Math.round((second * size) / step) * step;
    a = Math.max(0, Math.min(size - 1, a));
    b = Math.max(1, Math.min(size, b));
    if (b <= a) {
        b = Math.min(size, a + 1);
        a = Math.max(0, b - 1);
    }

    const minPixels = Math.max(1, Math.round(size * Math.max(0.01, Math.min(0.95, Number(minRatio)))));
    if (b - a < minPixels) {
        const center = (a + b) / 2;
        a = Math.round(center - minPixels / 2);
        b = a + minPixels;
        if (a < 0) {
            b -= a;
            a = 0;
        }
        if (b > size) {
            a -= b - size;
            b = size;
        }
    }
    return [roundRatio(a / size), roundRatio(b / size)];
}

function normalizeManualTileRect(node, raw) {
    const sourceSize = manualTileSourceSize(node);
    const minRatio = manualTileMinRatio(node);
    const align = manualTileAlign(node);
    let x0 = Number(raw?.x0 ?? raw?.x ?? 0);
    let y0 = Number(raw?.y0 ?? raw?.y ?? 0);
    let x1 = Number(raw?.x1 ?? (Number(raw?.x ?? 0) + Number(raw?.w ?? raw?.width ?? 0.5)));
    let y1 = Number(raw?.y1 ?? (Number(raw?.y ?? 0) + Number(raw?.h ?? raw?.height ?? 0.5)));
    [x0, x1] = normalizeManualTileAxis(x0, x1, sourceSize?.width, minRatio, align);
    [y0, y1] = normalizeManualTileAxis(y0, y1, sourceSize?.height, minRatio, align);
    return { x0, y0, x1, y1 };
}

const MANUAL_TILE_LAYOUT_STORAGE_PREFIX = "scail_manual_tile_layout";

function manualTileStorageNodeId(node) {
    return String(node?.id ?? node?.title ?? "node");
}

function manualTileStorageSizeKey(node) {
    const sourceSize = manualTileSourceSize(node);
    const sizeKey = sourceSize
        ? `${Math.round(sourceSize.width)}x${Math.round(sourceSize.height)}`
        : "unknown";
    return `${MANUAL_TILE_LAYOUT_STORAGE_PREFIX}:${manualTileStorageNodeId(node)}:${sizeKey}`;
}

function manualTileStorageLatestKey(node) {
    return `${MANUAL_TILE_LAYOUT_STORAGE_PREFIX}:${manualTileStorageNodeId(node)}:latest`;
}

function normalizeStoredManualTileLayout(node, value) {
    if (!Array.isArray(value?.tiles) || !value.tiles.length) {
        return null;
    }
    return {
        tiles: value.tiles.slice(0, MAX_TILES).map((tile) => normalizeManualTileRect(node, tile)),
    };
}

function readStoredManualTileLayout(node) {
    try {
        const keys = Array.from(new Set([
            manualTileStorageSizeKey(node),
            manualTileStorageLatestKey(node),
        ]));
        for (const key of keys) {
            const raw = globalThis.localStorage?.getItem(key);
            if (!raw) {
                continue;
            }
            const layout = normalizeStoredManualTileLayout(node, JSON.parse(raw));
            if (layout) {
                return layout;
            }
        }
    } catch {
        return null;
    }
    return null;
}

function storeManualTileLayout(node, layout) {
    const normalized = normalizeStoredManualTileLayout(node, layout);
    if (!normalized) {
        return;
    }
    const sourceSize = manualTileSourceSize(node);
    const payload = {
        ...normalized,
        source_size: sourceSize ? [Math.round(sourceSize.width), Math.round(sourceSize.height)] : null,
    };
    try {
        const serialized = JSON.stringify(payload);
        globalThis.localStorage?.setItem(manualTileStorageSizeKey(node), serialized);
        globalThis.localStorage?.setItem(manualTileStorageLatestKey(node), serialized);
    } catch {}
}

function manualTilesFromSplit(node, splitX = 0.5, splitY = 0.5) {
    const minRatio = manualTileMinRatio(node);
    const x = Math.max(minRatio, Math.min(1 - minRatio, Number(splitX)));
    const y = Math.max(minRatio, Math.min(1 - minRatio, Number(splitY)));
    return [
        normalizeManualTileRect(node, { x0: 0, y0: 0, x1: x, y1: y }),
        normalizeManualTileRect(node, { x0: x, y0: 0, x1: 1, y1: y }),
        normalizeManualTileRect(node, { x0: 0, y0: y, x1: x, y1: 1 }),
        normalizeManualTileRect(node, { x0: x, y0: y, x1: 1, y1: 1 }),
    ];
}

function manualTileEdgeValues(tiles, skipIndex, axis) {
    const keys = axis === "x" ? ["x0", "x1"] : ["y0", "y1"];
    const values = [0, 1];
    for (const [index, tile] of tiles.entries()) {
        if (index === skipIndex) {
            continue;
        }
        values.push(Number(tile[keys[0]]), Number(tile[keys[1]]));
    }
    return values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
}

function snapManualTileEdge(value, edges, threshold) {
    let best = Number(value);
    let bestDelta = Number.POSITIVE_INFINITY;
    for (const edge of edges) {
        const delta = Math.abs(Number(value) - Number(edge));
        if (delta <= threshold && delta < bestDelta) {
            best = Number(edge);
            bestDelta = delta;
        }
    }
    return best;
}

function snapManualTileMove(node, tile, tiles, tileIndex) {
    const next = { ...tile };
    for (const axis of ["x", "y"]) {
        const minKey = axis === "x" ? "x0" : "y0";
        const maxKey = axis === "x" ? "x1" : "y1";
        const edges = manualTileEdgeValues(tiles, tileIndex, axis);
        const threshold = manualTileSnapThreshold(node, axis);
        const candidates = [
            snapManualTileEdge(next[minKey], edges, threshold) - next[minKey],
            snapManualTileEdge(next[maxKey], edges, threshold) - next[maxKey],
        ];
        const delta = candidates.reduce(
            (best, item) => (Math.abs(item) < Math.abs(best) ? item : best),
            candidates[0]
        );
        if (Math.abs(delta) > 0 && Math.abs(delta) <= threshold) {
            const span = next[maxKey] - next[minKey];
            let start = next[minKey] + delta;
            start = Math.max(0, Math.min(1 - span, start));
            next[minKey] = start;
            next[maxKey] = start + span;
        }
    }
    return next;
}

function snapManualTileResize(node, tile, tiles, tileIndex) {
    const next = { ...tile };
    next.x1 = snapManualTileEdge(next.x1, manualTileEdgeValues(tiles, tileIndex, "x"), manualTileSnapThreshold(node, "x"));
    next.y1 = snapManualTileEdge(next.y1, manualTileEdgeValues(tiles, tileIndex, "y"), manualTileSnapThreshold(node, "y"));
    return next;
}

function uniqueSortedManualEdges(values) {
    const sorted = values
        .map((value) => roundRatio(value))
        .filter((value) => Number.isFinite(value))
        .sort((a, b) => a - b);
    const result = [];
    for (const value of sorted) {
        if (!result.length || Math.abs(value - result[result.length - 1]) > 1e-6) {
            result.push(value);
        }
    }
    return result;
}

function analyzeManualTileCoverage(tiles) {
    const normalizedTiles = (Array.isArray(tiles) ? tiles : []).map((tile) => ({
        x0: clampNumber(tile.x0, 0, 1),
        y0: clampNumber(tile.y0, 0, 1),
        x1: clampNumber(tile.x1, 0, 1),
        y1: clampNumber(tile.y1, 0, 1),
    }));
    const xEdges = uniqueSortedManualEdges([0, 1, ...normalizedTiles.flatMap((tile) => [tile.x0, tile.x1])]);
    const yEdges = uniqueSortedManualEdges([0, 1, ...normalizedTiles.flatMap((tile) => [tile.y0, tile.y1])]);
    const rowRuns = [];
    let uncoveredArea = 0;
    for (let yIndex = 0; yIndex < yEdges.length - 1; yIndex += 1) {
        const y0 = yEdges[yIndex];
        const y1 = yEdges[yIndex + 1];
        if (y1 - y0 <= 1e-6) {
            continue;
        }
        let run = null;
        for (let xIndex = 0; xIndex < xEdges.length - 1; xIndex += 1) {
            const x0 = xEdges[xIndex];
            const x1 = xEdges[xIndex + 1];
            if (x1 - x0 <= 1e-6) {
                continue;
            }
            const cx = (x0 + x1) / 2;
            const cy = (y0 + y1) / 2;
            const covered = normalizedTiles.some(
                (tile) => cx >= tile.x0 - 1e-6 && cx <= tile.x1 + 1e-6 && cy >= tile.y0 - 1e-6 && cy <= tile.y1 + 1e-6
            );
            if (covered) {
                if (run) {
                    rowRuns.push(run);
                    run = null;
                }
                continue;
            }
            uncoveredArea += (x1 - x0) * (y1 - y0);
            if (run && Math.abs(run.x1 - x0) <= 1e-6) {
                run.x1 = x1;
            } else {
                if (run) {
                    rowRuns.push(run);
                }
                run = { x0, y0, x1, y1 };
            }
        }
        if (run) {
            rowRuns.push(run);
        }
    }

    const gaps = [];
    const activeBySpan = new Map();
    const sortedRuns = rowRuns
        .slice()
        .sort((a, b) => a.x0 - b.x0 || a.x1 - b.x1 || a.y0 - b.y0 || a.y1 - b.y1);
    for (const run of sortedRuns) {
        const key = `${run.x0}:${run.x1}`;
        const previous = activeBySpan.get(key);
        if (
            previous &&
            Math.abs(previous.x0 - run.x0) <= 1e-6 &&
            Math.abs(previous.x1 - run.x1) <= 1e-6 &&
            Math.abs(previous.y1 - run.y0) <= 1e-6
        ) {
            previous.y1 = run.y1;
        } else {
            const gap = { ...run };
            gaps.push(gap);
            activeBySpan.set(key, gap);
        }
    }
    gaps.sort((a, b) => (b.x1 - b.x0) * (b.y1 - b.y0) - (a.x1 - a.x0) * (a.y1 - a.y0));
    return {
        gaps: gaps.map((gap) => ({
            x0: roundRatio(gap.x0),
            y0: roundRatio(gap.y0),
            x1: roundRatio(gap.x1),
            y1: roundRatio(gap.y1),
        })),
        uncoveredArea,
        uncoveredRatio: uncoveredArea,
    };
}

function parseManualTileLayout(node) {
    const layoutWidget = widgetByName(node, "layout_json");
    let value = {};
    try {
        value = JSON.parse(layoutWidget?.value || "{}");
    } catch {
        value = {};
    }
    if (Array.isArray(value?.tiles) && value.tiles.length) {
        return {
            tiles: value.tiles.slice(0, MAX_TILES).map((tile) => normalizeManualTileRect(node, tile)),
        };
    }
    const storedLayout = readStoredManualTileLayout(node);
    if (storedLayout) {
        return storedLayout;
    }
    const splitX = Math.max(0.01, Math.min(0.99, Number(value.split_x ?? 0.5)));
    const splitY = Math.max(0.01, Math.min(0.99, Number(value.split_y ?? 0.5)));
    return {
        split_x: splitX,
        split_y: splitY,
        tiles: manualTilesFromSplit(node, splitX, splitY),
    };
}

function normalizeManualTilePreview(payload) {
    let value = payload;
    if (Array.isArray(value) && value.length === 1) {
        value = value[0];
    }
    if (typeof value === "string") {
        try {
            value = JSON.parse(value);
        } catch {
            return { items: [] };
        }
    }
    if (Array.isArray(value?.items)) {
        return value;
    }
    if (Array.isArray(value)) {
        return { items: value };
    }
    return { ...(value ?? {}), items: [] };
}

function extractManualTilePreview(message) {
    const candidates = [
        message?.scail_manual_tile_preview,
        message?.scail_manual_tile_preview_json,
        message?.ui?.scail_manual_tile_preview,
        message?.ui?.scail_manual_tile_preview_json,
        message?.output?.scail_manual_tile_preview,
        message?.output?.scail_manual_tile_preview_json,
    ];
    for (const candidate of candidates) {
        const normalized = normalizeManualTilePreview(candidate);
        if (normalized.items.length) {
            return normalized;
        }
    }
    return null;
}

function writeManualTileLayout(node, tiles, selectedIndex = 0) {
    const normalizedTiles = (Array.isArray(tiles) ? tiles : [])
        .slice(0, MAX_TILES)
        .map((tile) => normalizeManualTileRect(node, tile));
    const layout = {
        tiles: normalizedTiles.length ? normalizedTiles : manualTilesFromSplit(node, 0.5, 0.5),
    };
    const layoutWidget = widgetByName(node, "layout_json");
    const serialized = JSON.stringify(layout);
    if (layoutWidget) {
        if (layoutWidget.value !== serialized) {
            layoutWidget.value = serialized;
        }
    }
    node.scailManualTileLayout = layout;
    node.scailManualTileSelectedIndex = Math.max(
        0,
        Math.min(layout.tiles.length - 1, Number(selectedIndex ?? 0))
    );
    storeManualTileLayout(node, layout);
}

function setManualTileTiles(node, tiles, selectedIndex = 0) {
    writeManualTileLayout(node, tiles, selectedIndex);
    renderManualTileEditor(node);
}

function setManualTileLayout(node, splitX, splitY) {
    setManualTileTiles(node, manualTilesFromSplit(node, splitX, splitY), 0);
}

function manualTilePixelRect(node, tile) {
    const sourceSize = manualTileSourceSize(node);
    if (!sourceSize) {
        return null;
    }
    const x0 = Math.round(tile.x0 * sourceSize.width);
    const y0 = Math.round(tile.y0 * sourceSize.height);
    const x1 = Math.round(tile.x1 * sourceSize.width);
    const y1 = Math.round(tile.y1 * sourceSize.height);
    return {
        x0,
        y0,
        x1,
        y1,
        width: Math.max(1, x1 - x0),
        height: Math.max(1, y1 - y0),
    };
}

function createManualTileButton(label, onClick, disabled = false) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.disabled = disabled;
    button.style.cssText = [
        "font:inherit",
        "color:" + (disabled ? "rgba(15,23,42,.45)" : "#0f172a"),
        "background:#e2e8f0",
        "border:1px solid #cbd5e1",
        "border-radius:4px",
        "padding:4px 7px",
        "cursor:" + (disabled ? "default" : "pointer"),
    ].join(";");
    if (!disabled) {
        button.onclick = onClick;
    }
    return button;
}

function ensureManualTileEditor(node) {
    if (node.scailManualTileContainer) {
        return node.scailManualTileContainer;
    }
    const layoutWidget = widgetByName(node, "layout_json");
    setWidgetVisible(layoutWidget, false);

    const container = document.createElement("div");
    container.className = "scail-manual-tile-editor";
    container.style.cssText = [
        "box-sizing:border-box",
        "width:100%",
        "padding:8px",
        "border:1px solid rgba(140,148,160,.35)",
        "border-radius:6px",
        "background:#101827",
        "color:#e5e7eb",
        "font:12px/1.35 system-ui,-apple-system,BlinkMacSystemFont,sans-serif",
    ].join(";");

    if (node.addDOMWidget) {
        const widget = node.addDOMWidget("manual_tile_editor", "Manual Tile Editor", container, {
            serialize: false,
            hideOnZoom: false,
        });
        widget.computeSize = () => [
            Math.max(460, node.size?.[0] ?? 460),
            Math.max(320, manualTileEditorHeight(node)),
        ];
    } else {
        node.addWidget("text", "manual_tile_editor", "Manual Tile Editor", () => {}, { serialize: false });
    }
    node.scailManualTileContainer = container;
    renderManualTileEditor(node);
    return container;
}

function renderManualTileEditor(node) {
    const container = node.scailManualTileContainer;
    if (!container) {
        return;
    }
    const current = node.scailManualTileLayout ?? parseManualTileLayout(node);
    const tiles = Array.isArray(current.tiles) && current.tiles.length
        ? current.tiles.map((tile) => normalizeManualTileRect(node, tile))
        : manualTilesFromSplit(node, current.split_x ?? 0.5, current.split_y ?? 0.5);
    const selectedIndex = Math.max(
        0,
        Math.min(tiles.length - 1, Number(node.scailManualTileSelectedIndex ?? 0))
    );
    writeManualTileLayout(node, tiles, selectedIndex);
    node.scailManualTileSelectedIndex = selectedIndex;

    const previewItems = manualTilePreviewItems(node);
    const previewIndex = Math.max(
        0,
        Math.min(previewItems.length - 1, Number(node.scailManualTilePreviewIndex ?? 0))
    );
    node.scailManualTilePreviewIndex = previewIndex;
    const previewItem = previewItems[previewIndex];
    const sourceSize = manualTileSourceSize(node);
    const selectedPixelRect = manualTilePixelRect(node, tiles[selectedIndex]);
    const align = manualTileAlign(node);
    const coverage = analyzeManualTileCoverage(tiles);
    const stageHeight = manualTileStageHeight(node);
    const editorHeight = manualTileEditorHeight(node);
    node.scailManualTileEditorHeight = editorHeight;

    container.replaceChildren();

    const header = document.createElement("div");
    header.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = "Manual Tile Rectangles";
    title.style.cssText = "font-weight:700;font-size:13px;";
    const value = document.createElement("div");
    value.textContent = selectedPixelRect
        ? `${tiles.length} tiles / T${selectedIndex + 1} ${selectedPixelRect.width}x${selectedPixelRect.height}px / snap ${align}px`
        : `${tiles.length} tiles / snap ${align}px`;
    value.style.cssText = "opacity:.78;text-align:right;";
    header.append(title, value);

    const stage = document.createElement("div");
    let previewImageElement = null;
    stage.style.cssText = [
        "position:relative",
        "width:100%",
        "aspect-ratio:" + manualTileAspectRatioCss(node),
        "border:1px solid rgba(226,232,240,.45)",
        "border-radius:6px",
        "overflow:hidden",
        "background:#172033",
        "touch-action:none",
        "cursor:crosshair",
    ].join(";");

    if (previewItem) {
        const image = document.createElement("img");
        image.src = matrixImageUrl(previewItem);
        image.loading = "eager";
        image.draggable = false;
        image.style.cssText = [
            "position:absolute",
            "inset:0",
            "width:100%",
            "height:100%",
            "object-fit:fill",
            "user-select:none",
            "pointer-events:none",
        ].join(";");
        stage.append(image);
        previewImageElement = image;
    } else {
        const empty = document.createElement("div");
        empty.textContent = "Run this node once to load preview frames.";
        empty.style.cssText = [
            "position:absolute",
            "inset:0",
            "display:flex",
            "align-items:center",
            "justify-content:center",
            "color:rgba(226,232,240,.72)",
            "text-align:center",
            "padding:16px",
            "pointer-events:none",
        ].join(";");
        stage.append(empty);
    }

    for (const gap of coverage.gaps) {
        const overlay = document.createElement("div");
        overlay.title = "Uncovered area";
        overlay.style.cssText = [
            "position:absolute",
            "box-sizing:border-box",
            "left:" + gap.x0 * 100 + "%",
            "top:" + gap.y0 * 100 + "%",
            "width:" + Math.max(0.1, (gap.x1 - gap.x0) * 100) + "%",
            "height:" + Math.max(0.1, (gap.y1 - gap.y0) * 100) + "%",
            "background:rgba(239,68,68,.30)",
            "border:1px dashed rgba(254,202,202,.85)",
            "pointer-events:none",
        ].join(";");
        stage.append(overlay);
    }

    const pointFromEvent = (event) => {
        const rect = stage.getBoundingClientRect();
        return {
            x: clampNumber((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1),
            y: clampNumber((event.clientY - rect.top) / Math.max(1, rect.height), 0, 1),
        };
    };
    const tileContainsPoint = (tile, point) => (
        point.x >= tile.x0 - 1e-6 &&
        point.x <= tile.x1 + 1e-6 &&
        point.y >= tile.y0 - 1e-6 &&
        point.y <= tile.y1 + 1e-6
    );
    const tileResizeHit = (tile, point) => {
        const rect = stage.getBoundingClientRect();
        const padX = Math.max(0.01, 18 / Math.max(1, rect.width));
        const padY = Math.max(0.01, 18 / Math.max(1, rect.height));
        return point.x >= tile.x1 - padX && point.x <= tile.x1 + padX &&
            point.y >= tile.y1 - padY && point.y <= tile.y1 + padY;
    };
    const setRegionTileStyle = (element, tile) => {
        if (!element) {
            return;
        }
        element.style.left = tile.x0 * 100 + "%";
        element.style.top = tile.y0 * 100 + "%";
        element.style.width = Math.max(0.1, (tile.x1 - tile.x0) * 100) + "%";
        element.style.height = Math.max(0.1, (tile.y1 - tile.y0) * 100) + "%";
    };
    const tileRenderOrder = tiles
        .map((_tile, index) => index)
        .filter((index) => index !== selectedIndex);
    tileRenderOrder.push(selectedIndex);
    const regionElements = new Map();
    const resolvePointerTileIndex = (point, fallbackIndex) => {
        const hits = tileRenderOrder.filter((index) => tileContainsPoint(tiles[index], point));
        if (hits.includes(selectedIndex)) {
            return selectedIndex;
        }
        if (Number.isInteger(fallbackIndex) && hits.includes(fallbackIndex)) {
            return fallbackIndex;
        }
        return hits.length ? hits[hits.length - 1] : fallbackIndex;
    };

    const beginTileDrag = (event, requestedTileIndex, mode, regionElement) => {
        event.preventDefault();
        event.stopPropagation();
        const pointerTileIndex = resolvePointerTileIndex(pointFromEvent(event), requestedTileIndex);
        const tileIndex = Math.max(0, Math.min(tiles.length - 1, Number(pointerTileIndex ?? requestedTileIndex ?? 0)));
        const activeRegionElement = regionElements.get(tileIndex) ?? regionElement;
        const dragMode = tileResizeHit(tiles[tileIndex], pointFromEvent(event)) ? "resize" : mode;
        node.scailManualTileSelectedIndex = tileIndex;
        if (activeRegionElement) {
            activeRegionElement.style.zIndex = "120";
        }
        const target = event.currentTarget;
        target.setPointerCapture?.(event.pointerId);
        const startPoint = pointFromEvent(event);
        const startTiles = tiles.map((tile) => ({ ...tile }));
        const startTile = startTiles[tileIndex];
        const apply = (moveEvent) => {
            moveEvent.preventDefault();
            moveEvent.stopPropagation();
            const point = pointFromEvent(moveEvent);
            const dx = point.x - startPoint.x;
            const dy = point.y - startPoint.y;
            const nextTiles = startTiles.map((tile) => ({ ...tile }));
            let nextTile;
            if (dragMode === "resize") {
                nextTile = {
                    ...startTile,
                    x1: startTile.x1 + dx,
                    y1: startTile.y1 + dy,
                };
                nextTile = snapManualTileResize(node, nextTile, startTiles, tileIndex);
            } else {
                const width = startTile.x1 - startTile.x0;
                const height = startTile.y1 - startTile.y0;
                let x0 = startTile.x0 + dx;
                let y0 = startTile.y0 + dy;
                x0 = Math.max(0, Math.min(1 - width, x0));
                y0 = Math.max(0, Math.min(1 - height, y0));
                nextTile = {
                    x0,
                    y0,
                    x1: x0 + width,
                    y1: y0 + height,
                };
                nextTile = snapManualTileMove(node, nextTile, startTiles, tileIndex);
            }
            nextTiles[tileIndex] = normalizeManualTileRect(node, nextTile);
            node.scailManualTileFillMessage = "";
            writeManualTileLayout(node, nextTiles, tileIndex);
            setRegionTileStyle(activeRegionElement, nextTiles[tileIndex]);
            scheduleCanvas(node);
        };
        const end = (endEvent) => {
            endEvent.preventDefault();
            endEvent.stopPropagation();
            target.releasePointerCapture?.(endEvent.pointerId);
            window.removeEventListener("pointermove", apply, true);
            window.removeEventListener("pointerup", end, true);
            renderManualTileEditor(node);
        };
        window.addEventListener("pointermove", apply, true);
        window.addEventListener("pointerup", end, true);
        apply(event);
    };

    for (const index of tileRenderOrder) {
        const tile = tiles[index];
        const selected = index === selectedIndex;
        const region = document.createElement("div");
        region.textContent = String(index + 1);
        region.title = "Tile " + (index + 1);
        region.style.cssText = [
            "position:absolute",
            "box-sizing:border-box",
            "display:flex",
            "align-items:center",
            "justify-content:center",
            "left:" + tile.x0 * 100 + "%",
            "top:" + tile.y0 * 100 + "%",
            "width:" + Math.max(0.1, (tile.x1 - tile.x0) * 100) + "%",
            "height:" + Math.max(0.1, (tile.y1 - tile.y0) * 100) + "%",
            "border:" + (selected ? "2px solid #f8fafc" : "1px solid rgba(226,232,240,.55)"),
            "background:" + (selected ? "rgba(14,165,233,.28)" : "rgba(15,23,42,.20)"),
            "color:#f8fafc",
            "font-weight:800",
            "letter-spacing:0",
            "box-shadow:" + (selected ? "0 0 0 1px rgba(15,23,42,.85),0 0 16px rgba(14,165,233,.55)" : "none"),
            "z-index:" + (selected ? "100" : String(10 + index)),
            "cursor:move",
            "user-select:none",
        ].join(";");
        regionElements.set(index, region);
        region.addEventListener("pointerdown", (event) => beginTileDrag(event, index, "move", region));
        const handle = document.createElement("div");
        handle.title = "Resize tile " + (index + 1);
        handle.style.cssText = [
            "position:absolute",
            "right:0",
            "bottom:0",
            "width:14px",
            "height:14px",
            "background:#f8fafc",
            "border-left:1px solid rgba(15,23,42,.8)",
            "border-top:1px solid rgba(15,23,42,.8)",
            "cursor:nwse-resize",
        ].join(";");
        handle.addEventListener("pointerdown", (event) => beginTileDrag(event, index, "resize", region));
        region.append(handle);
        stage.append(region);
    }

    stage.addEventListener("pointerdown", (event) => {
        if (event.target !== stage) {
            return;
        }
        const point = pointFromEvent(event);
        let bestIndex = 0;
        let bestDistance = Number.POSITIVE_INFINITY;
        for (const [index, tile] of tiles.entries()) {
            const centerX = (tile.x0 + tile.x1) / 2;
            const centerY = (tile.y0 + tile.y1) / 2;
            const distance = Math.hypot(point.x - centerX, point.y - centerY);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = index;
            }
        }
        node.scailManualTileSelectedIndex = bestIndex;
        renderManualTileEditor(node);
    });

    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;flex-direction:column;align-items:stretch;gap:8px;margin-top:8px;";
    const previewLabel = (item, index) => item
        ? `Preview ${index + 1}/${previewItems.length}: ${item.label ?? `frame ${item.frame_1_based ?? ""}`}`
        : "Drag rectangles. The Python node adds overlap and final aligned repaint sizes.";
    const note = document.createElement("div");
    note.textContent = previewLabel(previewItem, previewIndex);
    note.style.cssText = "opacity:.72;min-width:180px;overflow-wrap:anywhere;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;";
    const tileSelector = document.createElement("div");
    tileSelector.style.cssText = "display:flex;gap:3px;align-items:center;flex-wrap:wrap;";
    for (const [index] of tiles.entries()) {
        const selectorButton = createManualTileButton("T" + (index + 1), () => {
            node.scailManualTileSelectedIndex = index;
            renderManualTileEditor(node);
        });
        if (index === selectedIndex) {
            selectorButton.style.background = "#38bdf8";
            selectorButton.style.borderColor = "#7dd3fc";
            selectorButton.style.color = "#082f49";
        }
        tileSelector.append(selectorButton);
    }
    const add = createManualTileButton("Add tile", () => {
        const base = tiles[selectedIndex] ?? { x0: 0.25, y0: 0.25, x1: 0.75, y1: 0.75 };
        const offset = Math.min(0.12, 0.03 * tiles.length);
        const width = Math.max(0.2, Math.min(0.5, base.x1 - base.x0));
        const height = Math.max(0.2, Math.min(0.5, base.y1 - base.y0));
        const x0 = Math.min(1 - width, Math.max(0, base.x0 + offset));
        const y0 = Math.min(1 - height, Math.max(0, base.y0 + offset));
        node.scailManualTileFillMessage = "";
        setManualTileTiles(node, [...tiles, { x0, y0, x1: x0 + width, y1: y0 + height }], tiles.length);
    }, tiles.length >= MAX_TILES);
    const remove = createManualTileButton("Delete", () => {
        const nextTiles = tiles.filter((_tile, index) => index !== selectedIndex);
        node.scailManualTileFillMessage = "";
        setManualTileTiles(node, nextTiles, Math.max(0, selectedIndex - 1));
    }, tiles.length <= 1);
    const fillGaps = createManualTileButton("Fill gaps", () => {
        const nextTiles = tiles.map((tile) => ({ ...tile }));
        const additions = [];
        let latest = analyzeManualTileCoverage(nextTiles);
        while (latest.gaps.length && nextTiles.length < MAX_TILES) {
            const previousArea = latest.uncoveredArea;
            const addition = normalizeManualTileRect(node, latest.gaps[0]);
            nextTiles.push(addition);
            const nextCoverage = analyzeManualTileCoverage(nextTiles);
            if (nextCoverage.uncoveredArea >= previousArea - 1e-8) {
                nextTiles.pop();
                break;
            }
            additions.push(addition);
            latest = nextCoverage;
        }
        if (!additions.length) {
            node.scailManualTileFillMessage = latest.gaps.length
                ? `Need ${latest.gaps.length} more tile(s), but max is ${MAX_TILES}.`
                : "No uncovered areas.";
            renderManualTileEditor(node);
            return;
        }
        const remaining = latest.gaps.length;
        node.scailManualTileFillMessage = remaining
            ? `Filled ${additions.length}; ${remaining} uncovered gap(s) remain because max is ${MAX_TILES}.`
            : `Filled ${additions.length} uncovered gap(s).`;
        setManualTileTiles(node, nextTiles, tiles.length);
    }, !coverage.gaps.length || tiles.length >= MAX_TILES);
    const reset = createManualTileButton("Reset 2x2", () => {
        node.scailManualTileFillMessage = "";
        setManualTileLayout(node, 0.5, 0.5);
    });
    actions.append(tileSelector, add, remove, fillGaps, reset);
    controls.append(note, actions);

    const status = document.createElement("div");
    status.style.cssText = "display:flex;gap:10px;flex-wrap:wrap;margin-top:6px;opacity:.76;";
    const sourceLabel = sourceSize ? `${sourceSize.width}x${sourceSize.height} source` : "source size after first run";
    const selectedLabel = selectedPixelRect
        ? `T${selectedIndex + 1}: x${selectedPixelRect.x0}, y${selectedPixelRect.y0}, ${selectedPixelRect.width}x${selectedPixelRect.height}`
        : `T${selectedIndex + 1}`;
    const coverageLabel = coverage.gaps.length
        ? `${coverage.gaps.length} uncovered gap(s), ${(coverage.uncoveredRatio * 100).toFixed(2)}%`
        : "covered";
    const fillMessage = node.scailManualTileFillMessage ? ` / ${node.scailManualTileFillMessage}` : "";
    status.textContent = `${sourceLabel} / ${selectedLabel} / ${coverageLabel} / max ${MAX_TILES} tiles${fillMessage}`;

    if (previewItems.length > 1) {
        const frameControl = document.createElement("div");
        frameControl.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:8px;";
        const label = document.createElement("div");
        label.textContent = "Frame";
        label.style.cssText = "opacity:.78;min-width:40px;";
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "0";
        slider.max = String(previewItems.length - 1);
        slider.step = "1";
        slider.value = String(previewIndex);
        slider.style.cssText = "width:100%;";
        slider.oninput = () => {
            const nextIndex = Math.max(0, Math.min(previewItems.length - 1, Number(slider.value)));
            node.scailManualTilePreviewIndex = nextIndex;
            const nextItem = previewItems[nextIndex];
            if (previewImageElement && nextItem) {
                previewImageElement.src = matrixImageUrl(nextItem);
            }
            note.textContent = previewLabel(nextItem, nextIndex);
        };
        frameControl.append(label, slider);
        container.append(header, stage, controls, status, frameControl);
    } else {
        container.append(header, stage, controls, status);
    }
    resizeNode(node);
}

app.registerExtension({
    name: "scail_multi_cond.dynamic_inputs",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (String(nodeData.name ?? "").startsWith("SCAIL2")) {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                this.scailNodeName = nodeData.name;
                requestAnimationFrame(() => installScailWidgetTooltips(this));
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                this.scailNodeName = nodeData.name;
                requestAnimationFrame(() => installScailWidgetTooltips(this));
            };
        }

        if (nodeData.name === "SCAIL2SegmentPlanBuilder") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                addUpdateButton(this, "Update segment inputs", updateSegmentBuilder);
                updateSegmentBuilder(this);
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => updateSegmentBuilder(this));
            };
        }

        if (["SCAIL2ScheduledLongVideo", "SCAIL2TiledLongVideo"].includes(nodeData.name)) {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                addUpdateButton(this, "Update reference inputs", updateScheduledGenerator);
                updateScheduledGenerator(this);
                ensureLongVideoStatusWidget(this);
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => {
                    updateScheduledGenerator(this);
                    ensureLongVideoStatusWidget(this);
                });
            };
        }

        if (["SCAIL2ScheduledLongVideoWithSAM", "SCAIL2TiledLongVideoWithSAM"].includes(nodeData.name)) {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                addUpdateButton(this, "Update reference inputs", updateScheduledGeneratorWithSAM);
                updateScheduledGeneratorWithSAM(this);
                ensureLongVideoStatusWidget(this);
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => {
                    updateScheduledGeneratorWithSAM(this);
                    ensureLongVideoStatusWidget(this);
                });
            };
        }

        if (nodeData.name === "SCAIL2MultiReferenceColoredMask") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                addUpdateButton(this, "Update reference track inputs", updateMultiReferenceMask);
                updateMultiReferenceMask(this);
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => updateMultiReferenceMask(this));
            };
        }

        if (nodeData.name === "SCAIL2ManualTilePlanBuilder") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                ensureManualTileEditor(this);
                const layout = parseManualTileLayout(this);
                writeManualTileLayout(this, layout.tiles, this.scailManualTileSelectedIndex ?? 0);
                renderManualTileEditor(this);
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => {
                    ensureManualTileEditor(this);
                    renderManualTileEditor(this);
                });
            };

            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                originalOnExecuted?.apply(this, arguments);
                const preview = extractManualTilePreview(message);
                if (preview) {
                    this.scailManualTilePreview = preview;
                    this.scailManualTilePreviewIndex = Math.min(
                        Number(this.scailManualTilePreviewIndex ?? 0),
                        Math.max(0, preview.items.length - 1)
                    );
                    ensureManualTileEditor(this);
                    renderManualTileEditor(this);
                }
            };
        }

        if (nodeData.name === "SCAIL2KeyframeMatrixViewer") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                ensureMatrixWidget(this);
                resizeNode(this);
            };

            const originalOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                originalOnExecuted?.apply(this, arguments);
                const candidates = [
                    message?.scail_keyframe_matrix,
                    message?.scail_keyframe_matrix_json,
                    message?.scail_keyframe_matrix_list,
                    message?.ui?.scail_keyframe_matrix,
                    message?.ui?.scail_keyframe_matrix_json,
                    message?.ui?.scail_keyframe_matrix_list,
                    message?.output?.scail_keyframe_matrix,
                    message?.output?.scail_keyframe_matrix_json,
                    message?.output?.scail_keyframe_matrix_list,
                    message?.images,
                    message?.ui?.images,
                    message?.output?.images,
                ];
                const matrix = candidates.find((candidate) => normalizeMatrixPayload(candidate).items.length);
                if (matrix !== undefined && matrix !== null) {
                    renderMatrix(this, normalizeMatrixPayload(matrix));
                }
            };
        }
    },
});

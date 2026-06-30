import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[SCAIL Multi Cond] dynamic UI extension loaded");

const MAX_SEGMENTS = 8;
const MAX_REFERENCES = 8;
const MAX_TILES = 8;

function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
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
    return Math.max(1, Math.min(256, Math.round(Number(widget?.value ?? 32))));
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
            Math.max(320, Number(node.scailManualTileEditorHeight ?? 320)),
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

    const aspect = manualTileAspect(node);
    const preview = node.scailManualTilePreview ?? { items: [] };
    const previewItems = Array.isArray(preview.items) ? preview.items : [];
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
    const stageHeight = Math.max(180, Math.min(420, Math.round(260 * aspect)));
    const editorHeight = stageHeight + (previewItems.length > 1 ? 150 : 118);
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
    stage.style.cssText = [
        "position:relative",
        "height:" + stageHeight + "px",
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

    const beginTileDrag = (event, tileIndex, mode, regionElement) => {
        event.preventDefault();
        event.stopPropagation();
        node.scailManualTileSelectedIndex = tileIndex;
        if (regionElement) {
            regionElement.style.zIndex = "40";
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
            if (mode === "resize") {
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
            setRegionTileStyle(regionElement, nextTiles[tileIndex]);
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
            "z-index:" + (selected ? "30" : String(10 + index)),
            "cursor:move",
            "user-select:none",
        ].join(";");
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
    controls.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:8px;flex-wrap:wrap;";
    const note = document.createElement("div");
    note.textContent = previewItem
        ? `Preview ${previewIndex + 1}/${previewItems.length}: ${previewItem.label ?? `frame ${previewItem.frame_1_based ?? ""}`}`
        : "Drag rectangles. The Python node adds overlap and final aligned repaint sizes.";
    note.style.cssText = "opacity:.72;min-width:180px;";
    const actions = document.createElement("div");
    actions.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap;";
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
    actions.append(add, remove, fillGaps, reset);
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
            node.scailManualTilePreviewIndex = Number(slider.value);
            renderManualTileEditor(node);
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
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => updateScheduledGenerator(this));
            };
        }

        if (["SCAIL2ScheduledLongVideoWithSAM", "SCAIL2TiledLongVideoWithSAM"].includes(nodeData.name)) {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);
                addUpdateButton(this, "Update reference inputs", updateScheduledGeneratorWithSAM);
                updateScheduledGeneratorWithSAM(this);
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                originalOnConfigure?.apply(this, arguments);
                requestAnimationFrame(() => updateScheduledGeneratorWithSAM(this));
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

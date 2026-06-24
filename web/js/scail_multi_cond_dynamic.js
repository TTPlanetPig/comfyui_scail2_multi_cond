import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("[SCAIL Multi Cond] dynamic UI extension loaded");

const MAX_SEGMENTS = 8;
const MAX_REFERENCES = 8;

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

        if (nodeData.name === "SCAIL2ScheduledLongVideo") {
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

        if (nodeData.name === "SCAIL2ScheduledLongVideoWithSAM") {
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

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const MATRIX_UI_VERSION = "v2.1";

function imageUrl(item) {
    const image = item?.image ?? item;
    if (typeof image === "string") {
        if (/^(https?:)?\/\//.test(image) || image.startsWith("/")) {
            return image;
        }
        const params = new URLSearchParams({ filename: image, type: "temp", subfolder: "" });
        return api.apiURL(`/view?${params.toString()}`);
    }
    if (image?.url) {
        return image.url;
    }
    const params = new URLSearchParams({
        filename: image?.filename ?? "",
        type: image?.type ?? "temp",
        subfolder: image?.subfolder ?? "",
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function normalizeItem(item, index) {
    const image = item?.image ?? item;
    const filename = image?.filename ?? (typeof image === "string" ? image : "");
    const frameMatch = /frame(\d+)/i.exec(filename);
    const batchIndex = item?.batch_index ?? item?.index ?? index;
    const rawChunkIndex = item?.chunk_index;
    const chunkNumber = item?.chunk_number_1_based ?? (Number.isFinite(Number(rawChunkIndex)) ? Number(rawChunkIndex) + 1 : "-");
    const kind = item?.kind ?? "image";
    const frame = item?.frame_1_based ?? (frameMatch ? Number(frameMatch[1]) : null);
    const fallbackLabel = `${String(batchIndex).padStart(3, "0")} | chunk ${chunkNumber} | ${kind}${
        frame ? ` | frame ${frame}` : ""
    }`;
    return {
        ...(typeof item === "object" && item !== null ? item : {}),
        filename,
        subfolder: image?.subfolder ?? item?.subfolder ?? "",
        type: image?.type ?? item?.type ?? "temp",
        batch_index: batchIndex,
        chunk_index: rawChunkIndex ?? "-",
        chunk_number_1_based: chunkNumber,
        kind,
        frame_1_based: frame,
        output_range_1_based_inclusive: item?.output_range_1_based_inclusive,
        label: item?.label ?? fallbackLabel,
    };
}

function normalizePayload(payload) {
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
        return { ...value, items: value.items.map(normalizeItem) };
    }
    if (Array.isArray(value)) {
        return { items: value.map(normalizeItem) };
    }
    return { ...(value ?? {}), items: [] };
}

function ensureContainer(node) {
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
    container.textContent = `Run this node to build the keyframe matrix (${MATRIX_UI_VERSION}).`;
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
        node.addWidget("text", "keyframe_matrix", `Matrix UI ${MATRIX_UI_VERSION}`, () => {}, {
            serialize: false,
        });
    }
    node.scailMatrixContainer = container;
    return container;
}

function resize(node) {
    if (node.computeSize) {
        const size = node.computeSize();
        node.size = [Math.max(node.size?.[0] ?? 420, size[0]), size[1]];
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function render(node, payload) {
    const matrix = normalizePayload(payload);
    const items = Array.isArray(matrix.items) ? matrix.items : [];
    const container = ensureContainer(node);
    container.replaceChildren();

    const header = document.createElement("div");
    header.style.cssText = "display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:8px;";
    const title = document.createElement("div");
    title.textContent = `Keyframe Matrix ${MATRIX_UI_VERSION} (${items.length})`;
    title.style.cssText = "font-weight:700;font-size:13px;";
    const hint = document.createElement("div");
    hint.textContent = "Open/download use original PNG files.";
    hint.style.cssText = "opacity:.72;text-align:right;";
    header.append(title, hint);
    container.append(header);

    if (!items.length) {
        const empty = document.createElement("div");
        empty.textContent = "No keyframes returned.";
        empty.style.opacity = ".72";
        container.append(empty);
        resize(node);
        return;
    }

    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(156px,1fr));gap:8px;";
    for (const item of items) {
        const url = imageUrl(item);
        const card = document.createElement("div");
        card.style.cssText = "background:#f8fafc;color:#111827;border:1px solid #cbd5e1;border-radius:6px;overflow:hidden;";

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
        actions.style.cssText = "display:flex;gap:6px;margin-top:7px;flex-wrap:wrap;";
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
    resize(node);
}

function extractPayload(message) {
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
    for (const candidate of candidates) {
        const normalized = normalizePayload(candidate);
        if (normalized.items.length) {
            return candidate;
        }
    }
    return candidates.find((candidate) => candidate !== undefined && candidate !== null);
}

app.registerExtension({
    name: "scail_multi_cond.keyframe_matrix_v2",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SCAIL2KeyframeMatrixViewer") {
            return;
        }
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            ensureContainer(this);
            resize(this);
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalOnExecuted?.apply(this, arguments);
            const payload = extractPayload(message);
            if (payload !== undefined && payload !== null) {
                render(this, payload);
            }
        };
    },
});

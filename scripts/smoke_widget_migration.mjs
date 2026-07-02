import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync("web/js/scail_multi_cond_dynamic.js", "utf8")
    .replace(/^import .*;\n/gm, "");

const context = {
    console,
    app: {
        registerExtension() {},
        canvas: null,
        graph: null,
    },
    api: {
        addEventListener() {},
        apiURL(path) {
            return path;
        },
    },
    requestAnimationFrame(callback) {
        return callback();
    },
};
context.globalThis = context;
context.window = context;

vm.runInNewContext(
    [
        "var app = globalThis.app;",
        "var api = globalThis.api;",
        source,
        "globalThis.__scailTest = { migrateScailWidgetValues, SCAIL_WIDGET_ORDER_HISTORY };",
    ].join("\n"),
    context,
    { filename: "scail_multi_cond_dynamic.js" }
);

const { migrateScailWidgetValues, SCAIL_WIDGET_ORDER_HISTORY } = context.__scailTest;

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function currentOrder(nodeType) {
    const orders = SCAIL_WIDGET_ORDER_HISTORY.get(nodeType);
    return orders[orders.length - 1];
}

function orderByLength(nodeType, length) {
    const order = SCAIL_WIDGET_ORDER_HISTORY.get(nodeType).find((item) => item.length === length);
    assert(order, `missing ${nodeType} order length ${length}`);
    return order;
}

function makeNode(nodeType, initialValues = null) {
    const names = currentOrder(nodeType);
    const widgets = names.map((name, index) => ({
        name,
        type: "number",
        value: initialValues?.[index] ?? defaultValue(name),
    }));
    return {
        widgets,
        widgets_values: widgets.map((widget) => widget.value),
        setDirtyCanvas() {},
    };
}

function defaultValue(name) {
    const defaults = {
        segment_plan: "",
        seed: 1,
        cfg: 1,
        mode: "replacement",
        max_frames: 0,
        max_chunk_frames: 81,
        overlap_frames: 5,
        reference_count: 2,
        color_correction: true,
        cache_mode: "disk",
        max_tile_pixels: 921600,
        enforce_tile_pixel_limit: true,
        expected_size_mismatch_mode: "warn",
        aspect_mismatch_mode: "warn",
        aspect_tolerance: 0.03,
        image_resize_mode: "bilinear",
        mask_resize_mode: "nearest",
        composite_feather_px: 48,
        tile_fit_mode: "stretch",
        frame_mismatch_mode: "trim_to_shortest",
        composite_color_correction: false,
        tile_seed_mode: "offset_by_tile",
        composite_blend_mode: "core_feather",
        seam_alignment: false,
        seam_alignment_apply_mode: "shifted_canvas_crop",
        seam_alignment_device: "auto",
        max_seam_shift_px: 4,
        seam_alignment_frames: 9,
        free_tail_window: 0,
        junction_mode: "weighted_average",
        lookahead_reference: false,
        lookahead_lead_frames: 8,
        lookahead_search_window_frames: 24,
        lookahead_analysis_stride: 2,
        lookahead_min_visible_ratio: 0.01,
        lookahead_min_new_area_ratio: 0.015,
        lookahead_max_anchors_per_tile: 2,
        lookahead_reference_pick_mode: "tile_visible_max",
        lookahead_context_expand_ratio: 0.35,
    };
    return Object.hasOwn(defaults, name) ? defaults[name] : "";
}

function valuesForOrder(order, overrides = {}) {
    return order.map((name) => Object.hasOwn(overrides, name) ? overrides[name] : defaultValue(name));
}

function widget(node, name) {
    const found = node.widgets.find((item) => item.name === name);
    assert(found, `missing widget ${name}`);
    return found;
}

{
    const legacyOrder = orderByLength("SCAIL2TiledLongVideo", 28);
    const legacyValues = valuesForOrder(legacyOrder, {
        cache_mode: "off",
        seam_alignment_apply_mode: "fixed_crop",
        max_seam_shift_px: 12,
        seam_alignment_frames: 13,
        free_tail_window: 8,
    });
    const node = makeNode("SCAIL2TiledLongVideo");
    migrateScailWidgetValues(node, "SCAIL2TiledLongVideo", { widgets_values: legacyValues });
    assert(widget(node, "cache_mode").value === "off", "legacy cache_mode should be preserved");
    assert(widget(node, "seam_alignment_apply_mode").value === "fixed_crop", "legacy apply mode should be preserved");
    assert(widget(node, "seam_alignment_device").value === "auto", "missing legacy device should default to auto");
    assert(widget(node, "max_seam_shift_px").value === 12, "legacy max shift should map after device insertion");
    assert(widget(node, "seam_alignment_frames").value === 13, "legacy seam frame count should map after device insertion");
    assert(widget(node, "free_tail_window").value === 8, "legacy free tail should map after device insertion");
    assert(widget(node, "junction_mode").value === "weighted_average", "new junction mode should keep default");
}

{
    const legacyOrder = orderByLength("SCAIL2TiledLongVideo", 27);
    const legacyValues = valuesForOrder(legacyOrder, {
        max_seam_shift_px: 16,
        seam_alignment_frames: 17,
        free_tail_window: 12,
    });
    const node = makeNode("SCAIL2TiledLongVideo");
    migrateScailWidgetValues(node, "SCAIL2TiledLongVideo", { widgets_values: legacyValues });
    assert(widget(node, "seam_alignment_apply_mode").value === "shifted_canvas_crop", "missing legacy apply mode should default");
    assert(widget(node, "seam_alignment_device").value === "auto", "missing legacy device should default");
    assert(widget(node, "max_seam_shift_px").value === 16, "no-apply legacy max shift should map");
    assert(widget(node, "seam_alignment_frames").value === 17, "no-apply legacy seam frames should map");
    assert(widget(node, "free_tail_window").value === 12, "no-apply legacy free tail should map");
}

{
    const badCurrentValues = valuesForOrder(currentOrder("SCAIL2TiledLongVideo"), {
        seam_alignment_apply_mode: "fixed_crop",
        seam_alignment_device: 32,
        max_seam_shift_px: 11,
        seam_alignment_frames: 8,
        free_tail_window: "",
    });
    const node = makeNode("SCAIL2TiledLongVideo", badCurrentValues);
    migrateScailWidgetValues(node, "SCAIL2TiledLongVideo", { widgets_values: badCurrentValues });
    assert(widget(node, "seam_alignment_apply_mode").value === "fixed_crop", "valid shifted apply mode should remain");
    assert(widget(node, "seam_alignment_device").value === "auto", "bad current device should be repaired");
    assert(widget(node, "max_seam_shift_px").value === 32, "bad current device value should move to max shift");
    assert(widget(node, "seam_alignment_frames").value === 11, "bad current max shift value should move to seam frames");
    assert(widget(node, "free_tail_window").value === 8, "bad current seam frames value should move to free tail");
}

{
    const legacyOrder = orderByLength("SCAIL2TiledLongVideo", 37);
    const legacyValues = valuesForOrder(legacyOrder, {
        lookahead_reference: true,
        lookahead_lead_frames: 12,
        lookahead_max_anchors_per_tile: 3,
    });
    const node = makeNode("SCAIL2TiledLongVideo");
    migrateScailWidgetValues(node, "SCAIL2TiledLongVideo", { widgets_values: legacyValues });
    assert(widget(node, "lookahead_reference").value === true, "legacy lookahead enable should be preserved");
    assert(widget(node, "lookahead_lead_frames").value === 12, "legacy lookahead lead should be preserved");
    assert(widget(node, "lookahead_max_anchors_per_tile").value === 3, "legacy lookahead anchors should be preserved");
    assert(widget(node, "lookahead_reference_pick_mode").value === "tile_visible_max", "new lookahead pick mode should default");
    assert(widget(node, "lookahead_context_expand_ratio").value === 0.35, "new lookahead expansion should default");
}

{
    const legacyOrder = orderByLength("SCAIL2ScheduledLongVideo", 10);
    const legacyValues = valuesForOrder(legacyOrder, {
        cache_mode: "off",
    });
    const node = makeNode("SCAIL2ScheduledLongVideo");
    migrateScailWidgetValues(node, "SCAIL2ScheduledLongVideo", { widgets_values: legacyValues });
    assert(widget(node, "cache_mode").value === "off", "scheduled cache_mode should survive free-tail append");
    assert(widget(node, "free_tail_window").value === 0, "scheduled missing free tail should default to 0");
}

console.log("smoke_widget_migration: ok");

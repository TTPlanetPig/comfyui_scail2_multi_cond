import fs from "node:fs";

const source = fs.readFileSync("web/js/scail_multi_cond_dynamic.js", "utf8");

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function sliceBetween(startText, endText, from = 0) {
    const start = source.indexOf(startText, from);
    assert(start >= 0, `missing start marker: ${startText}`);
    const end = source.indexOf(endText, start);
    assert(end > start, `missing end marker: ${endText}`);
    return source.slice(start, end);
}

function sliceWithin(haystack, startText, endText, from = 0) {
    const start = haystack.indexOf(startText, from);
    assert(start >= 0, `missing start marker: ${startText}`);
    const end = haystack.indexOf(endText, start);
    assert(end > start, `missing end marker: ${endText}`);
    return haystack.slice(start, end);
}

const manualEditorSource = sliceBetween("function ensureManualTileEditor", "app.registerExtension");
const beginDragIndex = manualEditorSource.indexOf("const beginTileDrag");
assert(beginDragIndex >= 0, "missing manual tile drag handler");
const dragSource = sliceWithin(
    manualEditorSource,
    "const apply = (moveEvent) =>",
    "const end = (endEvent) =>",
    beginDragIndex
);

assert(
    !/renderManualTileEditor\(node\)/.test(dragSource),
    "manual tile drag must not rebuild the editor during pointermove"
);
assert(
    !/container\.scrollHeight/.test(manualEditorSource),
    "manual tile editor height must not depend on container.scrollHeight"
);
assert(/setPointerCapture/.test(manualEditorSource), "manual tile drag should capture pointer events");
assert(/releasePointerCapture/.test(manualEditorSource), "manual tile drag should release pointer capture");
assert(
    /window\.addEventListener\("pointermove", apply, true\)/.test(manualEditorSource),
    "manual tile drag should listen during capture phase"
);

console.log("smoke_manual_tile_editor: ok");

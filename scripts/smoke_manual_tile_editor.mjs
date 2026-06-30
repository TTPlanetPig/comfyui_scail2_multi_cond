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
const parseLayoutSource = sliceBetween("function parseManualTileLayout", "function normalizeManualTilePreview");
const writeLayoutSource = sliceBetween("function writeManualTileLayout", "function setManualTileTiles");
const beginDragIndex = manualEditorSource.indexOf("const beginTileDrag");
assert(beginDragIndex >= 0, "missing manual tile drag handler");
const dragSource = sliceWithin(
    manualEditorSource,
    "const apply = (moveEvent) =>",
    "const end = (endEvent) =>",
    beginDragIndex
);
const sliderSource = sliceWithin(
    manualEditorSource,
    "slider.oninput = () =>",
    "frameControl.append",
    beginDragIndex
);

assert(
    !/renderManualTileEditor\(node\)/.test(dragSource),
    "manual tile drag must not rebuild the editor during pointermove"
);
assert(
    !/renderManualTileEditor\(node\)/.test(sliderSource),
    "manual tile frame slider must not rebuild the editor during drag"
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
assert(/function analyzeManualTileCoverage/.test(source), "manual tile editor should analyze uncovered areas");
assert(/Uncovered area/.test(manualEditorSource), "manual tile editor should show uncovered area overlays");
assert(/Fill gaps/.test(manualEditorSource), "manual tile editor should expose Fill gaps action");
assert(/snapManualTileMove/.test(source), "manual tile editor should snap moved tiles");
assert(/snapManualTileResize/.test(source), "manual tile editor should snap resized tiles");
assert(/const tileRenderOrder = tiles/.test(manualEditorSource), "manual tile editor should compute a render order");
assert(/tileRenderOrder\.push\(selectedIndex\)/.test(manualEditorSource), "selected manual tile should render last");
assert(/const resolvePointerTileIndex/.test(manualEditorSource), "manual tile editor should hit-test overlapping tiles");
assert(/hits\.includes\(selectedIndex\)/.test(manualEditorSource), "selected overlapping tile should receive pointer interaction first");
assert(/const tileSelector = document\.createElement\("div"\)/.test(manualEditorSource), "manual tile editor should expose direct tile selectors");
assert(/z-index:" \+ \(selected \? "100"/.test(manualEditorSource), "selected manual tile should have the highest z-index");
assert(/activeRegionElement\.style\.zIndex = "120"/.test(manualEditorSource), "dragged manual tile should stay above other tiles");
assert(/function manualTileAspectRatioCss/.test(source), "manual tile editor should preserve source aspect ratio");
assert(/function manualTileStageHeight/.test(source), "manual tile editor should compute stage height from node width");
assert(/manualTileEditorHeight\(node\)/.test(manualEditorSource), "manual tile DOM widget should compute dynamic height");
assert(/"aspect-ratio:" \+ manualTileAspectRatioCss\(node\)/.test(manualEditorSource), "manual tile stage should use CSS aspect-ratio");
assert(!/"height:" \+ stageHeight/.test(manualEditorSource), "manual tile stage must not use fixed height that distorts on resize");
assert(/previewImageElement\.src = matrixImageUrl\(nextItem\)/.test(sliderSource), "manual tile frame slider should update the preview image directly");
assert(/MANUAL_TILE_LAYOUT_STORAGE_PREFIX/.test(source), "manual tile layouts should have a storage namespace");
assert(/function readStoredManualTileLayout/.test(source), "manual tile editor should read persisted layouts");
assert(/function storeManualTileLayout/.test(source), "manual tile editor should store persisted layouts");
assert(/readStoredManualTileLayout\(node\)/.test(parseLayoutSource), "manual tile parsing should restore persisted layouts before defaults");
assert(
    parseLayoutSource.indexOf("readStoredManualTileLayout(node)") < parseLayoutSource.indexOf("manualTilesFromSplit"),
    "manual tile parsing should try persisted layouts before rebuilding a default split"
);
assert(/storeManualTileLayout\(node, layout\)/.test(writeLayoutSource), "manual tile writes should persist the latest layout");

console.log("smoke_manual_tile_editor: ok");

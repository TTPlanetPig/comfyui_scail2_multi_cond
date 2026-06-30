import fs from "node:fs";

const nodesSource = fs.readFileSync("nodes.py", "utf8");
const uiSource = fs.readFileSync("web/js/scail_multi_cond_dynamic.js", "utf8");

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function sliceBetween(source, startText, endText, from = 0) {
    const start = source.indexOf(startText, from);
    assert(start >= 0, `missing start marker: ${startText}`);
    const end = source.indexOf(endText, start);
    assert(end > start, `missing end marker: ${endText}`);
    return source.slice(start, end);
}

const scheduledSource = sliceBetween(nodesSource, "class SCAIL2ScheduledLongVideo:", "class SCAIL2ScheduledLongVideoWithSAM");
const scheduledSamSource = sliceBetween(nodesSource, "class SCAIL2ScheduledLongVideoWithSAM", "def _try_track_data_to_mask");
const tiledRunnerSource = sliceBetween(nodesSource, "def _run_tiled_long_video", "class SCAIL2TiledLongVideo");
const uiStatusSource = sliceBetween(uiSource, "function findGraphNodeById", "function updateSegmentBuilder");
const registerSource = sliceBetween(uiSource, "app.registerExtension", "});");

assert(/LONG_VIDEO_STATUS_EVENT = "scail2_long_video_status"/.test(nodesSource), "backend should define long-video status event");
assert(/def _send_long_video_status/.test(nodesSource), "backend should expose status sender helper");
assert(/PromptServer\.instance\.send_sync\(LONG_VIDEO_STATUS_EVENT, payload\)/.test(nodesSource), "backend should send ComfyUI websocket events");
assert(/send_status\(\s*"planning"/.test(scheduledSource), "scheduled node should report planning stage");
assert(/send_status\(\s*"sampling"/.test(scheduledSource), "scheduled node should report sampling stage");
assert(/send_status\(\s*"decoding"/.test(scheduledSource), "scheduled node should report decoding stage");
assert(/send_status\(\s*"done"/.test(scheduledSource), "scheduled node should report completion");
assert(/send_status\(\s*"running_sam_pose"/.test(scheduledSamSource), "internal SAM node should report pose SAM stage");
assert(/send_status\(\s*"running_sam_reference"/.test(scheduledSamSource), "internal SAM node should report reference SAM stage");
assert(/status_unique_id=status_id/.test(scheduledSamSource), "internal SAM node should forward status target into base scheduler");
assert(/send_status\(\s*"cropping_tile_inputs"/.test(tiledRunnerSource), "tiled node should report tile input cropping");
assert(/send_status\(\s*"compositing_tiles"/.test(tiledRunnerSource), "tiled node should report tile compositing");
assert(/status_unique_id=unique_id/.test(tiledRunnerSource), "tiled child scheduler should report status to parent node");
assert(/status_prefix=f"Tile/.test(tiledRunnerSource), "tiled child status should include tile prefix");

assert(/const LONG_VIDEO_STATUS_EVENT = "scail2_long_video_status"/.test(uiSource), "frontend should define matching status event");
assert(/const SCAIL_TOOLTIP_DELAY_MS = 3000/.test(uiSource), "frontend widget tooltips should use the requested three-second hover delay");
assert(/function installScailWidgetTooltips/.test(uiSource), "frontend should install widget tooltip metadata");
assert(/function handleScailTooltipMove/.test(uiSource), "frontend should handle canvas widget hover tooltips");
assert(/startsWith\("SCAIL2"\)/.test(registerSource), "all SCAIL2 nodes should install widget tooltips");
assert(/function ensureLongVideoStatusWidget/.test(uiStatusSource), "frontend should create a status widget");
assert(/function renderLongVideoStatus/.test(uiStatusSource), "frontend should render status text");
assert(/api\.addEventListener\?\.\(LONG_VIDEO_STATUS_EVENT, handleLongVideoStatus\)/.test(uiStatusSource), "frontend should listen for backend status events");
assert(/ensureLongVideoStatusWidget\(this\)/.test(registerSource), "long-video nodes should install status widget on creation/configure");

console.log("smoke_long_video_status: ok");

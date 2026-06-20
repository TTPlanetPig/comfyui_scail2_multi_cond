import { app } from "../../../scripts/app.js";

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
    },
});

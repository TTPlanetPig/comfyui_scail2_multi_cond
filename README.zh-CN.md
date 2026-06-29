# ComfyUI SCAIL2 Scheduled Long Video 中文说明

[English README](README.md)

这是一个给 ComfyUI SCAIL2 长视频流程使用的自定义节点包。核心目标是把长视频动作迁移拆成可控的分段流程，同时支持多参考图、多提示词、分段 overlap、外置 SAM 蒙板、内置 SAM 蒙板，以及第二次脸部细化。

## 主要能力

- 将长视频按计划分段生成，每段可以使用不同参考图和提示词。
- 保持 SCAIL2 原生推荐的短窗口生成逻辑，默认围绕 `81` 帧以内的 chunk 工作。
- 支持 `previous_frames` 续帧，减少长视频断裂。
- 支持 `boundary_overlap`，在换参考图时减少旧参考图惯性。
- 支持外置 SAM 蒙板，也支持内置 SAM 版本。
- 提供脸部二次优化流程：先生成全身视频，再裁出稳定头部区域，二次生成高质量脸部细节，最后贴回全身视频。
- 提供参考图脸部对齐节点，让高清脸部参考图的人脸位置和大小对齐 crop 视频第一帧，提升二次脸部生成稳定性。

## 安装

将本仓库放入 ComfyUI 的 `custom_nodes` 目录：

```text
ComfyUI/custom_nodes/comfyui_scail2_multi_cond
```

重启 ComfyUI。

如果动态按钮没有出现，强制刷新浏览器页面。前端扩展文件是：

```text
web/js/scail_multi_cond_dynamic.js
```

浏览器控制台应能看到：

```text
[SCAIL Multi Cond] dynamic UI extension loaded
```

## 开发 Smoke Test

修改 tile 或 tiled long-video 节点后，可以先跑：

```bash
python3 -B scripts/smoke_tiled_nodes.py
node scripts/smoke_manual_tile_editor.mjs
```

这些测试不跑真实模型推理，只验证节点注册、7 块 tile manifest、32 像素对齐、像素预算拒绝、外置 mask / 内置 SAM 输入差异、“全局 SAM 一次后按 tile 裁切”的策略，以及手动 tile 编辑器的拖拽、高度保护、漏区补块和边缘吸附没有被改坏。

## 基础长视频节点

### SCAIL-2 Segment Plan Builder

用于生成分段计划，不需要手写 JSON。

每个 segment 包含：

- `frames`：该段最终保留的帧数；
- `reference`：该段使用第几个 `reference_N`；
- `prompt`：该段正向提示词；
- `negative`：该段负向提示词；
- `boundary_overlap`：换参考图时的 overlap 覆盖值。

设置 `segment_count` 后，点击 `Update segment inputs`，节点会隐藏未使用的 segment 输入。

### SCAIL-2 Scheduled Long Video

这是外置 SAM / 外置蒙板版本。你需要先在节点外生成：

```text
pose_video_mask
reference_N_mask
```

推荐流程：

```text
driving_track_data + reference_N_track_data
  -> SCAIL-2 Multi Reference Colored Mask
  -> pose_video_mask + reference_N_mask
  -> SCAIL-2 Scheduled Long Video
```

这个版本适合需要提前预览、检查、手动调整 SAM 追踪结果的工作流。

### SCAIL-2 Scheduled Long Video (Internal SAM)

这是内置 SAM 版本。它保留和外置版本一致的分段、chunk、overlap、续帧逻辑，但会在节点内部生成 driving video 和 reference image 的 SAM track / mask。

典型连接：

```text
pose_video + sam_model + sam_conditioning
reference_N + sam_model + sam_conditioning
  -> SCAIL-2 Scheduled Long Video (Internal SAM)
```

如果你想让工作流更简洁，可以用这个版本。如果你想看见并调试 SAM mask，建议使用外置版本。

### SCAIL-2 Multi Reference Colored Mask

用于一次性生成多参考图所需的 SCAIL2 colored mask。

输入：

- `driving_track_data`
- `reference_N_track_data`

输出：

- `pose_video_mask`
- 动态数量的 `reference_N_mask`

设置 `reference_count` 后，点击 `Update reference track inputs`，节点会隐藏未使用的输入和输出。

## 推荐长视频参数

基础推荐：

```text
max_chunk_frames = 81
overlap_frames = 5
```

Plan Builder 里的参考切换：

```text
boundary_overlap = 5
```

说明：

- `overlap_frames` 是普通续帧的全局 overlap。
- `boundary_overlap` 是换参考图时第一段 chunk 的特殊 overlap。
- `-1` 表示使用全局 `overlap_frames`。
- `0` 表示换参考图时不带上一段锚点。
- `1` 表示只带最小连续性，通常更利于新参考图快速接管。
- 当前示例 workflow 和 Plan Builder 默认值统一使用 `5`。

## 脸部二次优化流程

脸部优化不是替换原来的长视频节点，而是在全身视频生成完成之后增加第二阶段：

```text
SCAIL-2 Scheduled Long Video / Internal SAM
  -> 全身视频
  -> SCAIL-2 Head Track Crop
  -> SCAIL-2 Align Reference Face To Crop
  -> 第二次 SCAIL-2 Scheduled Long Video / Internal SAM
  -> SCAIL-2 Face Composite Back
  -> 输出最终视频
```

推荐做成两段式：

1. 第一段先生成全身视频。
2. 你确认动作、构图、衣服、整体结果满意之后，再进入脸部细化。
3. 第二段读取第一段生成的视频，裁出包含头部活动范围的稳定画框。
4. 使用高清脸部参考图进行二次生成。
5. 用原始 crop mask 和 manifest 把新脸部视频贴回全身视频。

这样做的好处是：全身结果不满意时，不会浪费时间跑第二次脸部细化。

## SCAIL-2 Head Track Crop

这个节点负责从全身视频里裁出脸部附近的正方形区域，并记录贴回去需要的位置信息。

输入可以使用：

- 外部 `head_masks`；
- 或 `sam_model + head_conditioning`，让节点内部通过 SAM3 获取脸部/头部 mask。

重要原则：

- SAM 或输入 mask 是源数据。
- 节点不会用身体 colored mask 兜底猜脸部。
- 如果 SAM 抓到了上半身，crop 也会暴露这个问题，而不是偷偷替你改成脸。
- `mask_component_mode = largest` 会只保留每帧最大的 mask 连通区域，避免零碎身体碎片把 crop 拉大。

关键参数：

- `crop_mode = center_follow`：从第一帧确定固定尺寸，后续跟随脸部中心移动。
- `crop_mode = fixed_canvas`：统计整段脸部运动范围，生成一个固定机位的最小正方形画框。二次脸部细化通常更推荐这个模式。
- `crop_padding_ratio`：脸部区域外扩比例，常用 `0.35` 到 `0.5`。
- `square_align`：正方形边长对齐倍数，建议 `32`，适配 SCAIL2 生成分辨率。
- `mask_expand_px` / `mask_blur_px`：用于 crop mask 的轻微扩张和柔化。

`face_crop_video` 是裁出的视频，`crop_masks` 是裁出区域里的原始脸部 mask，`crop_manifest` 记录每帧贴回全身视频的位置。

## SCAIL-2 Align Reference Face To Crop

感谢爱屋的提醒：二次脸部生成时，如果高清参考图里的人脸大小和位置没有对齐 crop 视频，生成结果更容易出现大小脸、脸部漂移、头部不稳定的问题。

因此新增了 `SCAIL-2 Align Reference Face To Crop` 节点。

这个节点会：

1. 读取 `face_crop_video` 的第一帧。
2. 检测第一帧里的人脸位置和宽度。
3. 检测高清参考图里的人脸位置和宽度。
4. 按 crop 视频的比例重新裁切参考图。
5. 让参考图里的人脸位置和大小尽量对齐 crop 第一帧。
6. 保持参考图原始像素清晰度；默认 `window_fit_mode=shift_inside_reference` 会在裁切窗口放得进参考图时先把窗口平移回图内，只有窗口本身比参考图还大时才按 `padding_mode` 补边。

如果你需要完全保留旧逻辑的严格相对位置，可以把 `window_fit_mode` 改成 `strict_alignment`。这个模式下只要严格计算出的窗口越界，就会按 `padding_mode` 补边。

检测后端：

- `face_detector_backend = auto`：优先使用 InsightFace，失败或未安装时自动使用 MediaPipe。
- `insightface`：检测更强，适合已经安装 `insightface` 和 `onnxruntime-gpu` 的环境。
- `mediapipe`：安装最简单。

MediaPipe 安装：

```text
python -m pip install mediapipe
```

InsightFace 安装：

```text
python -m pip install insightface onnxruntime-gpu
```

如果不用 GPU，可以把 `onnxruntime-gpu` 换成 `onnxruntime`。

## SCAIL-2 Face Composite Back

这个节点负责把二次生成的新脸部 crop 视频贴回原始全身视频。

基本逻辑：

1. 读取 `crop_manifest` 中记录的原始 crop 位置。
2. 将 refined face video 自动匹配回 crop canvas 尺寸。
3. 使用 `crop_masks` 决定真正贴回的脸部区域。
4. 可选做颜色校正。
5. 用 feather 后的 mask 混合到全身视频。

关键参数：

- `color_correction`：是否开启颜色校正。
- `face_fit_mode = center_crop`：保持比例并居中裁切，通常推荐。
- `face_fit_mode = pad`：保持比例并补边。
- `face_fit_mode = stretch`：直接拉伸到 crop 尺寸，只建议调试。
- `frame_mismatch_mode = trim_to_shortest`：当全身视频、脸部视频、mask、manifest 最后几帧数量不一致时，自动按最短帧数截断。
- `feather_px`：贴回边缘柔化。
- `mask_contract_px`：向内收缩 mask，减少脖子、头发边缘或背景被贴进去。
- `stitch_mask_expand_px`：贴回前向外扩张 mask。
- `stitch_offset_x_px` / `stitch_offset_y_px`：最终贴回的像素级偏移修正。

如果贴回后脸偏右，可以先尝试：

```text
stitch_offset_x_px = -1 或 -2
```

如果贴回区域把脖子或周围身体带进去，可以优先尝试：

```text
mask_contract_px 增大
stitch_mask_expand_px 减小
```

## Tile 超分流程

`SCAIL-2 Manual Tile Plan Builder` 会生成 `tile_manifest`。如果你填写了
`output_width` / `output_height`，节点不会拉伸原始视频比例，而是按源视频比例
自动修正最终目标尺寸。例如源视频是 `548x960`，目标填 `1080x1920` 时，
manifest 会解析为 `1096x1920`，并在 `target_size_adjustment` 里记录请求值和实际值。

`SCAIL-2 Tiled Long Video` 是自动生产节点。连接第一阶段的 `pose_video`、
`tile_manifest`、原始 `segment_plan`、模型输入，以及和普通长视频节点相同的
`reference_N` / mask 即可。节点内部会按 tile 自动裁切 `pose_video`、对应的
`reference_N` 和 mask，再逐块调用长视频生成，最后自动拼合。
拼合时使用 core 优先的 feather：overlap 主要作为生成上下文，不会整段大面积参与最终平均，
只有 core 边缘会柔和过渡到相邻 tile。

如果你想单独调试每一块，再使用 `SCAIL-2 Tile Extractor` 手动导出每块视频，
分别生成后接 `SCAIL-2 Tile Repaint Collector` 和 `SCAIL-2 Tile Composite Video`。

## 长视频缓存

长视频节点提供 `cache_mode`：

- `disk`：默认开启单槽磁盘缓存；
- `off`：关闭磁盘缓存。

缓存逻辑是单槽覆盖：

- 如果输入语义一致，节点会直接读取上一次磁盘缓存，跳过内部采样。
- 如果输入语义变化，会重新推理并覆盖旧缓存。
- 每个节点实例只保留一组缓存，不需要手动清理多版本缓存。

缓存位置在 ComfyUI 输出目录下：

```text
output/scail2_cache/long_video/
```

注意：这个磁盘缓存能避免节点内部重复采样，但不能强制 ComfyUI 核心调度器完全不调用节点函数。如果 ComfyUI 的运行时 output cache 被释放，节点仍可能被调用；此时磁盘缓存会尽量快速返回结果。

## 示例工作流

内置基础示例：

```text
workflow/SCAIL2_scheduled_long_video_template.json
workflow/SCAIL2_long_video_sample.json
workflow/comfyui_scail2_multi_cond_sample_external.json
workflow/comfyui_scail2_multi_cond_sample_internal.json
```

脸部细化示例：

```text
examples/workflows/Wan21_SCAIL2_00_key_frame_capture.example.json
examples/workflows/Wan21_SCAIL2_01_full_body_pause.example.json
examples/workflows/Wan21_SCAIL2_02_face_detail_resume.example.json
examples/workflows/Wan21_SCAIL2_combined_full_body_to_face_detail.example.json
examples/workflows/Wan21_SCAIL2_two_stage_guide.md
```

建议使用顺序：

1. `Wan21_SCAIL2_00_key_frame_capture.example.json`：先抽关键帧，准备参考图。
2. `Wan21_SCAIL2_01_full_body_pause.example.json`：生成全身动作迁移结果。
3. 确认全身结果满意。
4. `Wan21_SCAIL2_02_face_detail_resume.example.json`：读取全身视频，做脸部二次优化。

`Wan21_SCAIL2_combined_full_body_to_face_detail.example.json` 是合并版参考流程，但实际重任务更推荐两段式，方便暂停检查。

## 常见问题

### 为什么 Head Track Crop 抓到了整个上半身？

这通常说明 SAM 或输入 mask 返回的就不是脸部 mask，而是上半身 mask。这个节点不会用身体 mask 猜脸，所以需要调整 `head_conditioning`，例如使用更明确的 `face`、`head`，或改用外部可预览的 `head_masks`。

### 为什么 mask 被身体碎片拉大？

使用：

```text
mask_component_mode = largest
```

它会只保留每帧最大的 mask 主体，丢弃细小碎片。

### 为什么二次生成后脸大小不稳定？

优先确认三件事：

1. `crop_mode` 是否使用了 `fixed_canvas`。
2. 二次生成分辨率是否严格按 crop 视频比例设置。
3. 高清参考图是否先经过 `SCAIL-2 Align Reference Face To Crop`。

### 为什么贴回边缘有痕迹？

优先调：

```text
feather_px
mask_contract_px
stitch_mask_expand_px
color_correction
```

如果颜色差异明显，开启 `color_correction`。如果边缘把脖子或背景带进去，增加 `mask_contract_px`。

## 隐私说明

仓库不包含模型文件、生成视频、私有输入图片、私人路径或上传素材。示例 workflow 使用的是占位资源名，使用时需要替换成你自己的 ComfyUI 输入文件。

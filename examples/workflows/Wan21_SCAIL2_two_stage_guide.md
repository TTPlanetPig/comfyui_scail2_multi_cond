# Wan21 SCAIL-2 两阶段范例工作流

这个范例故意把主流程拆成两个 workflow，让“保存出来的视频文件”成为中间暂停点。

## 可选阶段 0：自动截取关键帧

打开 `Wan21_SCAIL2_00_key_frame_capture.example.json`。这个 workflow 用同一套分段计划，从输入视频里自动截取边界/分段关键帧，并用矩阵查看节点集中预览。

用途是先挑出适合做参考图的帧，再把选好的参考图放进阶段 1 的全身参考图输入里。阶段 0 不参与最终生成，只是帮你更快准备参考帧。

## 阶段 1：全身生成

打开 `Wan21_SCAIL2_01_full_body_pause.example.json`。运行到 `VHS_VideoCombine` 输出全身结果，输出前缀已经整理为：

`SCAIL2_STAGE1_full_body_review`

先检查这个 mp4。动作、身体、衣服、构图都满意之后，再进入阶段 2。若不满意，只重跑阶段 1，不要浪费脸部优化算力。

## 阶段 2：脸部优化

打开 `Wan21_SCAIL2_02_face_detail_resume.example.json`。在左侧分组 `01 继续` 里，把 `VHS_LoadVideo.video` 改成阶段 1 通过审核的全身 mp4。

如果你的 ComfyUI 只在 input 文件夹列出视频，就先把阶段 1 输出的 mp4 复制到 `ComfyUI/input`，再在 `VHS_LoadVideo` 里选择它。

阶段 2 的链路是：

1. 载入已确认满意的全身视频。
2. 用 SAM 追踪脸/头，裁出固定脸部画框。
3. 把高清参考图按脸部画框第一帧/分段帧对齐。
4. 用 SCAIL-2 replacement 模式对脸部 crop 做二次生成。
5. 把二次生成的脸部 crop 按原始 mask 和 manifest 贴回全身视频，插帧后保存最终视频。

## Detector 后端

参考图对齐节点使用 `face_detector_backend = auto`：优先 InsightFace，失败时自动 fallback 到 MediaPipe。MediaPipe 安装更轻：

`python -m pip install mediapipe`

## 为什么这样暂停

不要把全身和脸部优化强行串成一个必须一次跑完的大图。全身结果不满意时，脸部优化阶段全部都是浪费。两个 workflow 分开以后，阶段 1 输出的视频就是明确的人工审核点。

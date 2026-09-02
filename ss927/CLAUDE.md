# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **PTZ single-target (主目标) person tracker** for Huawei Ascend NPU (昇腾 / ACL).
A PTZ camera follows one designated person; the tracker keeps lock through occlusion,
crowding, and camera motion, then reports the main target's box so a downstream gimbal
controller can re-center it. There is no main(), no build system, and no test suite in
this repo — it is a library that is cross-compiled into `libfxtracker.so` and linked
into a device-side application.

Most code comments and design notes are in Chinese; match that when editing.

## Build

There is no CMake/Makefile. The canonical build is the manual cross-compile (aarch64)
documented in `c_api/README.md`. Core library (compile all listed `.cpp` together):

```bash
g++ -std=c++17 -O2 -fPIC -shared \
    AclRuntime.cpp LightTracker.cpp KalmanBoxTracker.cpp KalmanFilter.cpp \
    PoseEstimator.cpp Detector.cpp \
    PersonReID.cpp FaceRecognition.cpp FaceRecognitionSystem.cpp \
    Facekps.cpp ModelProcess.cpp utils.cpp \
    c_api/fx_tracker.cpp \
    -Ic_api -I${OPENCV_INC} -I${ASCEND_ACL_INC} \
    -L${OPENCV_LIB} -lopencv_core -lopencv_imgproc -lopencv_video \
    -lopencv_calib3d -lopencv_features2d \
    -L${ASCEND_ACL_LIB} -lascendcl \
    -o libfxtracker.so

# Pure-C application against the C API:
gcc -std=c99 -O2 c_api/example_app.c -Ic_api -L. -lfxtracker -o demo
```

Notes:
- **`-std=c++17` is required** (core uses `std::optional`).
- Do **not** add `Track.cpp` / `fx_wrapper_track.cpp` to the C-API build unless the old
  app still needs the legacy interface. Both interfaces can coexist (no symbol clash).
- `IveGmc.cpp` is HiSilicon-IVE-only; it is compiled behind `#define USE_HISI_IVE` and
  must be omitted on Ascend builds (the only build macro in the codebase).
- Hard dependencies: **OpenCV 4** (C++-only API — `cv::Mat`, `calcOpticalFlowPyrLK`,
  `estimateAffinePartial2D` RANSAC) and **Ascend ACL** (`acl/acl.h`, `-lascendcl`).
  These are why there is no pure-C reimplementation.

## Runtime requirements

- Models are loaded from a **hard-coded directory `/oem/model`** on the device
  (`LightTracker::init`, via `Utils::loadModel` into memory, then `LoadModelFromMem`).
  Five `.om` files must be present there:
  `v8n_face_body_head_full_body.om` (detector),
  `mobilev2_EmbeddingHead_reid_v1_GeMP_pre.om` (ReID),
  `faceReco_v2.om` (face recognition), `faceKps_v7_01.om` (face keypoints),
  `rtmpose-t.om` (candidate-level pose).
- `AclRuntime` owned by `LightTracker` is the sole ACL global/device/context owner;
  model wrappers only own their model, dataset and buffers.

## Two C interfaces (prefer the new one)

- **New: `c_api/fx_tracker.h`** — the maintained pure-C99 API. Use this. It fixes real
  bugs in the old one (header-defined global, NULL `main_target` crash, `count` vs
  `total_count` mismatch reading uninitialized memory, missing destroy → handle leak)
  and adds `score`/coast-confidence, `fx_tracker_reset`, `fx_tracker_get_reset_flag`.
  In this API `main_target == NULL` means "normal tracking frame" (no re-designation).
- **Legacy: `fx_wrapper_track.h` / `fx_wrapper_track.cpp` / `Track.{h,cpp}`** — kept for
  the old app. `Track` is a thin shim over `LightTracker`. Note `fx_wrapper_track.h`
  defines a global `unsigned short DEBUG_LOG = 0;` in the header (the bug the new API
  avoids) — don't include it from two translation units.

Both wrappers bottom out in `LightTracker::update(img, mainBox)` → returns
`(cv::Mat track_results [N×≥5: x1,y1,x2,y2,id...], person_count)`.

## Architecture

The brain is **`LightTracker`** (`LightTracker.cpp` is ~170KB — the bulk of the logic).
Everything else is either a model-inference component it owns or a helper it calls.

Per-frame flow inside `LightTracker::update`:
1. Detect (`Detector_yolox`, YOLOX 416) → boxes with labels. **Label convention:
   `0=face, 1=body, 2=head`** — the *opposite* of the old model; always interpret via
   the `LABEL_FACE/LABEL_BODY/LABEL_HEAD` constants.
2. Associate faces/heads to bodies → `PersonWithFace` (`matchPersonFaces`).
3. Predict existing tracks with Kalman filters (`KalmanBoxTracker` / `KalmanFilter`),
   apply GMC camera-motion compensation (optical-flow + RANSAC affine on background;
   identity fallback on failure — variable-frame-interval safe).
4. Run identity signals on candidates: **ReID** (`PersonReID`/OSNet), candidate-level
   **pose** (`PoseEstimator`/RTMPose-T), and **face** (`FaceRecognitionSystem` +
   `Facekps` keypoints + `FaceRecognition` embedding). Pose is optional late evidence:
   ordinary CLEAR frames use 0 calls, refresh uses 1, and true Top1/Top2 ambiguity uses
   at most 2 calls through the frame-level hard budget.
5. Cascade-match the main target (`match_main_target_unified` / `assign_cascade`),
   fuse the signals, update secondary tracks (`update_secondary_tracks`).
6. If the main target is lost: head-continuity may sustain output from a *real* standalone
   head detection (reconstructed body box); otherwise **no main row is emitted at all** —
   the gimbal (a centering controller) must hold when nothing real is observed, because
   with GMC off any held/predicted box pins an off-center error and drives an endless slew.
   `coast_weight_` is now a PTZ control weight: 1.0 for a real body hit or a strictly
   accepted head/face observation, 0 when no main box is returned. Observation source is
   tracked separately so a full-speed part output cannot impersonate body identity.
   The former `MainTargetPredictor` path was removed after being unwired in 2026-07;
   timestamp-derived `last_real_obs_ms_` replaces its miss counter for the reacquisition
   probation gate.

Two **orthogonal state machines** drive matching strategy (see the long comments in
`LightTracker.h`):
- `OcclusionState {CLEAR, OCCLUDED, RECOVERING}` — "is someone blocking the target?"
  During occlusion the design **prefers no match over a wrong match** (KF pure-predict,
  camera holds, target stays centered after separation). Quality monitoring
  (`quality_history_`, baselines, `id_switch_alert_`) watches for ID switches.
- `VisibilityState {FULL, MOSTLY_FULL, HALF, UPPER, HEAD_ONLY}` — "how much of the
  target's own body is visible?" When only the upper body/head shows, ReID and pose
  degrade and are down-weighted in favor of head-KF / IoU / center / face signals.

Key supporting ideas: **face hard-anchoring** (`face_locked_`, TTL `kFaceLockTTL`) lets a
confirmed face override ReID/fusion in same-clothing crowds; **head continuity**
(`try_head_continuity`, learned head↔body geometry) sustains identity through occlusion;
a lead-center (`lead_cx_/lead_cy_`) moves the re-capture search gate toward where the
person is heading rather than where they vanished.

Diagnostic builds use `-DFX_TRACKER_MATCH_TRACE=1`. Match traces are written to
`/tmp/fx_tracker_match_trace.log` through a 64KB stdio buffer, flushed on decision events
or every 25 frames, and capped at 16MB. Set the runtime environment variable
`FX_TRACKER_MATCH_TRACE_FILE` to choose another path. The file is truncated on the first
trace write of each process; copy it before restarting the application.

The unused Gait and legacy Association components have been removed from the maintained
source boundary; they do not contribute to matching.

## Conventions

- **Boxes are `xyxy` (x1,y1,x2,y2)** throughout matrices/signals (not xywh) — the C-API
  output converts to x/y/w/h at the boundary only.
- Frame timing is **timestamp-driven** (`now_ms()`, `frame_dt_sec_`), not frame-count
  driven — speeds are px/sec so the system tolerates skipped frames and variable FPS.
  When adding motion/decay logic, keep it timestamp-based.
- Model components share a uniform shape: a `Config` struct carrying `bufferModel` +
  `sizeModel`, an `init(config)`, and `ModelProcess` (ACL `aclmdl*`) underneath
  (supports dynamic batch via `CreateInputDynamicBatch`/`SetDynamicBatchSize`).
- `docs/tracking_pipeline.svg` (and `.pdf`) diagram the pipeline.

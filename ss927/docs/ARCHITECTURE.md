# Architecture

## 权威入口与源码地图

维护中的调用边界是 `c_api/fx_tracker.cpp`，算法入口是 `LightTracker::update()`：

```text
C application
  -> fx_tracker_create/init/run/reset/destroy
  -> FxTrackerHandle::tracker (LightTracker)
  -> LightTracker::update(BGR cv::Mat, mainBox-as-xyxy)
  -> rows [x1,y1,x2,y2,id]
  -> FxTrackResult [x,y,w,h,id,score]
  -> downstream PTZ controller
```

当前源码定位（行号仅对应本快照，函数名更稳定）：

| 责任 | 权威位置 |
|---|---|
| 模型初始化 / 主目标指定 / 普通帧入口 | `LightTracker.cpp` — `init()`、`setMainTarget()`、`update()` |
| 检测部件归属 / 高低分组 | `LightTracker.cpp` — `matchPersonFaces()`、`extract_detections()` |
| 预测、候选、主匹配 | `LightTracker.cpp` — `assign_cascade()`、`match_main_target_unified()`、`collect_nearby_dets()` |
| 人脸/头部恢复与丢失 HOLD | `LightTracker.cpp` — `try_head_continuity()`、`try_face_only_continuity()`、`try_confirmed_face_track_continuity()`、`face_recognition_verification()` |
| KF 提交与其他轨迹 | `LightTracker.cpp` — `generate_final_results()`、`update_secondary_tracks()`、`update_trackers_unified()` |
| Pose、质量监控、重置 | `LightTracker.cpp` — `request_pose()`、`commit_pose()`、`update_quality_monitor()`、`reset()` |
| 数值滤波 | `KalmanBoxTracker.cpp` — `build_box_kf()`、`predict()`、`correct_body_from_part()`；`KalmanFilter.cpp` — `predict()`、`update()` |
| C ABI | `c_api/fx_tracker.cpp:24-143`；结构和错误码见 `c_api/fx_tracker.h` |

## 组件依赖与未接线路径

```text
LightTracker
├─ AclRuntime ───────────── process ACL init / instance device / current context
├─ Detector ───────────────┐
├─ CPersonReID             │
├─ PoseEstimator           ├─ ModelProcess ── Ascend ACL
├─ CFaceKeypoint106        │
├─ FaceRecognitionSystem ─ CFaceRecognition
├─ KalmanBoxTracker ── KalmanFilterNew
└─ Utils / OpenCV / optional IveGmc
```

主目标走 `match_main_target_unified()`，非主轨迹走 `update_secondary_tracks()` 内的贪心 IoU。未接入热路径的历史 `Association.*`、`MainTargetPredictor.*` 和 `GaitRecognition.*` 已从维护源码删除。

## ACL Runtime 与模型资源生命周期

`AclRuntime` 是本库唯一的 ACL 全局资源 owner。ACL global runtime 在进程首个 Tracker
init 时调用一次 `aclInit`，之后保持到进程退出；每个 Tracker 实例独立创建/销毁
device/context。`LightTracker` 将它声明在所有模型成员之前，因此 C++ 逆序析构保证所有
模型先释放、该实例 runtime 最后退出：

```text
AclRuntime::init
  -> 首次：aclInit；后续：复用 process runtime
  -> aclrtSetDevice -> aclrtGetRunMode
  -> aclrtCreateContext -> aclrtSetCurrentContext
  -> FaceKps -> FaceReco -> ReID -> Detector -> Pose init

destroy / init rollback
  -> Pose -> Detector -> ReID -> FaceReco -> FaceKps release
  -> aclrtDestroyContext -> aclrtResetDevice
```

当前模型执行全部使用同步 `aclmdlExecute()`；工程没有 async execute/memcpy，因而不再
创建无用途的显式 stream。Detector、ReID、FaceKps、FaceReco、Pose 都是 runtime
borrower：只能释放自己的 `ModelProcess`、dataset/data-buffer 包装和明确拥有的输入
device buffer。`ModelProcess::DestroyInput()` 不释放 wrapper 传入的主输入 buffer；
`DestroyOutput()` 则同时释放 output data buffer 及其 device buffer。

第一阶段 runtime 契约是库内独占、同线程、同时只允许一个已初始化的 tracker handle；
第二个 handle 的 init 会明确失败。完整 destroy 后可再次 create/init/run/destroy，且
不会再次调用 `aclInit`。当前库不在实例析构时调用 `aclFinalize`；宿主应在最终退出前
保持共享库已加载，避免 destroy/recreate 之间 `dlclose` 再重新加载同一库。宿主预先调用
ACL 和多 handle/context 共享尚不支持。

## 初始化与指定目标

`LightTracker::init()` 从 `/oem/model` 读入五个 OM 文件，分别初始化 Detector、ReID、FaceKps、FaceReco 和 Pose；任一模型文件或组件初始化失败均返回错误。

`update()` 以 `mainBox.area()>0` 选择指定分支。这里的 `cv::Rect` 实际装的是 `xyxy`，所以该检查不是 `x2>x1/y2>y1` 的合法性验证。指定帧由 `setMainTarget()`：

1. `reset()` 全量清状态并把指定框作为 BODY 真值。
2. 提取主 ReID，建立 anchor/current/confirmed embedding 和主人体 KF。
3. 对已确认的指定 BODY 运行一次候选级 RTMPose，建立关键点与人体比例基线。
4. 在指定框 ROI 内再运行 Detector，选择头和至多一张几何明确的人脸。
5. 建立头部 KF/几何先验；只有脸尺寸、关键点和质量通过时才注册模板，否则留给延迟注册。

## 普通帧真实调用链

`LightTracker::update()` 的普通分支顺序如下：

1. 计算真实帧间隔；间隔超过 3 秒则 `reset()`；推进 alert/overload、模型预算、ego 软参考和 trace。
2. `Detector_yolox::run(img, 0.2)` 一次得到 face/body/head。Detector 采用 416×416 左上对齐 resize、右/下 114 padding，解析 `[1,3549,8]` 后执行分类别 NMS；成功但结果为空表示正常零检测；
   preprocess/memcpy/execute/postprocess 失败会抛出 `TrackerRuntimeError`，C API 返回
   `FX_ERR_MODEL_RUN`，该帧不会进入零检测/丢失恢复分支。
3. `matchPersonFaces()` 保存全部原始 `recovery_faces_` / `recovery_heads_`，并为分数 `>0.30` 的每个人体最多选择一张上部居中脸和一个头。
4. `extract_detections()` 将 body 分成高分 `dets_one`（默认 `>0.7`）和低分 `dets_second`（`>0.3`）。两组都可进入同一主匹配和次级轨迹流程；`source/index` 只保留来源，不存在另一个 ByteTrack 式“高分先匹配、低分二次补配”流程。
5. **无 BODY**：`get_predicted_tracks()` 预测并老化 KF，清理过期轨迹，可选 GMC，然后依次尝试 FaceReco 身份恢复、已确认脸的检测级短桥接、头部连续性；均失败后，若最近安全真实 Motion History 可用，则从历史一次性估计并冻结 short prediction，否则不输出主行。
6. **有 BODY 但没有任何 tracker**：只用 `add_other_det()` 返回非主检测，不建立或猜测新的主目标，也不进入主目标融合。
7. **有 BODY 且有主轨迹**：`assign_cascade()` 依次预测 KF、可选 GMC、拆分主/非主、收集候选并调用 `match_main_target_unified()`。
8. `match_main_target_unified()` 内依次推进状态、决定局部/全图候选并调度 ReID。多人近分时，原始 ReID/anchor preliminary ranking 只可安排一个有限 Face 排除验证，绝不生成最终 winner、margin、ambiguity、HOLD、pending 或 measurement；Face 后从 final identity-valid 集合移除当前帧可靠确认非 A 的 BODY，再重新计算 base ranking、Top1/Top2 Pose、winner/runner-up、margin 与 identity gate。完整 BODY 集合独立保留给 close/overlap/OCC/coexist/crowded/measurement safety。之后继续执行原有硬门、人脸仲裁、provisional/commit gate 和 measurement 可靠性分类。无 GMC 时，same-cloth trajectory outlier 和 prediction-only secondary overlap 只标记 spatial conflict：先保留 identity tentative winner 并继续完成后续候选身份评估，最终冲突未由更强身份解除则 HOLD，不能由 motion 直接选另一个人。`RELIABLE` 才构造 match；`UNCERTAIN` 只保存 pending 假设并让主 KF pure-predict。最终 winner 已有本帧候选 Pose 时，函数仅保存其临时副本供 `update()` 的 PTZ output stage 使用；该副本不反向参与本帧选择或跨帧状态。
9. 回到 `update()` 后，若没有 `RELIABLE` BODY 主行，再按 FaceReco、确认脸桥接、头连续性兜底。FaceReco 是强身份路径；FACE_TRACK/HEAD 是纯空间 continuity，提交前必须通过当前帧三态身份 evidence 与 `PersonIdentityAmbiguityContext` permission。被禁止后不产生真实 observation 副作用，继续进入同一 Frozen Motion Prediction/NONE。若 BODY 已提交但其身份未闭环，确认脸桥接仍继续维护：关联脸能明确归属该 BODY 时保留真实 BODY 几何，只有归属不明确且 permission 允许时才可用重构框覆盖。
10. `add_other_det()` 附加未关联检测；`stabilize_returned_box()` 只稳定 PTZ 主框；帧尾更新 aim/ego/trace。

## Prediction、Kalman 与 Association

人体和头部 KF 都是 6 维常速度状态 `[cx,cy,w,h,vx,vy]`、4 维量测 `[cx,cy,w,h]`，但头部使用更灵活的过程噪声/速度衰减。`get_predicted_tracks()` 给人体与头部写入钳制后的真实 dt；完全无 BODY/HEAD/FACE 真观测达到长盲门后，`set_long_coast()` 清零速度外推但保留轨迹以供身份重捕。

丢失后的 PTZ blind slide 不属于 `LightTracker`：外部 `track-ptz` 固定最后真实 BODY 框，并按自身 PTZ 控制状态限距滑动。一个 loss episode 到达距离、方向或故障保护停止条件后进入 `PTZ_BLIND_STOPPED`，后续 LOST 帧不得重新启动 blind；只有新的真实 BODY 才重置为 `IDLE`。Tracker 无安全真实 BODY/HEAD/FACE 时返回 `NONE/HOLD`，不再调用 `try_short_prediction()` 输出 `PRED`。Motion History 只记录最终安全接受的真实人体语义框：`RELIABLE BODY > FACE_IDENTITY > FACE_TRACK > HEAD`，同帧最多一条；HEAD/FACE 记录 clip 后、输出稳定前的重构人体框。遮挡相对运动的 `relative_motion_history_` 仍独立限制为最多 5 条；历史 `try_short_prediction()` 实现暂未接线。

PTZ 在下一处理帧把只读 blind phase 传回 legacy `TrackInput`：只有实际速度积分产生 pan 位移后才从 `IDLE` 进入 `SLIDING`，距离上限到达后为 `STOPPED`。这两个状态建立图像坐标系失效 epoch；PTZ 回到 IDLE 本身不能恢复旧空间先验，必须由可靠 BODY 重捕结束。Tracker 保存最后可靠 BODY `xyxy` 作为独立 search anchor，用它而不是 KF/lead/coast endpoint 生成局部 BODY pool；每帧最多额外注入一个全图公平探索候选到既有 ReID budget。epoch 内 base BODY ranking 仅使用当前 ReID/anchor，Face 正证据仍可覆盖 BODY；Face recovery 在当前帧全体 face 上公平轮转，旧 face reference/pending 坐标不参与调度，当前 BODY–Face 几何只回答归属。旧 KF IoU、head/pose/trajectory、prediction-only spatial conflict 和位置迟滞不参与 acceptance。confirmed-other、可靠 Face 非 A 和严格 BODY identity negative 不变；普通 Face mismatch 仍为 UNKNOWN。首个可靠 BODY 用当前检测强制重建 KF，并清理旧 spatial recovery state；它不立即更新身份模板，下一可靠 BODY 后才回到正常 feature gate。

`coast_search_hint_` 是历史 PRED 的 search-only 状态。由于 PRED 输出已停用，它不会再被更新，并会在进入外部 PTZ blind 的 HOLD 路径清除；当前 recovery 不得使用其 image-space 端点作为滑动后的匹配先验。

`KalmanFilterNew` 使用 Cholesky/SVD 求解和 Joseph 协方差更新。`KalmanBoxTracker` 在预测、量测和 GMC 前后检查有限性，保存最近健康快照，并可由强 BODY/HEAD/FACE 观测重建。HEAD/FACE 重构框通过 `correct_body_from_part()` 以更大不确定性校正人体数值状态，但不会伪造 BODY 的 `time_since_update`、hits 或 `last_observation`；真实人体重新出现后仍按 BODY 强观测恢复。

`collect_nearby_dets()` 同时生成：

- `match_candidates`：动态半径内按距离排序的前 5 个候选；
- `all_candidates`：全部高低分人体，不受半径/top-5 限制。

危险态、ID alert、上一帧 BODY 漏配、长盲或拥挤时改用 `all_candidates`。`BodyReidHypothesis` 用中心、尺寸和 IoU跨帧维护物理候选；局部槽优先保持连续，探索槽按扫描 epoch、最久未尝试和可信位置距离排序。未实际得到 ReID 的候选被赋 `total=-1`，不能靠纯几何提交。有限预算与假设寿命意味着这是防饥饿调度，不是“每个短暂目标必定识别”的保证。

## 主目标融合与恢复门

`match_main_target_unified()` 的 Stage 1 先从全部当前 BODY 取得 ReID/anchor 等事实；临时 preliminary 排名只服务 Face 排除调度。Face 后，当前帧 `reliable_face_non_main` 从主目标身份竞争集合移除，才组合当前/遮挡前 ReID、可见度分带 anchor、KF IoU、头部一致、遮挡者排斥及已确认他人的负证据，形成最终身份排序。Stage 2 仅在 final Top1/Top2 仍歧义时成对加入 OKS、身体比例和肩部连续；任一候选或任一特征 UNKNOWN 都不能形成单边 bonus。Pose refinement 不能绕过后续 hard gate；物理 scene/OCC 输入始终来自完整 BODY 集合。

评分后仍须经过 anchor、头部、运动、共存、runner-up gap、分离期保持和 OCC commit 等门。长盲、瞬移、头体不连续或全图探索命中会进入 provisional。其后还有最终 measurement 可靠性层：危险态、多人贴近、最近 200 ms 内存在人物身份歧义、ID alert 或 `UPPER/HEAD_ONLY` 中的 BODY，若没有 Face、强且有 margin 的 ReID 或强 anchor，只能标记为 `UNCERTAIN`；孤立且一致的头部支持只在不存在人物身份歧义时可恢复 BODY。`UNCERTAIN` 继续作为搜索假设，但不更新主/头 KF、ReID feature、质量时钟或 BODY 命中；外层可并行输出冻结快照的有限 prediction。

`PersonIdentityAmbiguityContext` 与 `OcclusionState`、当前帧 target identity recovery 相互独立。scene risk 只由实际 overlap/crossing、两框 merge、HEAD/FACE 多 owner 竞争、confirmed-other 占用，或 `id_switch_alert_` 同时存在局部候选竞争触发；单独的 close count、家具遮挡、单人低可见度和普通 OCCLUDED 不触发。竞争证据停止后通过独立 200 ms hold 自然过期。Face/BODY 强身份可以恢复当前目标输出，但不能提前清除 scene risk。

最终提交后才允许更新质量监控、可信全图搜索中心、embedding/anchor gallery 和共存轨迹所有权。embedding 只在孤立、CLEAR、身体基本完整且证据安全时慢速更新。

二级轨迹的空间占用与身份排除是两个信号。当前帧已关联真实 detection 的 secondary 由 `secondary_frame_observations_` 标为 current-corrected；其余 KF box 为 prediction-only。两者都可阻止 HEAD/FACE_TRACK 这种弱空间 continuity，但只有真实 Face/BODY/secondary embedding 比较可产生 `NEGATIVE`，prediction-only occupancy 保持 owner evidence `UNKNOWN`。无 GMC 的 emergence 另用 `relative_motion_history_`：只有 main 与同一 secondary 在同一 frame/timestamp 都提交真实 BODY observation 才写入，任何 KF prediction、provisional、UNCERTAIN 或 quarantine 都不得写入；所得浮现点只用于 search/hold，不进入 identity gate 或 trusted lead。当前 trace 会记录 secondary association 的 row/column runner 以及 A-relative 几何连续性，但这些诊断量尚未接入 association 或 relative segment gate。

## 状态与辅助锁

### OcclusionState

```text
CLEAR --onset overlap>=2 且 BODY 近期真实可见--> OCCLUDED
OCCLUDED --overlap<=1 且 BODY 或已接受 HEAD/FACE 近期可见 1 帧--> RECOVERING
RECOVERING --无再次重叠持续 200ms--> CLEAR
RECOVERING --overlap>=2--> OCCLUDED
OCCLUDED --持续 3600ms--> CLEAR（超时保护）
```

此外，OCC 中已通过最终门的强 BODY 身份或 Face identity 可直接推进到 RECOVERING，不必等待下一帧的 `time_since_update` 回流（`LightTracker.cpp` — `match_main_target_unified()`、`try_face_only_continuity()`）。进入 OCC 会保存遮挡前 embedding/速度、遮挡者和浮现方向；`occ_kf_clean_` 记录该遮挡窗口内人体 KF 是否吃过非人脸 BODY 观测。

### VisibilityState

`FULL / MOSTLY_FULL / HALF / UPPER / HEAD_ONLY` 由最终可靠 BODY 的 current Pose observation 每帧调用 `assess_visibility()` 更新：Pose 结构点足够时使用最深可靠结构点，Pose 失败或结构不足时使用当前可靠 BODY 框比例与底部截断退化。状态经 EMA 和 160 ms 迟滞后改变 ReID/空间/头部权重。它描述身体可见程度，不描述是否有人体交错。

### 影响控制流的辅助状态

| 状态 | 作用 |
|---|---|
| `id_switch_alert_` | 质量窗口触发后回滚主 embedding、冻结更新、扩大身份检查；按 3600 ms 自动清除或由人脸确认清除。 |
| `face_locked_` + 最近确认脸时间 | 提高危险期身份门，并允许短时真实脸检测桥接；身份、检测断档或歧义超时即停止。 |
| `face_global_pending_` | 远距 FaceReco 命中但分数未达直接门时，固定同一脸假设做二次确认。 |
| `face_priority_frame_` | 特殊失配上下文把当帧 face slot 提升到优先预算；与候选 Pose 硬预算独立。 |
| `person_identity_context_` | 记录人物竞争原因、最后证据时刻和 direct competition frame；独立保持 200 ms，强身份恢复与 scene risk 退出分离。 |
| current-frame identity evidence | Face 与 BODY 均为 `POSITIVE/NEGATIVE/UNKNOWN`；随 `matchPersonFaces()` 每帧重建，不跨帧保存 candidate negative。 |
| `preserve_face_search_state_` | BODY 已有候选但尚未完成人脸身份仲裁时，阻止 `update()` 在 CLEAR 提交后清空 raw-face 轮转/二次确认状态，并允许确认脸桥接覆盖未闭环 BODY。 |
| `frame_measurement_reliability_` / `MainMatchResult::reliability` | 区分 `NONE/UNCERTAIN/RELIABLE`；只有 `RELIABLE` BODY 才能成为主 KF measurement。 |
| `pending_active_` / `reacq_defer_count_` | 高惊奇或身份不确定 BODY/扫脸假设；未确认期间继续搜索，若可靠快照仍在安全窗口内可同时输出 prediction。 |
| BODY/Face hypothesis + scan epoch | 在固定模型预算下跨帧轮转全图物理候选。 |
| `overload_mode_` | 慢帧迟滞触发；当前把 face slot 限为 1并禁止 secondary ReID，不改变 Pose 单帧硬上限。 |
| `pose_cache_` / Pose budgets | 每帧 `update()` 开始清空；ambiguity candidate 最多 2 次，最终可靠 BODY 另有 1 个 `FINAL_BODY_OBSERVATION` slot。同 `(source,index)` 在同帧复用，最坏 3 次 inference。 |
| `current_pose_observation_` | 仅当前帧最终可靠 BODY 的 Pose、box、动态比例与输出消费者；每帧重置，服务 visibility、orientation、PTZ trunk，不是跨帧身份 reference。 |
| `last_committed_pose_*` / `committed_pose_box_` | 可靠跨帧 Pose reference：仅 final BODY Pose 至少 5 点且不在 PTZ spatial-invalid epoch 时更新，供下一帧短时 OKS/肩部连续性使用；`reset()` 全清。 |
| `last_body_observation_ms_` | 仅可靠 BODY measurement 刷新；HEAD/FACE 只刷新各自及通用真实观测时间，不能冒充 BODY freshness。 |
| `part_output_box_valid_` / `closeup_output_active_` | 在 BODY/HEAD/FACE 来源切换和近场裁剪时稳定最终输出，不参与身份判定。 |
| `motion_history_` | 最近 10 条/1000 ms 安全真实人体框；同帧固定优先级仲裁，prediction 不写入，`reset()`/换目标全清。 |
| `frozen_prediction_` | `IDLE/ACTIVE/EXHAUSTED`；ACTIVE 冻结真实 anchor、历史速度、quality/mode、两个时钟和位移上限，安全失败后 EXHAUSTED 防旧 anchor 重启，新真实观测或 `reset()` 回 IDLE。 |
| `coast_search_hint_` | 历史 PRED 搜索状态；PRED 输出已停用，进入外部 PTZ blind 的 HOLD 路径会清除，当前不参与 recovery。 |
| `secondary_frame_observations_` | 当前帧二级 tracker 与真实 detector BODY 成功 association/correction 的 metadata；每帧开头清空，用于区别 current-corrected 与 prediction-only 空间占用。 |
| `relative_motion_history_` | 以 secondary tracker id 分组的最多 5 条同帧双真实 BODY 相对中心；过龄、quarantine、ownership repair、tracker cleanup 或 reset 清理，只用于 emergence search/hold。 |

## Face、Head 与重构框

`matchPersonFaces()` 的 BODY 归属只决定绑定路径；全部 detector 原始脸/头仍保留在 recovery 池，所以被低分 BODY 包含、未正式关联或 standalone 的部件仍可进入恢复。每 BODY 只绑定一张上部居中脸，限制误归属和模型调用数。

`face_recognition_verification()` 在同一函数内处理 BODY 绑定脸与原始 recovery face。所有 FaceKps/FaceReco 路径共享预算并按脸框做帧内缓存；原始脸以 `FaceRecoveryHypothesis` 调度，空间门内脸优先但门外脸不删除。基础模板匹配阈值来自 `FaceRecognitionSystem`；远距离/大位移命中低于全局直接门时还要求 350 ms 内同一脸两次确认。多脸最佳/次佳差不足也拒绝。

Face inference cache 同时保存当前帧事实和三态 identity evidence。完成有效模板比较并命中 A 为 `POSITIVE`；普通 mismatch 仍为 `UNKNOWN`，不会仅因未命中 A 变成 BODY hard negative；没有完成有效比较的所有路径也为 `UNKNOWN`。A 正匹配门保持 `0.50`，可靠非 A 另用相似度 `<=0.40` 的保守负门，二者之间的缓冲带保持 UNKNOWN。只有 `match_main_target_unified()` 的 preliminary 危险竞争者 Face 验证作用域允许升级满足该负门的 completed mismatch（包括该作用域复用的当前帧 cache）；再同时通过既有高质量门、当前帧 FACE→BODY owner 唯一、且该脸是 owner 的正式上部居中关联脸，tracker 才产生 `reliable_face_non_main`。该作用域之外的普通 A 确认、face-only recovery 和缓存复用不具此权限。它仅从 final 主身份竞争集合排除该 BODY，当前帧结束即失效，不能跨 hypothesis 继承。FACE_TRACK/HEAD 选中脸后先查询该 evidence 及其 raw BODY owner evidence；明确可靠 Face/BODY negative 永远不能被空间连续性推翻。HEAD 使用同一 owner evidence 规则。scene risk 活跃时，没有当前帧强 positive 的 FACE_TRACK/HEAD 都不能提升为 `RELIABLE`；scene risk 不活跃且 evidence 为 UNKNOWN 时保持原有家具遮挡、只露头/侧脸的 continuity。

接受人脸后，优先用同帧包含该脸的真实头重构，否则用脸尺度；接受头则用学习到的头身几何。完整比例框经 `clip_reconstructed_body_to_frame()` 与画面求交，再校正人体/头部 KF、更新可信搜索参考并返回 FACE/HEAD。纯头没有身份 embedding，只能依赖现有头 KF的新鲜度、距离/尺寸/歧义及 owner/已知他人门，因此多人头不确定时会 HOLD。

## PTZ Pose-aware trunk output

这一步位于 `match_main_target_unified()` 已接受最终可靠 BODY 后：该 exact winner 每帧请求或复用 `FINAL_BODY_OBSERVATION` Pose；原始 BODY detection/measurement 仍先继续驱动既有 KF、ReID/Face、Motion History、lead、遮挡和恢复链路。current Pose 同时更新 visibility/orientation，并在满足 reference 条件时刷新下一帧 OKS reference；Pose 失败时 visibility 退到当前 BODY 几何、current orientation 为 UNKNOWN，但不得清除既有可靠 orientation reference。PTZ spatial-invalid epoch 不写入会参与后续身份评分的 orientation/shoulder reference。随后 `update()` 只对最终返回矩阵的主行计算 PTZ trunk。双肩都满足 confidence `>=0.30` 且水平肩宽 `>=3px` 时，`x1/x2` 为双肩左右界各扩 `0.35×肩宽`；若任一有效 wrist 位于同侧 shoulder 上方 `max(3px, 0.05×肩宽)`，`y1` 改为肩中点上方 `1.10×肩宽`，否则保持原 BODY `y1`；`y2` 永远保持原 BODY `y2`。专用 PTZ-only transition 位于既有 BODY 输出稳定器之后：trunk 有效时更新 cache，失效最多保持 1 帧，持续失效时有界过渡到当前稳定 raw BODY，恢复时再过渡回 trunk；可靠 BODY 存在时不返回 `NONE`。该状态不写回 tracking。

`kPoseInferEveryN=3` 已不再节流 final BODY Pose。它仅保留为 secondary ReID 的空闲帧节拍，并在每个 due frame 无条件消费，避免 final Pose/reference 失败让 secondary ReID 长期饥饿。`[POSE_OBSERVATION]`、`[POSE_REFERENCE]`、`[PTZ_TRUNK]`、`[PTZ_TRUNK_TRANSITION]` 是可关闭的 match trace。

身份来源与输出几何分开仲裁：BODY 已有强 ReID/anchor/近期绑定脸身份，或者其关联脸与最近确认脸连续、脸体几何一致且只有一个合理人体 owner 时，BODY 获得几何优先。`try_confirmed_face_track_continuity()` 若确认连续脸属于该 BODY，只刷新物理脸和头轨迹，最终继续输出检测 BODY；脸体不一致则拒绝本次物理脸桥接，也不会用部件重构框覆盖强 BODY。上述 BODY 条件不成立时保留原安全行为，由确认脸重构框覆盖未明确归属的 BODY；真正的 FaceReco 身份命中仍可通过原仲裁路径改写目标。实现见 `LightTracker.cpp` — `match_main_target_unified()`、`try_confirmed_face_track_continuity()`、`update()`，诊断见 `[BOX_ARB]`。

## 模型调用规则

| 模型 | 当前调用关系 |
|---|---|
| Detector_yolox | 普通帧固定一次 YOLOX 联合检测；指定帧只在指定 ROI 内一次。 |
| ReID | 指定时一次；普通帧仅对被预算选中的主候选执行；secondary ReID只在无危险、无全图搜索、无人脸优先、无 Pose 刷新、无周期 Face 验证的空闲帧最多追加一次。该许可在主融合前计算，并不感知随后因匹配失败临时触发的人脸验证。 |
| Pose | 指定时对确认 BODY 一次；每个最终可靠 BODY 帧请求/复用 1 次 `FINAL_BODY_OBSERVATION`；真实 Top1/Top2 ambiguity 候选最多 2 次。全部经过 `request_pose()` 与同帧 cache，最坏 3 次。 |
| FaceKps / FaceReco | 注册、BODY 绑定脸和 recovery face 共用 slot；Kps/质量失败会消费 slot但不调用 FaceReco；同脸结果帧内复用。 |

精确预算、阈值和默认开关只在 `CURRENT_STATE.md` 维护。`ModelProfiler.h` 当前默认开启，每帧向 stdout 打印模型调用和 pre/infer/post 时间；它不是匹配 trace。

## C API、输出与 PTZ

`fx_tracker_run()` 零拷贝包装连续 BGR `CV_8UC3`；可选 Y 平面只供 GMC。非 NULL `main_target` 被原样打包为 xyxy-in-`cv::Rect`，C API 本身不验证坐标次序或边界。所有 C++ 异常在 C 边界转为错误码。

核心输出为 `[x1,y1,x2,y2,id]`。当前主 id 固定为 1；`generate_final_results()` 还可输出非主 KF 轨迹 id 2..，`add_other_det()` 输出 900+。C API只复制前 `FX_MAX_PERSONS=5` 行，但 `total_count` 保留核心总行数，并把坐标转换为 `x,y,w,h`。

已接受的 BODY/HEAD/FACE 产生主行且控制权重为 1。有限短时预测也产生置于结果首行的主 id，权重为 `(0,1)` 的时间/位移衰减值；不满足安全条件或到达上限后主行消失、权重为 0。C API没有单独暴露 observation source，因此下游只能由 score 区分真实观测与预测。`get_aim_point()` 和 `set_ego_enabled()` 只在 C++ 类公开，C API未暴露；当前 C API PTZ 集成只能消费 box 和 score。GMC 当前默认关闭，ego 当前也默认关闭；ego 仅应在下游确实将输出拉向画面中心时通过 C++ setter 显式开启。默认值与产品风险见 `CURRENT_STATE.md`。

## 构建、Reset 与诊断

仓库没有构建系统；权威手工命令见 `AGENTS.md` 和 `c_api/README.md`，并必须包含 `AclRuntime.cpp`。本地缺少 OpenCV/Ascend 工具链，当前改造尚未完成目标设备编译验证。未使用的 Predictor/Association/Gait 文件已从源码和构建清单删除；`IveGmc.cpp` 仍是 `USE_HISI_IVE` 条件构建的真实可选能力。C++ 构造函数接收 `LightTrackerConfig`，但其中若干字段仅被保存或只用于轨迹构造/清理；绝大多数当前门控、预算与 GMC/ego 开关仍是 `LightTracker.h` 私有常量。C API始终构造默认配置，没有 runtime setter。

`reset()` 清除轨迹、模板、状态机、候选假设、缓存和输出状态并置 reset flag；`setMainTarget()` 也先调用它，所以 flag 不只表示异常断流。学习到的 ego 闭合速率按源码注释跨 reset 保留。

匹配 trace 默认编译关闭。`-DFX_TRACKER_MATCH_TRACE=1` 后默认写 `/tmp/fx_tracker_match_trace.log`；编译宏/同名环境变量可改路径，64 KiB stdio 缓冲、事件或 25 帧心跳 flush、无 `fsync`、16 MiB 后停止。当前丢失路径重点观察 `MATCH/MEASUREMENT/OUTPUT`：`OUTPUT src=NONE` 表示外部 PTZ 应进入或继续 blind。历史 `SHORT_PRED_*` trace 已不应在当前运行路径产生。

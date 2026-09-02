# Current State

## 状态快照

本文件基于 2026-09-01 工作区源码静态审计。仓库目录没有有效 Git 元数据，因此当前 branch、commit、未提交 diff 和历史变更归属均为 **UNKNOWN**。本环境没有完整 SS927 交叉链接工具链和测试素材；历史维护记录显示曾以可用 OpenCV/ACL 头文件完成 C++17 `-fsyntax-only` 检查（含 `FX_TRACKER_MATCH_TRACE=1`），但当前工作区未携带这些 SDK 头文件，故本轮无法重复该检查，也不等同于设备编译或场景回放验证。最新实现已将 ACL global/device/context 生命周期收敛到唯一 `AclRuntime` owner，并把旧全图 YOLOv8n-Pose 替换为候选级 RTMPose-T 0/1/2 次调度；联合检测器现为 YOLOX。目标设备上的 ABI、destroy→recreate、模型 tensor 契约和场景回放仍待验证。

## 当前已经实现

- 已删除未被当前跟踪器 include、持有或调用的历史代码文件：`MainTargetPredictor.{h,cpp}`、`Association.{h,cpp}`、`GaitRecognition.{h,cpp}`，并从 `AGENTS.md`、`c_api/README.md` 和维护构建命令移除。`ModelProcess.*` 是 SS927 五个模型 wrapper 的实际底层，继续保留；`IveGmc.*` 是 `USE_HISI_IVE` 条件构建的真实可选路径；legacy `Track/fx_wrapper` 按 DV500 边界继续保留。
- SS927 与 DV500 联合检测器均已切换为 `/oem/model/yolox_416_face_body_head.om`。两平台都严格验证 FP32 尾维 `[...,3549,8]` 输出；SS927 按 ACL descriptor 验证 416×416 HWC U8 输入，DV500 因原生 SVP-NPU descriptor 表示差异直接从 `input_dims[2]/[3]` 读取高宽，不再用 SS927 的 layout/dtype 规则提前拒绝模型。前处理为 OpenCV BGR、左上对齐 resize、右/下 114 padding，按 stride 8/16/32 解码、`objectness×class score` 过滤和 face/body/head 分类别 NMS。SS927 通过 ACL `ModelProcess`，DV500 保留 `sample_common_svp_npu_*` 原生 task、cached MMZ flush/invalidate 与 output row stride；二者都向跟踪层输出项目约定的 xyxy 框。模型错误继续向上层传播，不伪装成零检测；跟踪阈值与后续匹配算法未调整。
- SS927 当前从 `/oem/model/mobilev2_EmbeddingHead_reid_v1_GeMP_pre.om` 加载人体 ReID；本次只替换部署模型文件名，输入尺寸、预处理、匹配阈值和跟踪策略均未改变，其特征维度及阈值适配仍需设备端确认。
- SS927 与 DV500 的 FaceReco 模板 cosine 基础门现统一为 `0.50`，RTMPose SimCC 关键点有效门统一为 `0.40`；Pose 后处理、OKS 和 body proportion 使用同一可见性门。Face 的全局直接重定位门 `0.65`、margin、二次确认、Pose/OKS 调度及其它身份阈值均未改变。降低 FaceReco/关键点基础门可提高边界样本召回，但也扩大噪声或误认风险，仍需两平台固定 replay 验证。
- `AclRuntime` 现为 ACL global/device/context 的唯一 RAII owner：ACL global runtime 在进程首个 Tracker init 时只调用一次 `aclInit` 并保持到进程退出；每个 Tracker 实例只创建/销毁自己的 context/device。`LightTracker` 中它先于所有模型声明，并由显式析构函数在 runtime 仍有效时先调用幂等 `release_models()`，随后成员逆序析构仍为安全 no-op，确保所有模型在 context/reset 前释放。同步推理没有使用 stream，旧显式 stream 已删除；第一阶段拒绝第二个并存的已初始化 runtime，完整 destroy 后可重新创建而不重复 `aclInit`（`AclRuntime.*`、`LightTracker.h/.cpp`）。
- Detector/ReID/FaceKps/FaceReco/Pose（以及未接线 Gait）现只释放自身 `ModelProcess`、dataset 和输入 buffer；release 幂等，init 任一步失败立即回到未初始化状态。`ModelProcess` 明确区分 wrapper 所有的主 input buffer 与自身所有的 output/work/weight buffer，并补齐部分创建失败回滚。
- `g_isDevice` 已从维护代码和 legacy `Track.cpp` 删除。所有 `cv::Mat`/CPU vector 输入显式 Host→Device；模型输出显式 Device→Host。Detector memcpy/Execute 失败不再释放长期输入 buffer，普通帧失败由 C API 返回 `FX_ERR_MODEL_RUN`，不会伪装成正常零检测。
- `c_api/lifecycle_smoke.c` 提供 100 轮真实推理 destroy→recreate 回归；`c_api/lifecycle_fault_smoke.c` 配合默认关闭的 `FX_TRACKER_LIFECYCLE_TEST_HOOKS`，覆盖五个模型级 init 失败后同对象重试，以及 Detector memcpy/Execute 失败后的下一帧恢复。测试程序已经加入源码，但尚未在目标设备执行。
- Pose 模型已切换为 `/oem/model/rtmpose-t.om`：每次对一个 BODY xyxy 候选做 1.25 padding 的 TopdownAffine，输入 192×256 RGB U8/AIPP，按两个 SimCC 输出 `17×384`、`17×512` 解码并映射回原图（`PoseEstimator.h/.cpp`）。
- 主匹配仍先做不依赖当前帧 Pose 的 base ranking；真实 Top1/Top2 歧义最多使用 2 次候选 Pose，且双方同一 Pose feature 都有效时才公平加入排序。最终可靠 BODY 另有独立 `FINAL_BODY_OBSERVATION` slot：每个可靠 BODY 帧请求或复用该 exact winner 的当前帧 Pose，稳定帧通常 1 次、歧义帧最坏 3 次。全部请求经 `(frame,source,index)` cache 去重；final slot 不会被候选预算挤掉（`LightTracker.h/.cpp`）。
- Pose 状态已分层：current observation 每个可靠 BODY 帧更新，服务当前 visibility、orientation、动态 body proportions 与 PTZ trunk；Pose 失败时 visibility 退到当前 BODY 几何，current orientation 为 UNKNOWN，但不清除已有跨帧可靠朝向参考。`last_keypoints_`、`committed_pose_box_`、`last_committed_pose_*` 仅在 Pose 至少 5 个有效点且不处于 PTZ 旧坐标系失效 epoch 时作为跨帧可靠 reference 刷新。`anchor_body_proportions_` 保持注册级长期身份锚点，不能被逐帧 Pose 覆盖。候选 loser、UNCERTAIN/pending、HEAD/FACE 与预测不提交主 Pose 状态。
- 原 `kPoseInferEveryN=3` 不再节流 final BODY Pose；它只保留为 secondary ReID 的空闲帧节拍，并在 due frame 无条件消费，故 final Pose 失败不会让 secondary ReID 连续饥饿。Pose 与 secondary ReID 的模型预算、调度语义相互独立。
- OKS 只比较历史与候选共同可见的关键点，共同点少于 3 个即 UNKNOWN；历史关键点按 committed BODY box 到当前 KF predicted BODY box 做平移/缩放补偿，并在 600 ms 内线性衰减，过期关闭。
- `last_body_observation_ms_` 单独记录可靠 BODY 观测。HEAD/FACE 仍可刷新通用 `last_real_obs_ms_` 并以独立 source 写 Motion History，但不会伪造 BODY freshness 或 BODY hit。
- 丢失后的 PTZ blind slide 已统一由外部 `track-ptz` 控制层完成：该层固定最后真实 BODY 框，按实际下发 pan speed 积分，累计图像等效距离达到 `1.0×last_real.w` 后停止。`PTZ_BLIND_STOPPED` 是当前丢失 episode 的硬终态，后续 LOST 帧只保持停止；只有新的真实 BODY 才恢复 `IDLE` 并允许下一次 blind slide，避免重复按同一 BODY 宽度移动。`LightTracker` 的 `update()` 不再调用 `try_short_prediction()` 返回 `PRED`；无安全真实 BODY/HEAD/FACE 时只返回 `NONE/HOLD`，使 PTZ 层进入或继续 blind。进入该路径会清理任何历史 frozen prediction 与 `coast_search_hint_`，因此不再让 Tracker 的 image-space 预测端点参与滑动后搜索。历史 `try_short_prediction()` 实现暂保留为未接线内部代码，后续可单独删除。
- `track-ptz` 现在在每次 `track_run()` 前只读回传 `IDLE/SLIDING/STOPPED` blind phase；只有速度积分已产生实际 pan 位移才上报 `SLIDING/STOPPED`，PTZ 死区内维持原空间快速路径。Tracker 在此边界保存最后可靠 BODY `xyxy` 框为独立 PTZ 搜索锚点：滑动中/停止后只用它生成本地 BODY pool，并从全图公平轮转最多注入一个既有 ReID budget slot。该锚点不回写 KF、lead、Motion History、ReID/Face template 或真实观测时钟。
- PTZ 坐标系失效 epoch 内，BODY base score 不再使用旧 KF IoU、head、occluder、trajectory/weak spatial conflict、Pose 或 face-lock/incombent 位置保持；confirmed-other、可靠 Face 非 A、anchor/reID 等真实身份安全门仍保留。普通 Face mismatch 仍是 UNKNOWN。Face recovery 改为当前帧全脸公平轮转，不使用旧 face reference/pending 坐标；当前 BODY–Face 包含关系只用于归属。长盲的旧位置 provisional 也不会跨该边界复验；当前首版只允许 Face 正证据、强 ReID direct 或强 anchor BODY 真正提交。首次可靠 BODY 提交会令主 KF 以当前框重建，清空旧 spatial search/secondary relative 状态，并冻结该首帧 feature 更新；下一可靠 BODY 后恢复现有 feature gate。设备 replay 尚未验证。
- `LightTracker` 是唯一接线的主 tracking pipeline；维护中的 C API 直接持有它（`c_api/fx_tracker.cpp:18-20`）。
- 单次联合 face/body/head 检测后，高分 BODY（默认 `>0.7`）和低分 BODY（`>0.3`）都进入当前候选体系；它们不是两轮 ByteTrack 关联（`LightTracker.cpp` — `matchPersonFaces()`、`extract_detections()`、`update()`）。
- 主候选已接入 ReID、可见度分带 anchor gallery、空间/头/肩/Pose、遮挡者和已知他人负证据，并在最终门后才提交（`LightTracker.cpp` — `match_main_target_unified()`）。
- 三态遮挡机、五态可见度机以及 alert、face lock、全局脸二次确认、face-priority/preserve-search、BODY provisional、全图扫描 epoch、long-coast、短时 prediction 和近场输出稳定等辅助状态均已接线；不能只用三态机解释 HOLD（状态见 `LightTracker.h`，推进见 `LightTracker.cpp` — `update()`、`match_main_target_unified()`）。
- BODY 未提交时，代码可按顺序尝试原始脸 FaceReco、确认脸检测级桥接和头 KF 连续；接受后重构、裁剪、校正人体 KF 并输出 FACE/HEAD（`LightTracker.cpp` — 三个 `try_*continuity()` 与 `update()`）。
- 人体与头部 KF 具有有限性检查、Joseph 更新、健康快照、长盲速度冻结和强观测重建；部件校正不会伪造 BODY 命中计数（`KalmanFilter.cpp` — `predict()/update()`；`KalmanBoxTracker.cpp` — `predict()`、`correct_body_from_part()`）。
- BODY 和 raw face 都有跨帧物理假设与防饥饿轮转；弱 BODY 的全局人脸仲裁可撤销 BODY并路由到 face-only 重定位。该调度受预算和假设寿命限制，不承诺短窗口内扫描完所有目标。
- 人脸模板支持初始、确认头场景补注册、延迟注册、质量升级和优质模板冻结；flag 与模板实际状态会在恢复/延迟注册路径校正（`LightTracker.cpp` — `setMainTarget()`、`try_register_face_from_confirmed_head()`、`try_deferred_face_register()`）。
- 主输出优先来自已接受的 BODY/HEAD/FACE。BODY 最终分为 `RELIABLE/UNCERTAIN`：前者才形成主 measurement，后者保持 pending 搜索但按 unmatched 推进 KF。最近最多 10 条安全真实 BODY/FaceReco FACE/确认脸 FACE_TRACK/HEAD 人体框以固定优先级写入 Motion History；同帧只保留 `BODY > FACE_IDENTITY > FACE_TRACK > HEAD`，UNCERTAIN/provisional 不得写入。没有可靠真实观测时，Tracker 只返回 `NONE/HOLD`；PTZ blind 的移动不再由 Tracker 的历史速度、冻结 anchor 或 PRED 负责。`try_short_prediction()` 及其 Motion History 估计实现仍在源码中但当前未接线，不更新身份、特征、真实观测时钟、Motion History 或 KF measurement。
- 已确认的最终可靠 BODY 每帧请求/复用 current Pose 后，在 `update()` 的最后 BODY 输出分支生成 shoulder-based trunk：双肩 confidence `>=0.30`、肩宽 `>=3px` 时，左右边界取肩宽加 `0.35×` padding；任一手腕高于同侧肩 `max(3px, 0.05×肩宽)` 时，顶部改为肩中点上方 `1.10×肩宽`，否则保持原 BODY 顶部，底部始终保持原 BODY 底部。PTZ-only transition 在 trunk 有效时输出/缓存 trunk，失败后最多保持 1 帧，继续失败则从最近 PTZ 输出有界过渡到当前稳定 raw BODY；恢复 trunk 时反向平滑。可靠 BODY 存在时绝不因 Pose/trunk 失败返回 `NONE`。该层不改 detector、BODY/KF measurement、ReID/Face、Motion History、lead 或身份/恢复状态。trace：`[POSE_OBSERVATION]`、`[POSE_REFERENCE]`、`[PTZ_TRUNK]`、`[PTZ_TRUNK_TRANSITION]`。
- `PersonIdentityAmbiguityContext` 现在独立记录人物身份歧义：只有实际 BODY overlap/crossing、两框合并、部件 owner 竞争、confirmed-other 占用或 `id_switch_alert_ +` 局部多人竞争才刷新；`close_det_count>=2` 本身不会触发。证据消失后风险独立保持 200 ms。FaceReco/强 BODY 可以立即恢复当帧真实目标，但不会清除仍存在的 scene risk，避免下一帧脸消失后 HEAD/FACE_TRACK 又靠空间连续性接管（`LightTracker.h/.cpp`；trace：`[IDENTITY_CTX]`、`[IDENTITY_RECOVERY]`）。
- FaceReco 当前帧事实已拆成 `POSITIVE/普通 mismatch/UNKNOWN`：完成比较并命中 A 才是 `POSITIVE`；普通 mismatch、未调度、预算不足、脸太小、KPS/质量/朝向/特征失败或无模板均保持 `UNKNOWN`，不能仅因“未命中 A”排除 BODY。现有 A 正匹配门保持 `0.50`；只有相似度 `<=0.40` 才具有进入可靠非 A 审计的资格，`(0.40,0.50)` 缓冲带始终为 UNKNOWN。只有 preliminary 危险竞争 BODY 的专用 Face 验证作用域允许把该低相似度 completed mismatch（包括该专用作用域复用的当前帧 cache）再审计为 `reliable_face_non_main`；还必须同时通过既有高质量门（`0.80`）、当前帧唯一 FACE→BODY owner、且该脸是该 BODY 的正式上部居中关联脸。该专用作用域之外的普通 A 确认、face-only recovery 与缓存复用中的 mismatch 一律保持 UNKNOWN。该负证据只取消该 BODY 成为 A 的资格，不建立 B/C 模板、不跨帧/跨 hypothesis 保存；confirmed-other 与可靠 FULL/MOSTLY_FULL BODY 严格 hard reject 保持原有语义。
- `match_main_target_unified()` 现把 Face 排除拆成两个无副作用层次：原始 ReID/anchor 的 preliminary 排名只在多人近分、且共享 Face budget 至少还剩两个 slot 时，为一个危险竞争 BODY 安排一次专用 Face 验证；随后从 `final identity-valid` 集合移除当前帧 `reliable_face_non_main`，重新计算 ReID ambiguity、Top1/Top2 Pose、winner/runner-up/margin、identity spatial/motion HOLD 和 provisional/commit。preliminary 的 Face 返回不会作为最终 winner 或 measurement 使用。完整 BODY 检测集合不因此缩小，`close/overlap/OCC/coexist/crowded/measurement safety` 继续看所有真实人物。正常预算、ReID/Face/Anchor/OCC 阈值均未改；设备 replay 尚未验证排除强度和误排率。
- HEAD 与 FACE_TRACK 在原有距离、尺寸、d1/d2 歧义和 known-other 门之后、任何真实 observation 副作用之前执行统一 continuity permission：`NEGATIVE` 永远禁止；scene risk 活跃时 `UNKNOWN` 禁止；普通低风险场景的 `UNKNOWN` 保留旧 continuity。Face `POSITIVE` 只走 FACE_IDENTITY 强身份路径，不能再由 FACE_TRACK 作为空间桥接重复提交。被阻止后自然回落到 Frozen Motion Prediction/NONE，不更新 KF measurement、Motion History、真实观测时钟或输出（trace：`[FACE_EVIDENCE]`、`[BODY_EVIDENCE]`、`[CONTINUITY_GATE]`）。
- `gmc_enabled_==false` 时，same-cloth predicted-center outlier 与仅由二级 KF predicted box 产生的 coexist overlap 已从 identity hard veto 拆为 `spatial conflict`。候选仍完成已有身份评估；motion 冲突只能让 tentative identity winner HOLD，不能先删掉它再让后续弱身份候选获胜。二级 embedding 明显排除、严格 `reliable_face_non_main` 和 FULL/MOSTLY_FULL BODY identity reject 仍保留 `NEGATIVE`；普通 Face mismatch 不产生负证据。HEAD/FACE_TRACK 对 current-corrected 或 prediction-only known-other 的空间占用都可安全阻止弱 continuity，但 prediction-only 不写 owner `NEGATIVE`、不刷新跨帧 scene risk（trace：`[COEXIST_GATE]`、`[MOTION_CONFLICT]`、`[KNOWN_OTHER_SPATIAL]`）。
- 无 GMC 的 occlusion emergence 方向只来自 main 与同一 secondary 在同一 frame/timestamp 都有可靠真实 BODY observation 时的相对中心历史；最多 5 条、使用相邻相对速度的稳健中位方向，quarantine/reset/轨迹过期会清理。没有可靠 relative history 时 source 为 `none`，不再回退 `main->occluder` 或 `pre_occ_velocity_`。emergence 只服务 recovery search、ReID hypothesis 调度和 separation HOLD，不覆盖 trusted lead、不产生身份负证据或 KF measurement（trace：`[RELATIVE_OBS]`、`[EMERGE]`）。
- 为验证 secondary IoU 换人是否真实污染 emergence，诊断构建现记录配对的双侧 association ambiguity：`[SEC_ASSOC]` 同时给出 tracker→detection 的 row runner/gap 和 detection→tracker 的 column runner/gap；`[SEC_ASSOC_REL]`、`[RELATIVE_GEOM]` 记录相对主目标的归一化位置/尺寸连续性，`[RELATIVE_ESTIMATE]` 记录污染是否被现有方向一致性与 residual 检查拦截。`[EMERGE_POINT]` 和 `[SEP_DECISION]` 用于确认方向是否生成有效浮现点及是否实际造成 separation HOLD。上述字段仅用于 trace，不改变 secondary association、relative history 写入或 emergence 判定。
- 已提交 BODY 若已有强 ReID/anchor/近期绑定脸身份，或其关联脸与最近确认脸连续、脸体几何一致且 owner 明确，确认脸连续性只更新脸/头物理状态，最终输出和人体 KF 保留真实 BODY；FACE_TRACK 与该 BODY 几何不一致时只拒绝桥接，不反向覆盖强 BODY。BODY 身份/归属仍不明确时才允许 FACE/HEAD 重构框覆盖（`LightTracker.cpp` — `match_main_target_unified()`、`try_confirmed_face_track_continuity()`、`update()`；trace：`[BOX_ARB]`）。
- 文件 trace 与逐帧 ModelProfiler 均已接线；trace 默认关闭，ModelProfiler 当前默认开启（`LightTracker.cpp:38-144`、`ModelProfiler.h:18-31`）。

## 当前运行策略与重要值

这些值是当前源码常量，不是推荐调参结论：

| 策略 | 当前值 |
|---|---|
| 遮挡分离确认 / RECOVERING | 1 帧 / 200 ms |
| 人物交错身份风险保持 | 独立 `kPersonIdentityRiskHoldMs=200 ms`；真实人物竞争持续存在时刷新，强身份恢复不清零 |
| OCC / ID alert 超时 | 3600 ms / 3600 ms |
| 长盲身份复核 | 500 ms；弱假设最多延迟 1 次 |
| 主 ReID 上限 | CLEAR 3；危险 4；CLEAR 且 close>=3 也可到 4 |
| ReID 直接门 | FULL 0.40；partial 0.36；anchor floor 0.36；margin 0.10 |
| Anchor 直接确认 | 0.68 |
| Anchor veto | 基础 CLEAR 0.40 / danger 0.50；满足空间/视角护栏时才可放宽到 relaxed 0.28、danger 0.22 或 danger-crowded 0.32 |
| Pose | RTMPose-T；每个最终可靠 BODY 1 个 final observation slot；Top1/Top2 ambiguity 候选最多 2；同帧 cache 复用，最坏 3 次 |
| Face slot | normal 1；priority 3；最小脸高 14 px；FaceKps 质量 0.40 |
| Face 身份门 | 模板匹配基础阈值 0.50；远距/大位移 direct sim 0.65，否则同脸 2 次且间隔 <=350 ms；多脸 margin 0.08 |
| 头轨迹新鲜期 | 15 帧；扩展重捕 2 帧确认 |
| KF 长盲冻结 | 1500 ms |
| PTZ 丢失滑动 | 外部 `track-ptz` 固定最后真实 BODY 框，按速度积分至 `1.0×last_real.w`；`LightTracker` 无安全真实观测时输出 `NONE/HOLD`，历史 PRED 实现未接线 |
| 非主轨迹寿命 | 900 ms；最多 8 个 |
| 全图物理假设寿命 | BODY / face 均为最后出现后 8 帧 |
| GMC / ego | GMC=false；ego=false（可由 C++ setter 显式开启） |

实际 ReID/Face 阈值是否适配当前部署模型、设备 FPS 和场景分布：**UNKNOWN**，需要回放评测而不是继续凭单段日志调整。

## 已停用或未接线

- 历史 `MainTargetPredictor.*`、`GaitRecognition.*`、`Association.*` 均未接入 `LightTracker`，现已从维护源码和构建清单删除。
- GMC：OpenCV 和可选 IVE 实现存在，但私有 `gmc_enabled_=false`，C API 也无开启接口。
- `get_aim_point()` 与 `set_ego_enabled()`：C++ 可用，C API 未暴露。当前公开 PTZ 集成只得到框和控制权重。

## 源码明确标记为停用或替代的方案

- 丢失期 `MainTargetPredictor` coast/保持框及 `LightTracker::try_short_prediction()` 输出均未接线。外部 `track-ptz` 是唯一的 blind slide owner；Tracker 在没有安全真实观测时返回无主框，后续只执行主目标重捕。
- 次级轨迹的颜色直方图关联/排除已从当前 `LightTracker` 删除，次级轨迹改为纯 IoU，身份负证据使用受预算约束的 ReID（`LightTracker.cpp` — `update_secondary_tracks()`、`update_secondary_features()`）。
- 头/脸重构框不直接把画外完整人体交给 PTZ；当前先按完整比例重构，再与画面求交，并单独稳定可见框（`LightTracker.cpp` — `clip_reconstructed_body_to_frame()`、`stabilize_returned_box()`）。
- 当前部件输出不使用较低 PTZ 权重表达“身份信任不足”；通过安全门后 HEAD/FACE 与 BODY 一样为 1，身份安全由接受门负责。由于没有 Git 历史，无法确认该策略在何次修改中替代了旧行为。

## 当前验证状态与可确认失败边界

- **设备编译、ABI 加载、RTMPose 模型输出、实时帧率和所有跟踪场景结果均未在本工作区验证。** 仓库没有测试视频、标注、测试程序或结果报告；任何“已经解决某场景”的结论均为 **UNKNOWN**。
- 检测器不产出目标 BODY/HEAD/FACE，或人物遮挡场景仅剩身份不确定 BODY 候选时，恢复分支可能没有可靠 measurement；Tracker 此时返回 `NONE/HOLD`，由外部 PTZ blind 以最后真实框限距滑动。跨帧人物风险已覆盖常见“两框接近/重叠后合并为一框”，但若 detector 从未在任何一帧同时给出主目标与遮挡者，代码没有可确认的人物交错证据；该边界以及 200 ms 保持是否足够均尚未 replay 验证。
- 任意位置露脸找回并非无条件保证：必须已有有效人脸模板和历史 BODY 尺寸，脸高至少 14 px，FaceKps 质量至少 0.40，并在有限 slot/假设寿命内得到实际推理机会。
- 全图 BODY/face 调度的目标是避免稳定候选长期饥饿；快速出现/消失、检测框跨帧合并分裂、关键点反复失败或候选寿命小于轮转周期时仍可能错过。
- 纯头连续没有身份特征；头 KF 陈旧、多个头距离接近、尺寸不一致或 owner/已知他人门触发时会安全 HOLD。
- 远距人脸基础命中低于 0.65 需要 350 ms 内同一物理脸再次命中；检测/预算不能在该窗口复验时不会重定位。
- GMC/ego 默认关闭时，KF、lead、IoU 和其它图像空间先验会混合人物与相机运动。已知云台闭环具有启动延迟、死区和随“距构图目标”变化的非线性速度：持续移动的主目标正常会停在中心（或产品配置的左/右构图点）一侧，停止后才渐近回到构图点。因此“目标没有立即回中心”不是 KF 滞后的充分证据。外部 blind slide 开始后，旧图像空间先验是否应在身份重捕中旁路仍待下一轮实现与设备 replay；当前历史 `coast_search_hint_` 不再更新。
- 二级轨迹仍用纯 IoU 关联；即使 relative history 的每个样本都是双真实 detection，交叉期间 secondary ownership 若发生 IoU 换人，方向历史仍可能失真。quarantine/ownership repair 会立即清理该 tracker 的 relative history，但 replay 必须验证修复触发是否及时。
- 当前尚未启用 A-relative continuity validator。若 replay 证明污染会穿过现有 estimator 并影响行为，计划的 segment 语义固定为：`AMBIGUOUS` 表示当前 observation 本身不可信，清空旧 segment 且当前点不作新 seed；`DISCONTINUOUS` 表示当前 observation 可信但与旧 segment 不连续，清空旧 segment并允许当前点作新 seed。新 segment 仍须至少 3 个连续可靠 observation、2 个有效速度样本才可产生 emergence。该计划不是当前已实现行为。

## 已知问题与风险

### 构建与 ABI

1. 规范 C API 构建源码列表现已包含 `AclRuntime.cpp`，且不再依赖 legacy `Track.cpp` 提供 `g_isDevice`。但本工作区没有 OpenCV/Ascend SDK，尚未在目标工具链确认编译、共享库加载和 ACL API/SDK 版本兼容性。
2. legacy wrapper 已提供 `track_release()`，其直接 delete `Track` 并触发完整模型/context/device 释放；ACL global runtime 按设备实测保持到进程退出。destroy/recreate 之间不得 `dlclose` 后重新加载库，否则新库副本无法得知已初始化的 ACL process runtime。
3. C API 只检查 frame 非空和尺寸为正，不验证 `main_target` 的 `x2>x1/y2>y1` 或画面边界。由于核心把 `cv::Rect.width/height` 当 `x2/y2`，普通 `area()` 不是可靠的 xyxy 合法性检查；调用方必须先校验和裁剪指定框。
4. `reset_flag` 在显式 reset、自动 reset 和 `setMainTarget()` 内部 reset 时都会置 true，不区分原因；完成正常指定后也应由上层按自身状态清除。
5. 无 CI、无本地最小编译检查、无 ABI smoke test；当前源码能否在设备工具链完整构建：**UNKNOWN**。
6. `rtmpose-t.om` 必须确实为 192×256 RGB888_U8 AIPP 输入，并输出 float32 SimCC `17×384` 与 `17×512`；代码允许 ACL input buffer 存在尾部对齐空间，但有效输入不得小于 147456 字节，并严格校验输出元素数。模型契约不同会使 Pose init/run 明确失败。

### ACL 资源生命周期

2026-08-13 已完成源码级重构：

- `AclRuntime` 唯一执行 `aclInit/SetDevice/GetRunMode/CreateContext` 及实例的 `DestroyContext/ResetDevice`；模型 wrapper 中不再存在这些调用。ACL global runtime 按进程生命周期保持，不在单实例 release 时 `aclFinalize`。
- 所有模型资源和 `ModelProcess` 的 release 均可重复调用；半初始化失败会立即回滚，逐帧推理失败不再释放长期 buffer。
- Detector 错误与正常空检测已分离；C API 错误输出仍保持清零。

2026-08-31 已新增 OTA 前释放的观测日志，前缀为 `[TRACK_LIFECYCLE]`：`track_release`、`LightTracker` 析构及每个模型阶段会记录 begin/done 和耗时；`ModelProcess` 的 dataset/data buffer/device buffer/model/model desc 释放，以及 `AclRuntime` 的 context destroy/device reset 会记录真实 ACL 返回码和耗时。设备日志已确认五个模型显式 ACL release 成功后、`lighttracker_dtor_body_done` 之后发生 `free(): invalid pointer`，因而尚未进入 `AclRuntime` 析构。为定位成员自动析构范围，额外接入 `[DTOR_TRACE]`：LightTracker 的主要容器组边界、五个 wrapper、FaceRecognitionSystem 的 clear/release/reset 子步骤、ModelProcess、KalmanBoxTracker、条件 IVE GMC 与 AclRuntime 都会记录 begin/done。该变更只增加诊断，不改变释放顺序、失败后的既有继续析构行为、ACL 参数或 OTA 流程。下一步先在设备上锁定首个未完成的析构范围，再恢复 OTA/UBIFS 因果分析。

尚未验证的边界：目标 ACL SDK 是否接受当前显式 Host→Device/Device→Host copy 路径；真实 destroy→recreate 100 轮（ACL process runtime 仅 init 一次）是否无设备内存增长；各阶段 fault injection 是否覆盖所有驱动失败。这些必须在设备上验证，不能把静态审计视为已解决现场错误。

### 算法/配置边界

- `ego_enabled_` 当前默认关闭；其实现是“输出误差以近似固定响应拉向画面中心”的简化先验。实际产品还存在启动延迟、死区、位置相关速度，且构图目标可以不是中心；当前 tracker 未接收构图目标、云台命令或实际执行位移。因此在取得这些反馈并用 replay 标定前不得开启，C API 也未暴露该开关。
- `LightTrackerConfig` 对 C++ 调用方可见，但 C API始终使用默认构造。`appearance_thresh`、`iou_threshold`、`inertia` 和 `gate` 当前只在构造函数中保存，未在 `LightTracker.cpp` 后续热路径读取；不要把它们误当作可用的 runtime 调参入口（`LightTracker.h:78-88`、`:143-152`，`LightTracker.cpp:158-172`）。
- 过载模式当前把 Face slot 限为 1并停用 secondary ReID；候选级 Pose 不再使用单独的过载 cadence，所有状态都受单帧 2 次硬上限约束。主 ReID 的 3/4 上限和 `body_global_search` 未受 overload 限制。
- Detector 只接受当前 YOLOX 的 416×416、`[...,3549,8]` anchor-major float32 契约；模型若改成转置输出、图内 NMS、FP16 或未内置 sigmoid，初始化/解析会失败或结果错误，必须重新适配而不是放宽检查。
- 公共 C API 文档只强调 id 1 和 900+；当前 `generate_final_results()` 也可能输出非主 Kalman 轨迹 id 2..。上层不应把“非 1”全部假定为 900+ 原始检测。

## 现有文档与源码不一致

- `docs/tracking_pipeline.svg/.pdf` 已明显过时：仍描述 `MainTargetPredictor` coast、固定约 0.7 权重、预测器 GMC 补偿、仅 top-5 候选和每帧全图 Pose，且没有当前 face-only/全图公平轮转/候选 RTMPose 和外部 PTZ blind slide。不要用它恢复当前流程。
- `docs/parameter_tuning_guide.md` 混入大量已删除或改名的帧计数和 Predictor/颜色参数；例如当前是 `kRecoveryMs=200`、`kSeparationConfirmFrames=1`、`kReacqMaxDefer=1`、`kReidAmbiguousMin=0.20`、`kAnchorVetoDangerCrowded=0.32`，颜色直方图已删除。该文档只能作为历史调参记录。
- `docs/algorithm_review_2026-07.md` 是时间点审计，部分“OPEN”、颜色关联和未编译声明已被后续源码取代；其中风险必须重新对照源码。
- `CLAUDE.md` 的主流程总体方向正确，但把 GMC 写成常规步骤、称多个模型 wrapper 自行初始化 ACL，并沿用部分旧名；实际 GMC 关闭，ACL 已由 `AclRuntime` 单一持有。

## TODO / FIXME 与版本状态

源码搜索只发现 `c_api/example_app.c` 两个 TODO：替换空帧取流、接入 PTZ 控制。未发现业务源码中的 FIXME/TBD。没有 Git 元数据，无法判断哪些源码修改尚未提交，也无法从 commit history确认“最近完成”的精确边界。

跨平台同步状态：SS927 停用 `LightTracker` PRED 输出、改由外部 PTZ blind 接管的恢复语义已同步到 `/mnt/c/Users/hgpar/Desktop/700_dv500`：PTZ 只读 phase 透传、最后可靠 BODY 锚框的局部候选池、一个固定预算内的全局公平 ReID 探索槽、滑动期 identity-only 仲裁，以及可靠 BODY 后的 KF 重建/模板 warm-up 均已接入。DV500 的 SVP 模型初始化、`update(..., z)` 和 legacy wrapper 生命周期保持原样。DV500 端尚需设备 replay 验证 PTZ SLIDING/STOPPED、异衣/同衣干扰和重捕时延；本地缺少 SVP ACL 头文件，未完成编译。

## 推荐下一步

1. 在目标 aarch64/OpenCV/ACL 环境按更新后的源码列表编译并加载共享库，确认 `rtmpose-t.om` 输入/SimCC 输出契约、`AclRuntime.cpp` 接线和 `FX_ERR_MODEL_RUN` ABI。
2. 先做单候选 RTMPose 数值 smoke：检查 17 点位置、置信度、逆仿射以及越界裁剪；再执行正常生命周期、100 轮实际推理 destroy→recreate、模型 init fault injection和 release 幂等测试。
3. 建立可复现回放集比较迁移前后 Wrong Follow、ID Switch、Safe Hold、重捕延迟、Pose calls/frame、p50/p95/max frame time。重点覆盖 CLEAR 单人、相似人交叉、OCCLUDED、RECOVERING 二选一、半身/近场、HEAD_ONLY/FACE_ONLY。
4. 用 `[POSE_REQ]`、`[POSE_RUN]`、`[POSE_PAIR]`、`[POSE_COMMIT]`、`[POSE_FRAME]` 和 `[OBS_CLOCK]` 验证普通 CLEAR=0、refresh=1、ambiguity<=2、失败也不越预算、未确认候选不 commit；同时对齐原有 `MEASUREMENT/FEATURE_GATE/MATCH/OUTPUT` 判断 ID 安全。
5. 明确 PTZ 产品接口：只用 box、改用 aim point、构图目标是否为中心，以及是否需要公开 ego/GMC 开关。
6. 重新生成 `tracking_pipeline.svg/.pdf`，或显著标记为历史图；清理 `parameter_tuning_guide.md` 的失效参数。
7. 先用诊断构建回放 B/C secondary 交叉窗口，并人工标注每帧 physical secondary。分别统计：physical ID switch、relative history contamination、污染是否通过 `[RELATIVE_ESTIMATE]`、是否生成错误 `[EMERGE]`、是否实际造成额外 HOLD 或重捕延迟。若换人发生但 estimator 输出 `source=none` 且无行为影响，不据此直接启用 validator。

## 新 session 建议阅读顺序

1. `AGENTS.md`
2. 本文件中的“已知问题与风险”和“推荐下一步”
3. `PROJECT_CONTEXT.md` 的坐标、模型和安全原则
4. `ARCHITECTURE.md` 的目标子系统
5. `LightTracker.h`：当前常量、状态和结构
6. `LightTracker.cpp`：`update()`、`matchPersonFaces()`、`assign_cascade()`、`match_main_target_unified()`、三个 continuity 函数
7. 按任务深入：`KalmanBoxTracker.cpp`、模型 wrapper、`c_api/fx_tracker.cpp`

历史 Predictor、Gait、Association 源码已删除；当前行为以 `LightTracker`、Kalman 和各实际模型 wrapper 为准。

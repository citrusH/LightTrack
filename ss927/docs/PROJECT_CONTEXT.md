# Project Context

## 项目定位

本仓库实现一个部署在 PTZ 摄像机侧的单主目标行人跟踪库。上层先框选一名主目标，库在后续 BGR 帧中返回主目标的画面内目标框；下游云台控制器据此调整构图。跟踪器只产生观测结果，不直接发送 PTZ 指令。入口和输出边界分别见 `c_api/fx_tracker.cpp::fx_tracker_run()` 与 `LightTracker.cpp::update()`。

核心目标不是“每帧必须有框”，而是在遮挡、多人交错、身体漏检和相机运动下尽量保持身份。错误跟随陌生人的代价高于短时不输出，因此无可信 BODY/HEAD/FACE 观测时返回无主目标，云台应保持。

## 平台与技术栈

- C++17，共享库目标名为 `libfxtracker.so`；新集成通过纯 C99 API 使用。
- Huawei Ascend ACL/OM 模型运行时，设备号在各模型配置中固定为 0。
- OpenCV 4：`cv::Mat`、图像预处理、Kalman 数学、光流和 RANSAC 等。
- 仓库构建文档明确面向设备侧/aarch64；具体 Ascend/HiSilicon SoC、SDK/驱动版本、输入分辨率和实际 FPS：**UNKNOWN**。
- 仓库没有构建系统、本地可执行程序、自动化测试或回放数据集；设备交叉编译命令见 `c_api/README.md`。

## 长期模块边界

仓库布局保持扁平：根目录存放 C++ 核心与模型 wrapper，`c_api/` 存放维护中的 C ABI、示例和设备构建说明，`docs/` 存放架构/审计/调参资料，`.claude/` 只有本地 agent 设置。没有 `src/`、`tests/`、模型资产目录或构建脚本。

- `LightTracker.{h,cpp}`：主 pipeline、状态机、身份融合、恢复、输出稳定和模型预算；权威入口是 `LightTracker::update()`。
- `KalmanBoxTracker.*` / `KalmanFilter.*`：人体与头部的 6 维状态 `[cx,cy,w,h,vx,vy]`，含数值防护、快照恢复和长盲重建。
- `Detector.*`：一次检测同时输出 face/body/head。
- `PersonReID.*`：人体外观特征，L2 归一化后使用余弦相似度。
- `PoseEstimator.*`：17 点姿态、OKS、身体比例、可见度和肩部连续性。
- `Facekps.*` + `FaceRecognition.*` + `FaceRecognitionSystem.*`：人脸关键点、对齐、模板注册和身份验证。
- `ModelProcess.*`：ACL 模型加载、输入输出 dataset、执行和动态 batch 基础设施。
- `AclRuntime.*`：本库唯一的 ACL global/device/context RAII owner；模型 wrapper
  只借用其 current context，并只管理自身模型、dataset 和 buffer。
- `c_api/`：维护中的 C ABI；根目录 `Track.*` 与 `fx_wrapper_track.*` 仅为 legacy 兼容层。
- 历史 `Association.*`、`MainTargetPredictor.*`、`GaitRecognition.*` 未接入热路径，现已从维护源码删除。

## 关键数据与坐标约定

内部 tracking/association 矩阵统一使用像素坐标 `xyxy = [x1,y1,x2,y2]`。常见矩阵格式：

- 检测：`[x1,y1,x2,y2,score]`
- 邻近候选：`[x1,y1,x2,y2,source,index]`
- 输出：`[x1,y1,x2,y2,id]`

一个危险但既定的兼容约定是：检测和主目标路径中的部分 `cv::Rect` 把 `width/height` 字段当作 `x2/y2`，不是宽高。只有构造裁剪 ROI 时才显式转换为真实 `xywh`。`PoseResult::box` 是例外，使用 OpenCV 的真实 `xywh`。C API 的 `FxRect` 输入仍是 `xyxy`，`FxPersonInfo` 输出才转换为 `x,y,w,h`。不得在未确认语义时直接使用 `cv::Rect::area()`、`br()` 或与普通 Rect 混算。

主要结构包括 `ObjDetInfo` / `PersonWithFace`（`types.h`）、`DetectionGroups` / `TrackerInfo` / `ProximityInfo` / `MainMatchResult`（`LightTracker.h`）、`BodyProportionDescriptor`（`PoseEstimator.h`）、`Verification_Result`（`FaceRecognitionSystem.h`）和 `FxTrackResult`（`c_api/fx_tracker.h`）。准确字段以这些头文件为准。

## 模型与运行时资产

`LightTracker::init()` 从硬编码目录 `/oem/model` 读取五个模型，再通过 `ModelProcess::LoadModelFromMem()` 加载：

| 文件 | 职责 |
|---|---|
| `yolox_416_face_body_head.om` | YOLOX face/body/head 联合检测；当前代码按类别 `0/1/2` 解释 |
| `mobilev2_EmbeddingHead_reid_v1_GeMP_pre.om` | 人体 ReID |
| `rtmpose-t.om` | 候选 BODY 的 17 点 RTMPose-T（192×256、双 SimCC 输出） |
| `faceKps_v7_01.om` | 106 点人脸关键点 |
| `faceReco_v2.om` | 对齐人脸特征与模板验证 |

模型版本、校验和、训练数据与量化方式：**UNKNOWN**。当前 YOLOX 检测契约固定为 416×416 BGR HWC U8/AIPP 输入、左上对齐 resize 与右/下 114 padding、单输出 `[1,3549,8]`，8 维为 `cx/cy/w/h/objectness/3-class score`；decode 按 stride 8/16/32、`score=objectness*class_score` 和分类别 NMS 执行。更换 detector OM 必须同时核对输入通道/AIPP、输出布局、是否已 sigmoid 以及类别顺序。

## 核心设计原则

1. 初始指定框建立不可变/慢变 ReID anchor、当前 embedding、人体 KF、头部先验、姿态比例，并在质量允许时注册人脸。
2. 外观不是单阈值决策：ReID、可见度分带 anchor、IoU/中心连续、头部、姿态、肩部、遮挡者排斥和已知他人负证据共同参与。当前融合与最终门控集中在 `LightTracker::match_main_target_unified()`。
3. `OcclusionState` 与 `VisibilityState` 正交：前者描述是否发生人与人遮挡，后者描述主目标身体露出程度。
4. 身体消失时，严格接受的真实头或已验证人脸可以按学习几何重构人体框；重构框先与画面求交，再以较大测量噪声校正人体 KF。
5. 长盲或拥挤时，人体和原始人脸池采用有界预算、跨帧物理假设与防饥饿调度；距离用于排序，不作为全局候选的永久资格门。该设计不能保证短暂候选一定在有限帧内完成识别。
6. Prediction 与 measurement 严格隔离：BODY/HEAD/FACE 接受后控制权重为 1；
   主目标 measurement 短时不可可靠确认时，可从最近有界的安全真实 BODY/HEAD/FACE
   人体框 Motion History 一次性估计运动并冻结有限 short prediction；历史是速度主依据，
   BODY KF 只做方向 corroboration/强冲突检查。prediction 不得更新身份、Motion History、
   真实观测时钟或 KF measurement；超过安全上限后不得重用旧 anchor 重启。
7. 时间相关安全门优先使用单调墙钟毫秒；帧数只保留为短证据计数或候选轮转寿命。

## PTZ 闭环与画面运动约束

跟踪器输出画面内目标框/瞄准点，PTZ 控制器在下游闭环执行；两者之间不是“输出一帧，
画面立即重新居中”的刚性坐标变换。已知控制规律是：目标越靠近画面边缘，云台速度通常越高；
越接近构图目标，速度越低；并且存在启动延迟和死区。因此持续行走的目标可稳定停留在构图点
左侧或右侧一段距离，方向取决于其运动方向和控制响应。这是正常的闭环稳态偏差，不能单独判为
KF 或检测滞后。目标停止后，云台会继续把它渐近带回构图目标。

构图目标通常是画面中心，但产品可以配置左/右等非中心位置。当前维护中的 tracker 没有获得该
控制目标、云台实际速度或已执行位移的输入；因此任何 image-space 运动都可能混入人运动和未观测的
PTZ 自运动。没有真实控制反馈/标定前，不能把预测框、输出框或“离画面中心的偏差”当成精确的
相机坐标补偿量。PRED 仍只能作为无 measurement 时的有限输出，不能刷新 KF measurement、身份特征、
真实观测时钟或其他依赖真实画面观测的状态。

## 性能与安全约束

模型调用是主要耗时。长期约束是：普通帧只做一次联合 Detector；ReID、Pose 和人脸模型由状态感知预算调度，同一检测脸或同一 Pose 候选在一帧内缓存去重。Pose 是候选级 late-stage evidence，不是基础评分必需输入。具体预算与阈值属于当前实现状态，统一记录在 `CURRENT_STATE.md`，不在本长期文档固化。任何预算都只是调用数上限，不是设备时延保证。

长期安全规则：

- 不用弱匹配更新 anchor、模板或搜索中心。
- 人脸模板替换必须先成功提取，再原子替换；优质模板冻结。
- 部件重构框不能被当作真实完整人体去快速污染 ReID。
- Kalman 输入和状态必须有限且满足 `x2>x1, y2>y1`；失败时恢复快照或以强观测重建。
- 同一句柄不可并发调用；任何 C++ 异常都必须在 C ABI 内转为错误码。

## 已确定的长期决策

- 新集成只使用 `c_api/fx_tracker.h`，不扩展 legacy wrapper。
- 历史 `MainTargetPredictor` coast 输出保持停用，其源码已删除。主目标 measurement 短时不可
  可靠确认时，允许从最近有界的安全真实 BODY/HEAD/FACE 人体框历史一次性估计速度，
  并在 PRED 启动时冻结 anchor、quality、速度和安全边界；位置按真实 anchor timestamp 外推，
  400 ms 只约束真正 PRED 持续时间。该语义同时覆盖静态遮挡和人物遮挡，预测不能刷新真实观测时钟、
  Motion History、身份特征、KF measurement 或 lost/recovery 状态，安全窗口结束后旧 anchor 不得重启。
- Gait 模块未接入跟踪决策，相关未使用源码已删除。
- GMC 是可选的坐标补偿能力；自运动前馈即使启用也只能移动软搜索参考，不能直接改写 KF 状态。当前默认开关归 `CURRENT_STATE.md` 管理。
- 匹配诊断必须使用有界、缓冲的文件 trace，不能把逐帧日志放回 UART 热路径。当前宏、路径和 flush 策略见 `ARCHITECTURE.md`。

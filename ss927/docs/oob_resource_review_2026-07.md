# OOB / 资源调度 稳健性复审（2026-07-06）

范围：全量重读 `LightTracker.{h,cpp}`、`Association.cpp`、`MainTargetPredictor.cpp`、
`KalmanBoxTracker.cpp`、`KalmanFilter.cpp`、五个模型组件（`Detector` / `PersonReID` /
`PoseEstimator` / `FaceRecognition` / `Facekps`）、`ModelProcess.cpp`、`utils.cpp`、
`FaceRecognitionSystem.cpp`、C API 桥接层 `c_api/fx_tracker.cpp`。
目标：越界读写、间歇性系统/资源调度错误、未初始化读、能让算法失效的缺陷。

> 说明：本机无 OpenCV4 / Ascend ACL，无法编译验证；以下为静态阅读结论。

## 结论摘要

**逐帧跟踪主路径（LightTracker 及其最近新增特性）未发现活跃越界。** 检测/轨迹矩阵的
每一处 `.row(idx)` / `.at()` 都有 `idx>=0 && idx<rows` 或循环上界护栏；空 `cv::Mat`、
空 `vector`、空画廊、`candidates`/`poses`/`trackers` 下标访问均已判空或受循环边界约束。
注释中记述的 A2/A6/A10/A11/A12 崩溃护栏（幽灵框零初始化、`-1` 哨兵、空关键点判空、
检测输出按真实 `len` 封顶、空特征判空）均确实到位。最近未编译验证的特性（隔离轨迹 B3、
框补全 B8、肩部/朝向门、暂定提交闸、多目标外观排除、非主轨迹维护）下标全部受控。

**但在模型封装组件里发现两个真实的 ACL 资源生命周期缺陷，正好落在用户关注的
"间歇性/资源调度错误"类别。** 它们不是越界，而是把"可恢复的瞬时设备错误"升级成
"持久性设备内存损坏 + 退出期重复 free/finalize"。

---

## 发现 1（高）：瞬时设备错误路径 free 了"整生命周期"的 `picDevBuffer_` 却不置空 → 后续帧 use-after-free（设备端）+ 退出期 double-free

`picDevBuffer_` 是 **init 时一次性分配、供对象整个生命周期复用** 的设备输入缓冲。
但若干 **逐帧错误路径** 在一次瞬时失败时 `aclrtFree(picDevBuffer_)` 且不置 `nullptr`：

- `Detector.cpp:193` — `inference()` 内 `Execute()` 失败 → `aclrtFree(picDevBuffer_)`（不置空）。
- `Detector.cpp:180` — `preProcess()` 内 `aclrtMemcpy` 失败 → 同上。
- `Facekps.cpp:310` — `preProcess()` memcpy 失败 → 同上。
- `FaceRecognition.cpp:361` — `preProcess()` memcpy 失败 → 同上。

失败链条（以 Detector 为例，`LightTracker::update` 第 866 行忽略 `detector.run` 返回值）：

1. 某帧 NPU `aclmdlExecute` 或 `aclrtMemcpy` 瞬时失败（Ascend 在负载/热/调度争用下会间歇发生 —— 即"资源调度错误"）。
2. 错误路径 `aclrtFree(picDevBuffer_)`，指针悬空，`run` 返回 −1（被上层忽略）。
3. **下一帧** `preProcess` → `MemcpyFileToDeviceBuffer(dst.data, picDevBuffer_, …)` 向 **已释放的设备地址** 写入：
   - 若该地址已被复用（例如另一模型的输入缓冲）→ 静默污染他模型输入 → 乱码推理 → 跟踪"莫名其妙地错/飘"，极难定位；
   - 否则 ACL 报无效指针错 → 每帧持续失败。
4. 析构（`~Detector` 第 17 行）再次 `aclrtFree(picDevBuffer_)` → **double-free**。

影响：一次**可恢复的瞬时**设备错误被永久化为设备内存损坏，恰好绕过全代码库精心构建的
"宁可漏配也不崩溃"优雅降级。这是最可能匹配"间歇触发系统错误、影响算法正确工作"的机制。

正确范式（可参照）：`CPersonReID::preProcess` 与 `PoseEstimator::preProcess` 在 memcpy 失败时
**不** free 持久缓冲；两者的 `DestroyResource` free 后 **立即置 `nullptr`**。

修复建议：
- 逐帧 `preProcess/inference` 错误路径 **一律不要** `aclrtFree(picDevBuffer_)`（它属于对象生命周期，非本帧资源）；仅返回错误码交由上层降级。
- 所有确实要释放它的地方（析构 / DestroyResource）在 free 后 **立即 `picDevBuffer_ = nullptr`**，杜绝 double-free。
- 建议上层 `LightTracker::update` 至少记录 `detector.run` 的失败（当前完全忽略返回值）。

---

## 发现 2（中）：退出/重建期 ACL 生命周期错乱 —— 多次 `aclFinalize()`/`aclrtResetDevice()` + Detector 上下文/流泄漏

- ACL 的 `aclInit`/`aclrtSetDevice`/`aclrtCreateContext`/`aclrtCreateStream` **仅** 由
  `Detector::InitResource`（`Detector.cpp:29`）执行一次；`PersonReID`/`FaceRecognition`/`Facekps`
  的 `InitResource` 均被注释掉（`context_`/`stream_` 恒为 `nullptr`）。设计意图正确（不重复 init）。
- 但三者的析构 → `DestroyResource()` 在 `bInit_/initialized_==true` 时 **无条件** 调用
  `aclrtResetDevice(deviceId_)` + `aclFinalize()`（`PersonReID.cpp:372/377`、
  `FaceRecognition.cpp:480/485`、`Facekps.cpp:415/420`）—— 尽管它们从未 init 过 ACL。
- 而真正持有 ACL 的 `Detector`，其析构（`Detector.cpp:12`）**只** free 缓冲、
  **从不** 调用 `DestroyResource` → 已创建的 `context_`/`stream_` **泄漏**、且从不 `aclFinalize`。

`LightTracker` 成员逆序析构：`pose_estimator → detector → emb_model → face_recognizer → detector_fk`。
于是：
1. `emb_model`（PersonReID）先 `aclFinalize()` —— ACL 运行时在此被拆除；
2. 随后 `face_recognizer`(CFaceRecognition)、`detector_fk` 的 `UnloadModel`/`aclrtFree` 在
   **已 finalize 的运行时** 上执行 → 报错，部分 ACL 版本此处 **段错误**；
3. 共 3× `aclFinalize` + 3× `aclrtResetDevice`。

影响：单进程"创建一次、常驻"时多为退出期错误日志/退出崩溃（若看门狗把异常退出视作故障即触发整机重启，
正是别处竭力避免的循环）；**任何 destroy→recreate（重初始化）流程会真正踩到 finalize 后的 ACL** → 不稳定。

修复建议：让 **唯一一个** 组件（如一个显式的 AclEnv/单例，或就用 Detector）负责
`aclInit/SetDevice/Context/Stream + Finalize` 的成对生命周期；其余组件的 `DestroyResource`
**不得** 调用 `aclFinalize`/`aclrtResetDevice`。并让 Detector 在析构时真正销毁其 `context_`/`stream_`。

---

## 次要 / 观察（不阻塞）

- **init 期错误路径 double-free**：`Detector.cpp:73/79`、`Facekps.cpp:150/157`、
  `FaceRecognition.cpp:181/188/199`、`PoseEstimator.cpp:448/455` 在 init 失败时 free 了
  `picDevBuffer_` 但不置空；若 init 部分成功后失败，析构再次 free → double-free。init 失败
  本就中止启动，危害小于发现 1，但同属"free 后不置空"通病，建议一并规范。
- **`GaussianSmoothBox::smooth`（utils.cpp:630）** 有潜在核索引下溢（`kernelOffset =
  kernel.size() - currentSize`），但该类**全代码库从未实例化**（`LightTracker` 里 `smoother`
  成员已注释），且默认 `windowSize==kernel.size()==7` 时 offset≥0 —— 死代码，无风险。
- **`FaceRecognitionSystem::update_primary_face_with_adaptive_strategy`** 首行即用
  `faces_embedding_[0]` 而不判空；但其唯一调用点（recognition() 第 163 行）已被注释 → 死路径，
  当前无触发。若日后启用需先判空。
- **`Facekps::postProcess`** 只写 `kp.x/kp.y` 不写 `confidence`；下游 `get_kps10` 仅取 x/y，
  故无未初始化读；若将来有代码读该 confidence 需注意。
- **每帧无 ACL 资源泄漏**：`picDevBuffer_`/输入输出 dataset 均 init 一次性分配、逐帧复用，
  `Execute`/`OutputModelResult` 不做每帧 `aclrtMalloc` —— 运行期不会因资源累积而逐渐劣化。
  （即：发现 1 之外，没有"跑久了必崩"的渐进式资源耗尽。）

---

## 一句话给现场

逐帧匹配逻辑本身没有越界；真正会"间歇性触发系统/资源错误并让算法失效"的是
**发现 1**——模型组件在一次瞬时 NPU/memcpy 失败时错误地释放了整生命周期的设备输入缓冲
且不置空，使后续每帧写入已释放设备内存。优先修这一条。

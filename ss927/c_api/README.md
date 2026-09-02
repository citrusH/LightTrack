# c_api — 跟踪器完整 C 语言接口层

本目录把整个 PTZ 单目标跟踪器封装为**纯 C 接口**：应用层可以 100% 用 C 编写，
通过 `fx_tracker.h` 调用；C++ 核心（LightTracker 及全部调优行为）编译进
`libfxtracker.so`，**行为与 C++ 直接调用完全一致**（同一份编译代码，零算法改动）。

> 为什么不是"把核心翻译成 C"：核心算法依赖 OpenCV 4 的 C++-only API
> （`cv::Mat` 代数 / `calcOpticalFlowPyrLK` / `estimateAffinePartial2D`-RANSAC 等，
> OpenCV 4 已移除 C API）。手写 C 重新实现这些算法必然产生**不同的数值行为**，
> 违背"效果完全一致"的要求；而 C 接口层 + C++ 核心库是工业界标准做法，
> 效果逐字节一致。

## 文件

| 文件 | 说明 |
|---|---|
| `fx_tracker.h`   | 纯 C99 头文件（应用层唯一需要包含的文件） |
| `fx_tracker.cpp` | 桥接实现（以 C++ 编译，导出 `extern "C"` 符号） |
| `example_app.c`  | 纯 C 示例（`gcc -std=c99` 可编译） |

## 相对旧接口（`fx_wrapper_track.h`）的修复

旧接口保留未动，二者可并存；新接口修复了：
1. 旧头文件里 `unsigned short DEBUG_LOG = 0;` 是变量**定义**——两个 C 文件
   同时包含即重复定义链接错误。
2. 旧 `track_run` 无条件解引用 `input->mainTarget`（注释却写"可选"）——传
   NULL 直接崩溃。新接口 `main_target == NULL` 表示正常跟踪帧。
3. 旧接口 `out_result->count` 写入全部行数，但 `infos[]` 只填了前 5 个——
   按 `count` 遍历会读未初始化内存。新接口 `count`=实际填充数、
   `total_count`=全部行数。
4. 旧接口无销毁函数——句柄泄漏。新增 `fx_tracker_destroy`。
5. 新增：`score` 字段在主目标行携带控制权重（1.0=严格接受的身体/头/脸真实观测，
   `(0,1)`=稳定 BODY 后的有限短时 Kalman prediction；其余无主目标行），云台控制器可按其加权；`fx_tracker_reset` /
   `fx_tracker_get_reset_flag` 暴露核心重置状态。

## 构建（设备侧 / aarch64 交叉）

```bash
# 1) 把 C++ 核心 + 桥接编译成共享库（注意 -std=c++17，核心用了 std::optional）
g++ -std=c++17 -O2 -fPIC -shared \
    ../AclRuntime.cpp ../LightTracker.cpp ../KalmanBoxTracker.cpp ../KalmanFilter.cpp \
    ../PoseEstimator.cpp ../Detector.cpp \
    ../PersonReID.cpp ../FaceRecognition.cpp ../FaceRecognitionSystem.cpp \
    ../Facekps.cpp ../ModelProcess.cpp ../utils.cpp \
    fx_tracker.cpp \
    -I.. -I${OPENCV_INC} -I${ASCEND_ACL_INC} \
    -L${OPENCV_LIB} -lopencv_core -lopencv_imgproc -lopencv_video \
    -lopencv_calib3d -lopencv_features2d \
    -L${ASCEND_ACL_LIB} -lascendcl \
    -o libfxtracker.so

# 2) 纯 C 应用只需 gcc + 头文件 + 链接该库（链接器自动带上 C++ 运行时依赖）
gcc -std=c99 -O2 example_app.c -I. -L. -lfxtracker -o demo
```

生命周期回归程序会在每轮实际执行注册帧和普通推理帧，再销毁并重建，共 100 轮：

```bash
gcc -std=c99 -O2 lifecycle_smoke.c -I. -L. -lfxtracker -o lifecycle_smoke
LD_LIBRARY_PATH=. ./lifecycle_smoke
```

运行时同时观察 ACL 错误日志和设备内存；每轮应各出现一次 `[ACL] init` 与
`[ACL] finalize`，且全部模型 `[... release]` 日志必须位于 finalize 之前。

初始化回滚与 Detector 错误恢复使用默认关闭的测试钩子。共享库构建时临时加入
`-DFX_TRACKER_LIFECYCLE_TEST_HOOKS=1`，再执行：

```bash
gcc -std=c99 -O2 lifecycle_fault_smoke.c -I. -L. -lfxtracker \
    -o lifecycle_fault_smoke
LD_LIBRARY_PATH=. ./lifecycle_fault_smoke
```

该程序依次注入 FaceKps、FaceReco、ReID、Detector、Pose 初始化失败，并在同一
句柄上取消故障后重新初始化；随后分别注入 Detector memcpy/Execute 失败，确认
返回 `FX_ERR_MODEL_RUN`、输出为空且取消故障后的下一次注册推理仍成功。正式发布
构建不要定义 `FX_TRACKER_LIFECYCLE_TEST_HOOKS`。

说明：
- 不要把 `../Track.cpp`、`../fx_wrapper_track.cpp` 一起编入，除非旧 App
  仍需要旧接口（可并存，符号不冲突）。
- ACL Runtime 由 `AclRuntime` 在库内独占管理；当前只允许一个已初始化句柄，
  但支持完整销毁后重新 create/init/run/destroy。ACL global runtime 只在进程
  首次初始化，随后保持到进程退出；宿主进程不得预先初始化 ACL。
- OpenCV 模块按你设备上的实际打包调整（有的发行版是单一 `-lopencv_world`）。
- 模型路径仍是核心内写死的 `/oem/model/*.om`（与现状一致）。

## 使用要点

- **单线程**：同一句柄的所有调用必须串行（与 C++ 核心一致）。
- **单 Runtime**：同一进程同时只允许一个成功初始化的句柄；完整 destroy 后可重新
  create/init/run/destroy，后续实例复用已初始化的 ACL global runtime。宿主不要预先
  调用 ACL 初始化或设备设置，也不要在 destroy/recreate 之间 `dlclose` 后重新加载库。
- **输入**：BGR 8UC3 连续内存；调用期间零拷贝包装，返回后缓冲即可复用。
- **注册主目标**：`main_target` 非 NULL 且面积>0 的那一帧执行注册/重置；
  之后传 NULL 正常跟踪。
- **输出**：`id == FX_MAIN_ID(1)` 是主目标真实观测行；本帧没有严格接受的
  身体/头/脸时不返回该行。`id >= 900` 是未关联的其他人检测，仅供展示。
- 错误码见 `fx_tracker.h`；`FX_ERR_EXCEPTION` 表示内部异常已被捕获，
  `FX_ERR_MODEL_RUN` 表示本帧模型/设备执行失败且未按“零检测”推进跟踪状态，
  细节看设备 stdout/stderr 日志。

## 匹配 trace 文本文件

诊断构建在共享库编译命令中加入：

```bash
-DFX_TRACKER_MATCH_TRACE=1
```

完整的逐帧匹配时间线默认写入 `/tmp/fx_tracker_match_trace.log`。实现使用
64KB stdio 用户态缓冲，关键事件立即 `fflush`，平稳期每 25 帧 `fflush`，不调用
`fsync`；文件达到 16MB 后停止记录。启动应用时可通过环境变量选择文件：

```bash
FX_TRACKER_MATCH_TRACE_FILE=/tmp/crossing_case.log ./demo
```

第一次 trace 写入会截断同名旧文件，应用重启前应先复制需要保留的日志。
`ModelProfiler` 仍输出到 stdout；若也需要保存，可另行重定向：

```bash
./demo > /tmp/model_profiler.log 2>&1
```

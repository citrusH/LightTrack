


## 1.0 （基础版本）
### 谁在前面就跟谁

## 1.2 (h test puas )

## 1.3 (h 2026.1.30)



## v1.0 (2026.2.2 , 在研发部测试跟踪效果还行， 特点是偶尔会上下摆动)
## v1.1 (2026.2.3 , 研发部测试效果好， 大会议室3人测试效果好， 5人以上4米距离会出现丢失， 不放大画面可以显著减少上下摆动情况)

## v1.2 (2026.2.4 , 测试效果比v1.1好， 5人以上丢失率降低)
# v1.4 (2026.2.5 , 人脸识别提升追踪效果明显， 目前最优)
### v1.4.2 (2026.2.7, 添加 debug, 优化 run 的入参)

## SVP ACL 生命周期

`SvpAclRuntime` 是库内唯一的 SVP ACL global/device owner。`LightTracker` 先初始化
runtime，再初始化 Detector、ReID、FaceKps、FaceReco、Pose；销毁或初始化失败时先
逆序释放全部模型/task/dataset/device buffer，最后才 reset device 和 finalize。

当前约束：宿主不得预先初始化 SVP ACL；同一进程同时只允许一个成功初始化的
Tracker；完整 `track_destroy()` 后支持重新 `track_create/init/run/destroy`。模型推理
使用同步 execute，没有显式 context 或 stream。

C API：

- `track_init()` 现在返回错误码；
- 新增 `track_destroy()`，调用方必须在退出或重建前调用；
- `track_run()` 的 NPU/Detector 错误返回 `TRACK_ERR_MODEL_RUN`，不会当成正常零检测；
- `mainTarget == NULL` 表示普通跟踪帧。

设备侧至少验证 100 轮：每轮 create、init、实际运行注册帧和若干普通帧、destroy。
日志中每轮只能各有一次 `[SVP_ACL][I] init` 和 `finalize`，全部模型 unload/free 必须
发生在 finalize 前，并观察 NPU 内存不能持续增长。

## 维护源文件边界

`CMakeLists.txt` 显式列出动态库源文件，不再使用根目录通配符。已移除未接入当前
DV500 调用链的通用 ACL 模型封装、独立主目标预测器、全注释 IVE GMC 实现，以及
空白示例入口。当前 GMC 只保留 OpenCV 软件路径；`lifecycle_smoke.c` 是独立设备测试，
不编入 `libaiDetect.so`。

仓库提供 `lifecycle_smoke.c`。设备侧链接新库后执行：

```bash
aarch64-v01c01-linux-gnu-gcc -std=c99 lifecycle_smoke.c \
  -L. -laiDetect -o lifecycle_smoke
LD_LIBRARY_PATH=. ./lifecycle_smoke 2>&1 | tee /tmp/dv500_lifecycle.log
```

## RTMPose-T 候选级调度

Pose 模型路径为 `/oem/model/rtmpose-t.om`。模型输入为 192×256 RGB U8/AIPP，
前处理按 BODY `xyxy` 框做 MMPose TopdownAffine，输出为两个 SimCC 张量
`17×384` 和 `17×512`。

所有 Pose 推理统一经过候选级 cache，单帧硬预算为 2。CLEAR 普通帧为 0 次，
已确认 winner 到刷新周期时为 1 次；只有 base score 的 Top1/Top2 仍真实歧义时
才成对推理 2 次。OCCLUDED 默认 0 次。单边 Pose 或单边有效特征不参与相对加分，
只有最终通过现有安全门的 BODY winner 才能提交 keypoints/visibility/body shape。

设备 replay 应检查 trace 中 `POSE_RUN` 每帧不超过 2 条，并比较：稳定 CLEAR、
前后交叉、遮挡分离、相似衣着、半身/头脸可见和目标离开重入。重点统计 Pose
调用数、p50/p95/max 帧耗时、ID switch、wrong follow、safe hold 和重捕时延。

/* ════════════════════════════════════════════════════════════════════
 * fx_tracker.h — PTZ 单目标跟踪器 完整 C API（纯 C99 头文件）
 *
 * 设计原则：
 *   - 应用层可 100% 用 C 编写：本头文件不含任何 C++ / OpenCV 类型。
 *   - C++ 核心（LightTracker 全部调优行为）编译进 libfxtracker，
 *     行为与 C++ 直接调用 **完全一致**（同一份编译代码，零算法改动）。
 *   - 异常绝不跨越 C 边界：桥接层全部捕获并转为错误码。
 *
 * 相对旧接口 fx_wrapper_track.h 修复的问题（旧接口保留不动，可并存）：
 *   ① 旧头文件中 `unsigned short DEBUG_LOG = 0;` 是"定义"——被多个 C
 *      编译单元包含时产生重复定义链接错误。本接口无此问题。
 *   ② 旧 track_run 无条件解引用 input->mainTarget（注释却说"可选"）
 *      —— 传 NULL 即崩溃。本接口 main_target 为 NULL 表示"本帧无注册框"。
 *   ③ 旧接口 out_result->count 写入"全部行数"，但 infos[] 只填了
 *      MAX_PERSON_COUNT 个 —— C 端按 count 遍历会读到未初始化内存。
 *      本接口 count = 实际填充数，total_count = 全部检测数。
 *   ④ 旧接口没有 destroy —— 句柄泄漏。本接口提供 fx_tracker_destroy。
 *
 * 输出语义（与 C++ 核心一致）：
 *   - persons[i].id == FX_MAIN_ID(1)  → 主目标框，只来自真实身体/头/脸观测，score=1.0。
 *   - persons[i].id >= 900            → 未关联的其他人检测框（仅展示用）。
 *   - 无真实观测时没有 id==1 的行；外部 PTZ 可按自身 blind 策略处理丢失期。
 *
 * 线程/Runtime 模型：单线程使用，同一句柄的调用不可并发；当前同一进程同时只允许
 * 一个成功初始化的句柄。完整 destroy 后可再次 create/init/run/destroy；ACL global
 * runtime 在进程内只初始化一次并保持到进程退出。
 * ACL 由库内部独占初始化，宿主程序不得预先初始化 ACL runtime/device。
 * ════════════════════════════════════════════════════════════════════ */
#ifndef FX_TRACKER_C_API_H
#define FX_TRACKER_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#define FX_TRACKER_API_VERSION "2.1.0"

/* 输出数组容量（与旧接口 MAX_PERSON_COUNT 保持一致） */
#define FX_MAX_PERSONS 5

/* 主目标输出行的固定 id（核心约定：注册后 main tracker id 恒为 0 → 输出 id=1） */
#define FX_MAIN_ID 1

/* ── 错误码 ──────────────────────────────────────────────────────── */
enum {
    FX_OK              = 0,
    FX_ERR_NULL_HANDLE = -1,   /* 句柄为 NULL */
    FX_ERR_NULL_ARG    = -2,   /* 必填参数为 NULL */
    FX_ERR_BAD_FRAME   = -3,   /* 图像数据/尺寸非法 */
    FX_ERR_INIT_FAILED = -4,   /* 模型初始化失败（看设备日志定位具体模型） */
    FX_ERR_EXCEPTION   = -5,   /* 内部异常（已捕获，未跨越 C 边界） */
    FX_ERR_MODEL_RUN   = -6    /* 模型/设备执行失败；本帧未进入跟踪状态机 */
};

/* ── 数据结构（纯 C，POD）────────────────────────────────────────── */

/* 不透明句柄：内部为 C++ 跟踪器实例 */
typedef struct FxTrackerHandle FxTrackerHandle;

/* 矩形：左上角 (x1,y1) / 右下角 (x2,y2)，像素坐标 */
typedef struct {
    int x1, y1, x2, y2;
} FxRect;

/* 输入帧：BGR 8UC3 连续内存（H×W×3）。
 * 调用期间零拷贝包装，调用返回后即可复用/释放该缓冲。
 *
 * ⚠ 务必零初始化本结构（如 `FxFrame f = {0};`），否则可选的 y_* 字段为随机值，
 *   可能被误当作有效 Y 平面。未提供 Y 平面时把 y_phys 置 0 即可（→ 走 BGR 路径）。 */
typedef struct {
    const unsigned char* bgr_data;
    int width;
    int height;

    /* ── 可选：NV12/YUV420SP 的 Y(亮度)平面，用于 GMC 相机运动补偿的 IVE 零拷贝加速 ──
     * 提供后，GMC 直接把 Y 平面当全分辨率灰度交给 IVE 缩放，省掉 BGR→灰度+缩放+上传
     * 的 CPU 前段（仅 USE_HISI_IVE 构建生效；其余构建保存但忽略）。
     *   y_phys / y_virt : Y 平面物理 / 虚拟地址（需 MMZ/VB 物理连续、16 字节对齐）。
     *   y_stride        : Y 平面行跨距（字节，需 16 对齐，>= width）。
     *   平面尺寸复用上面的 width/height（须与 bgr_data 同尺寸）。
     * y_phys==0 表示未提供 → GMC 回退到 BGR 路径（行为完全不变）。 */
    unsigned long long y_phys;
    unsigned long long y_virt;
    int                y_stride;
} FxFrame;

/* 单个输出目标 */
typedef struct {
    int   id;       /* FX_MAIN_ID=主目标；>=900=其他检测 */
    float score;    /* 主目标行=控制权重(1.0=身体/头/脸真实观测)；其他行恒为 0 */
    int   x, y;     /* 左上角 */
    int   w, h;     /* 宽高（已从核心的 xyxy 转换） */
} FxPersonInfo;

/* 跟踪结果 */
typedef struct {
    FxPersonInfo persons[FX_MAX_PERSONS];
    int count;        /* persons[] 中实际填充的个数（<= FX_MAX_PERSONS） */
    int total_count;  /* 本帧全部输出行数（可能 > count，超出部分被截断） */
} FxTrackResult;

/* ── 生命周期 ───────────────────────────────────────────────────── */

/* 创建跟踪器实例；失败返回 NULL */
FxTrackerHandle* fx_tracker_create(void);

/* 销毁实例（NULL 安全）。释放模型、context 和 device；ACL global runtime 保持到进程退出。 */
void fx_tracker_destroy(FxTrackerHandle* handle);

/* 初始化全部模型（检测/ReID/人脸/关键点/姿态）。
 * 返回 FX_OK 或错误码。必须在首次 fx_tracker_run 前调用且成功。 */
int fx_tracker_init(FxTrackerHandle* handle);

/* ── 每帧调用 ───────────────────────────────────────────────────── */

/* 处理一帧。
 *   frame       : 必填，BGR 图像。
 *   main_target : 可选。非 NULL 且面积>0 → 本帧注册/重置主目标为该框
 *                 （xyxy，与 App 选框一致）；NULL → 正常跟踪帧。
 *   out         : 必填，结果输出（函数内先清零）。
 * 返回 FX_OK 或错误码。模型/设备执行失败返回 FX_ERR_MODEL_RUN，输出保持清零，
 * 且该帧不会被当作正常零检测推进丢失/恢复状态。 */
int fx_tracker_run(FxTrackerHandle* handle,
                   const FxFrame* frame,
                   const FxRect* main_target,
                   FxTrackResult* out);

/* ── 状态访问 / 控制 ───────────────────────────────────────────── */

/* 全量重置（清空主目标/模板/全部状态机；下一帧需重新注册主目标） */
void fx_tracker_reset(FxTrackerHandle* handle);

/* 当前云台控制权重：1.0=真实身体或严格接受的头/脸观测；0.0=无主框输出。
 * 与主目标输出行的 score 字段一致；
 * 句柄为 NULL 返回 0。 */
float fx_tracker_get_coast_weight(FxTrackerHandle* handle);

/* 核心内部发生过重置（如帧间隔>3s 自动重置）→ 返回 1，否则 0。
 * App 可据此提示用户重新框选主目标。读取后用 clear 清除标志。 */
int  fx_tracker_get_reset_flag(FxTrackerHandle* handle);
void fx_tracker_clear_reset_flag(FxTrackerHandle* handle);

/* 库版本字符串（编译期常量） */
const char* fx_tracker_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FX_TRACKER_C_API_H */

/* ════════════════════════════════════════════════════════════════════
 * example_app.c — 纯 C 应用示例（gcc -std=c99 即可编译，无任何 C++ 依赖）
 *
 * 演示完整生命周期：create → init → 注册主目标 → 逐帧跟踪 → destroy。
 * 实际应用中把 read_frame() 换成你的取流代码（VI/VPSS/解码输出 BGR）。
 * ════════════════════════════════════════════════════════════════════ */
#include "fx_tracker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_W 1280
#define FRAME_H 720

/* 占位取流函数：返回 0 表示取到一帧 BGR 写入 buf */
static int read_frame(unsigned char* buf) {
    /* TODO: 替换为真实取流。这里用空帧示意。 */
    memset(buf, 0, (size_t)FRAME_W * FRAME_H * 3);
    return 0;
}

int main(void) {
    printf("fx_tracker version: %s\n", fx_tracker_version());

    FxTrackerHandle* trk = fx_tracker_create();
    if (!trk) { fprintf(stderr, "create failed\n"); return 1; }

    int rc = fx_tracker_init(trk);
    if (rc != FX_OK) {
        fprintf(stderr, "init failed: %d\n", rc);
        fx_tracker_destroy(trk);
        return 1;
    }

    unsigned char* bgr = (unsigned char*)malloc((size_t)FRAME_W * FRAME_H * 3);
    if (!bgr) { fx_tracker_destroy(trk); return 1; }

    /* 零初始化：可选的 y_*（GMC 硬件加速 Y 平面）字段保持为 0 → 走 BGR 路径。
     * 若要启用 IVE 零拷贝加速，再对 frame.y_phys/y_virt/y_stride 赋值即可。 */
    FxFrame frame = {0};
    frame.bgr_data = bgr;
    frame.width    = FRAME_W;
    frame.height   = FRAME_H;

    FxTrackResult result;

    /* ── 第 1 帧：注册主目标（来自 App 的用户框选，xyxy）──────────── */
    if (read_frame(bgr) == 0) {
        FxRect main_box = { 500, 200, 700, 650 };
        rc = fx_tracker_run(trk, &frame, &main_box, &result);
        printf("register rc=%d, outputs=%d\n", rc, result.count);
    }

    /* ── 后续帧：正常跟踪（main_target 传 NULL）──────────────────── */
    for (int f = 0; f < 100; ++f) {
        if (read_frame(bgr) != 0) break;

        rc = fx_tracker_run(trk, &frame, NULL, &result);
        if (rc != FX_OK) { fprintf(stderr, "run rc=%d\n", rc); continue; }

        /* 核心曾自动重置（如断流>3s）→ 提示重新框选 */
        if (fx_tracker_get_reset_flag(trk)) {
            printf("[app] tracker auto-reset, please re-select target\n");
            fx_tracker_clear_reset_flag(trk);
        }

        /* 找主目标行（id == FX_MAIN_ID），score = coast 置信度 */
        int found = 0;
        for (int i = 0; i < result.count; ++i) {
            if (result.persons[i].id == FX_MAIN_ID) {
                printf("frame %d: main box=(%d,%d %dx%d) conf=%.2f\n",
                       f,
                       result.persons[i].x, result.persons[i].y,
                       result.persons[i].w, result.persons[i].h,
                       result.persons[i].score);
                /* TODO: 在此驱动云台，建议用 score 对控制量加权 */
                found = 1;
                break;
            }
        }
        if (!found)
            printf("frame %d: no main target this frame (cold/lost)\n", f);
    }

    free(bgr);
    fx_tracker_destroy(trk);
    return 0;
}

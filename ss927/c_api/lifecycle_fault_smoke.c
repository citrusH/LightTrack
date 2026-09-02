#define _POSIX_C_SOURCE 200809L

#include "fx_tracker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kWidth = 640, kHeight = 480 };

static int check_init_rollback(const char* stage)
{
    FxTrackerHandle* tracker = fx_tracker_create();
    if (!tracker) return 1;

    if (setenv("FX_TRACKER_TEST_FAIL_INIT", stage, 1) != 0) {
        fx_tracker_destroy(tracker);
        return 2;
    }
    int injected_rc = fx_tracker_init(tracker);
    unsetenv("FX_TRACKER_TEST_FAIL_INIT");
    if (injected_rc != FX_ERR_INIT_FAILED) {
        fprintf(stderr, "%s: injected init returned %d\n", stage, injected_rc);
        fx_tracker_destroy(tracker);
        return 3;
    }

    int retry_rc = fx_tracker_init(tracker);
    if (retry_rc != FX_OK) {
        fprintf(stderr, "%s: retry init returned %d\n", stage, retry_rc);
        fx_tracker_destroy(tracker);
        return 4;
    }

    fx_tracker_destroy(tracker);
    printf("init rollback %s ok\n", stage);
    return 0;
}

static int check_detector_recovery(const char* stage,
                                   const FxFrame* frame,
                                   const FxRect* target)
{
    FxTrackerHandle* tracker = fx_tracker_create();
    if (!tracker) return 10;
    if (fx_tracker_init(tracker) != FX_OK) {
        fx_tracker_destroy(tracker);
        return 11;
    }

    if (setenv("FX_TRACKER_TEST_FAIL_DETECTOR", stage, 1) != 0) {
        fx_tracker_destroy(tracker);
        return 12;
    }
    FxTrackResult result;
    int injected_rc = fx_tracker_run(tracker, frame, target, &result);
    unsetenv("FX_TRACKER_TEST_FAIL_DETECTOR");
    if (injected_rc != FX_ERR_MODEL_RUN || result.count != 0) {
        fprintf(stderr, "%s: injected run returned %d, count=%d\n",
                stage, injected_rc, result.count);
        fx_tracker_destroy(tracker);
        return 13;
    }

    int retry_rc = fx_tracker_run(tracker, frame, target, &result);
    if (retry_rc != FX_OK) {
        fprintf(stderr, "%s: retry run returned %d\n", stage, retry_rc);
        fx_tracker_destroy(tracker);
        return 14;
    }

    fx_tracker_destroy(tracker);
    printf("detector recovery %s ok\n", stage);
    return 0;
}

int main(void)
{
    static const char* kInitStages[] = {
        "face_kps", "face_reco", "reid", "detector", "pose"
    };
    for (size_t i = 0; i < sizeof(kInitStages) / sizeof(kInitStages[0]); ++i) {
        int rc = check_init_rollback(kInitStages[i]);
        if (rc != 0) return rc;
    }

    size_t frame_bytes = (size_t)kWidth * kHeight * 3;
    unsigned char* bgr = (unsigned char*)malloc(frame_bytes);
    if (!bgr) return 20;
    memset(bgr, 0, frame_bytes);

    FxFrame frame = {0};
    frame.bgr_data = bgr;
    frame.width = kWidth;
    frame.height = kHeight;
    FxRect target = {220, 80, 420, 440};

    int rc = check_detector_recovery("memcpy", &frame, &target);
    if (rc == 0) rc = check_detector_recovery("execute", &frame, &target);
    free(bgr);
    return rc;
}

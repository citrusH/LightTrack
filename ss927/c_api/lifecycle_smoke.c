#include "fx_tracker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kWidth = 640, kHeight = 480, kRounds = 100, kFramesPerRound = 4 };

int main(void)
{
    size_t frame_bytes = (size_t)kWidth * kHeight * 3;
    unsigned char* bgr = (unsigned char*)malloc(frame_bytes);
    if (!bgr) return 2;
    memset(bgr, 0, frame_bytes);

    FxFrame frame = {0};
    frame.bgr_data = bgr;
    frame.width = kWidth;
    frame.height = kHeight;
    FxRect target = {220, 80, 420, 440};

    for (int round = 0; round < kRounds; ++round) {
        FxTrackerHandle* tracker = fx_tracker_create();
        if (!tracker) {
            fprintf(stderr, "round %d: create failed\n", round);
            free(bgr);
            return 3;
        }

        int rc = fx_tracker_init(tracker);
        if (rc != FX_OK) {
            fprintf(stderr, "round %d: init failed: %d\n", round, rc);
            fx_tracker_destroy(tracker);
            free(bgr);
            return 4;
        }

        for (int f = 0; f < kFramesPerRound; ++f) {
            FxTrackResult result;
            rc = fx_tracker_run(tracker, &frame, f == 0 ? &target : NULL, &result);
            if (rc != FX_OK) {
                fprintf(stderr, "round %d frame %d: run failed: %d\n",
                        round, f, rc);
                fx_tracker_destroy(tracker);
                free(bgr);
                return 5;
            }
        }

        fx_tracker_destroy(tracker);
        printf("lifecycle round %d/%d ok\n", round + 1, kRounds);
    }

    free(bgr);
    return 0;
}

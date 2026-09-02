#include "fx_wrapper_track.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kWidth = 640, kHeight = 480, kRounds = 100, kFrames = 4 };

int main(void)
{
    size_t bytes = (size_t)kWidth * kHeight * 3;
    unsigned char* bgr = (unsigned char*)malloc(bytes);
    if (!bgr) return 2;
    memset(bgr, 0, bytes);

    MainTarget target = {220, 80, 420, 440};
    TrackInput input = {0};
    input.pBGRData = bgr;
    input.width = kWidth;
    input.height = kHeight;
    input.zoomValue = 1.f;

    for (int round = 0; round < kRounds; ++round) {
        TrackHandle* tracker = track_create();
        if (!tracker) {
            free(bgr);
            return 3;
        }
        int rc = track_init(tracker);
        if (rc != TRACK_OK) {
            fprintf(stderr, "round %d init failed: %d\n", round, rc);
            track_destroy(tracker);
            free(bgr);
            return 4;
        }

        for (int frame = 0; frame < kFrames; ++frame) {
            OutResult result;
            input.mainTarget = frame == 0 ? &target : NULL;
            rc = track_run(tracker, &input, &result);
            if (rc != TRACK_OK) {
                fprintf(stderr, "round %d frame %d failed: %d\n",
                        round, frame, rc);
                track_destroy(tracker);
                free(bgr);
                return 5;
            }
        }

        track_destroy(tracker);
        printf("round %d/%d ok\n", round + 1, kRounds);
    }

    free(bgr);
    return 0;
}

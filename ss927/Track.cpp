#include "Track.h"

using namespace cv;
using namespace std;

void Track::tracker_init() {
    tracker.init();
}


std::pair<cv::Mat, int> Track::tracker_run(cv::Mat image, cv::Rect box,
                                           int ptz_blind_phase) {

    // printf(" ========= in tracker_run ========= \n"); // fx 29debug
    cv::Mat track_results;
    int person_cnt = 0;

        /*cv::Rect bbox(1188, 252, 1627, 1037);*/
    tracker.set_ptz_blind_phase(ptz_blind_phase);
    std::tie(track_results, person_cnt) = tracker.update(image, box);

	return { track_results, person_cnt };
}

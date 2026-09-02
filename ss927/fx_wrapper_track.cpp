
#include "fx_wrapper_track.h"
#include "Track.h"

// test push

// 定义不透明结构体，实际是C++类的包装
struct TrackHandle {
    Track* track;
};


extern "C" {
    // TrackHandle* handle = new TrackHandle();
    TrackHandle* track_create(){
		std::cout << "====== 927, v3.0  ======" <<  std::endl;

        TrackHandle* handle = new TrackHandle();
        handle->track = new Track();
        return handle;
    }

    void track_init(TrackHandle* handle) {
        if (handle && handle->track) {
            handle->track->tracker_init();
        }
    }

    void track_release(TrackHandle* handle) {
        if (!handle) return;
        std::cout << "fx_wrapper_track_release called" << std::endl;
        delete handle->track;
        handle->track = nullptr;
        delete handle;
    }

    int track_run(TrackHandle* handle, TrackInput* input, OutResult* out_result){

        if (!handle || !handle->track) {printf("FAIL track_run \n"); return 1; }

        *out_result = (OutResult){0};
        
        // 将原始数据转换为cv::Mat
        // cv::Mat image(416, 416, CV_8UC3, const_cast<unsigned char*>(pBGRData));
        cv::Mat image(input->height, input->width, CV_8UC3, input->pBGRData);

        cv::Mat track_results;
        int person_cnt = 0;

        cv::Rect mainTargetRect(input->mainTarget->x, input->mainTarget->y, input->mainTarget->x2, input->mainTarget->y2);
        // cv::Rect mainTargetRect = cv::Rect(mainTarget->x, mainTarget->y, mainTarget->x2, mainTarget->y2);
        // mainTargetRect.x = mainTarget.x;

        // printf(" ========= in tracker_run[c] ========= \n"); // fx 29debug
        std::tie(track_results, person_cnt) = handle->track->tracker_run(
            image, mainTargetRect, input->ptz_blind_phase);

        int retC = track_results.rows < MAX_PERSON_COUNT ? track_results.rows : MAX_PERSON_COUNT;
        out_result->count = track_results.rows;

        for (int i = 0; i < retC; i++) { // track_results.rows

            int id = int(track_results.at<float>(i, 4));
            int x = int(track_results.at<float>(i, 0));
            int y = int(track_results.at<float>(i, 1));
            int x2 = int(track_results.at<float>(i, 2));
            int y2 = int(track_results.at<float>(i, 3));
            
            out_result->infos[i].id = id;
            // out_result->infos[i].score = 0;
            out_result->infos[i].x =  x;
            out_result->infos[i].y =  y;
            out_result->infos[i].w =  x2-x;
            out_result->infos[i].h =  y2-y;
        }



        return 0;
    }

}

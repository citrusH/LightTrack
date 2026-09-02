
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
		std::cout << "====== 500 V1 ======" <<  std::endl;

        TrackHandle* handle = new TrackHandle();
        handle->track = new Track();
        return handle;
    }

    int track_init(TrackHandle* handle) {
        if (handle && handle->track) {
            try {
                return handle->track->tracker_init() == 0 ? TRACK_OK : TRACK_ERR_INIT;
            } catch (...) {
                return TRACK_ERR_EXCEPTION;
            }
        }
        return TRACK_ERR_NULL;
    }

    void track_destroy(TrackHandle* handle) {
        if (!handle) return;
        delete handle->track;
        handle->track = nullptr;
        delete handle;
    }

    /*int track_run(TrackHandle* handle, unsigned char* pBGRData, OutResult* out_result, MainTarget* mainTarget){

        if (!handle || !handle->track || !input || !out_result) return TRACK_ERR_NULL;

        *out_result = (OutResult){0};
        
        // 将原始数据转换为cv::Mat
        // cv::Mat image(416, 416, CV_8UC3, const_cast<unsigned char*>(pBGRData));
        cv::Mat image(416, 416, CV_8UC3, pBGRData);

        cv::Mat track_results;
        int person_cnt = 0;

        cv::Rect mainTargetRect(mainTarget->x, mainTarget->y, mainTarget->x2, mainTarget->y2);
        // cv::Rect mainTargetRect = cv::Rect(mainTarget->x, mainTarget->y, mainTarget->x2, mainTarget->y2);
        // mainTargetRect.x = mainTarget.x;

        // printf(" ========= in tracker_run[c] ========= \n"); // fx 29debug
        std::tie(track_results, person_cnt) = handle->track->tracker_run(image, mainTargetRect);

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

        // printf("人数:%d 匹配结果个数:%d \n", person_cnt, track_results.rows);
        // for (int i = 0; i < track_results.rows; i++) { // track_results.rows

        //     int id = int(track_results.at<float>(i, 4));
            
        //     int x = int(track_results.at<float>(i, 0));
        //     int y = int(track_results.at<float>(i, 1));
        //     int x2 = int(track_results.at<float>(i, 2));
        //     int y2 = int(track_results.at<float>(i, 3));
            

        //     printf("id=%d, x=%d, y=%d, w=%d, h=%d \n", x, y, x2-x, y2-y);

        // }

        return 0;
    }*/

    int track_run(TrackHandle* handle, TrackInput* input, OutResult* out_result){

        if (!handle || !handle->track || !input || !out_result) return TRACK_ERR_NULL;

        *out_result = (OutResult){0};
        
        // 将原始数据转换为cv::Mat
        // cv::Mat image(416, 416, CV_8UC3, const_cast<unsigned char*>(pBGRData));
        if (!input->pBGRData || input->width <= 0 || input->height <= 0)
            return TRACK_ERR_FRAME;
        cv::Mat image(input->height, input->width, CV_8UC3, input->pBGRData);

        cv::Mat track_results;
        int person_cnt = 0;

        cv::Rect mainTargetRect(0, 0, 0, 0);
        if (input->mainTarget) {
            mainTargetRect = cv::Rect(input->mainTarget->x, input->mainTarget->y,
                                      input->mainTarget->x2, input->mainTarget->y2);
        }
        // cv::Rect mainTargetRect = cv::Rect(mainTarget->x, mainTarget->y, mainTarget->x2, mainTarget->y2);
        // mainTargetRect.x = mainTarget.x;

        // printf(" ========= in tracker_run[c] ========= \n"); // fx 29debug
        try {
            std::tie(track_results, person_cnt) =
                handle->track->tracker_run(image, mainTargetRect, input->zoomValue,
                                            input->ptz_blind_phase);
        } catch (const TrackerRuntimeError& e) {
            std::cerr << "track_run model error: " << e.what() << std::endl;
            return TRACK_ERR_MODEL_RUN;
        } catch (...) {
            return TRACK_ERR_EXCEPTION;
        }

        int retC = track_results.rows < MAX_PERSON_COUNT ? track_results.rows : MAX_PERSON_COUNT;
        out_result->count = retC;

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

        // printf("人数:%d 匹配结果个数:%d \n", person_cnt, track_results.rows);
        // for (int i = 0; i < track_results.rows; i++) { // track_results.rows

        //     int id = int(track_results.at<float>(i, 4));
            
        //     int x = int(track_results.at<float>(i, 0));
        //     int y = int(track_results.at<float>(i, 1));
        //     int x2 = int(track_results.at<float>(i, 2));
        //     int y2 = int(track_results.at<float>(i, 3));
            

        //     printf("id=%d, x=%d, y=%d, w=%d, h=%d \n", x, y, x2-x, y2-y);

        // }

        return TRACK_OK;
    }

}

#pragma once
#ifndef PERSON_INFO_H
#define PERSON_INFO_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <iostream>

namespace VISION_ENGINE {
    typedef enum Result {
        SUCCESS = 0,
        FAILED = 1
    } Result;

    struct ObjDetInfo {
                int label;
                float score;
                cv::Rect box;
                std::vector<cv::Point3f> points;
        };

    struct ObjClsInfo {
                int label;
                float score;
        };

    enum TrackState {
        Tentative = 0,
        Confirmed = 1,
        Occluded = 2,
        Lost = 3,
        Deleted = 4
    };

    struct ExitDetInfo {
        cv::Mat det;
        int det_type;
        int det_idx;

        ExitDetInfo(const cv::Mat& det, int det_type, int det_idx)
            : det(det), det_type(det_type), det_idx(det_idx) {
        }
    };


    struct ExitMatch {
        int trk_idx;
        int det_idx;
        int det_type;
        float distance;
        ExitMatch(int trk_idx, int det_idx, int det_type, float distance)
            : trk_idx(trk_idx), det_idx(det_idx), det_type(det_type), distance(distance) {
        }
    };

    struct FacePose {
        float yaw;
        float pitch;
        float roll;
    };


    struct DetResult {
        cv::Rect_<double> bbox;
        float score;
        int label;
    };

    struct PersonWithFace {
        cv::Rect_<float> person_box;
        float person_score;
        std::vector<cv::Rect> face_box;
        std::vector<float> face_scores;
        // ── 头部关联（新增，label==2）──
        // 每个人体最多关联一个头部框；用于遮挡期的可靠空间判别
        cv::Rect head_box;             // 关联到的头部框（xyxy 语义，与全局一致）
        float    head_score = 0.f;     // 头部置信度
        bool     has_head   = false;   // 是否成功关联到头部
    };
    struct TrackResult {
        int id;
        cv::Rect_<float> bbox;
        float score;
        TrackState state;
        bool is_main;
        cv::Mat appearance_feat;
        FacePose face_pose;
        std::string face_id;
    };


    struct TrackData {
        int id;
        int age;
        int hits;
        int hit_streak;
        int time_since_update;
        int time_since_update_emb;
        cv::Mat last_observation;
        std::map<int, cv::Mat> observations;
        cv::Mat velocity;
        float speed;
        bool is_main;
        bool frozen;
        bool in_interaction;
        int interaction_state;
        cv::Mat emb;
    };


}
#endif

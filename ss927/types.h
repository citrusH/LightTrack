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


}
#endif

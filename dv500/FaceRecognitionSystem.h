#pragma once

#ifndef FACERECOGNITIONSYSTEM_H
#define FACERECOGNITIONSYSTEM_H

#include "FaceRecognition.h"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>
#include <iostream>
#include "types.h"
#include "utils.h"

using namespace VISION_ENGINE;


struct PoseAngles {
    float yaw;      // 水平偏转
    float pitch;    // 垂直俯仰
    float roll;     // 平面旋转
};

struct FaceEntry {
    std::string user_name;
    std::vector<float> feature;
};

struct FaceRecognitionResult {
    float similarity = 0.f;
    std::string user_name = "unknown";
    bool comparison_completed = false;
    std::string reason = "unknown";
};

struct Verification_Result {
    cv::Mat matches_main_one;
    cv::Mat matches_main_second;
    int got_match = 0;
    // 恢复人脸命中：可能是 standalone，也可能绑定在最终未接受/可疑的人体上；
    // 均不能复用 source/index 的 body 语义。调用方只能用 standalone_body_box
    // 输出/引导云台，不应拿重构框刷新 ReID/anchor 模板。
    bool matched_standalone = false;
    int standalone_face_idx = -1;
    float standalone_sim = 0.f;
    // true 表示该脸位于旧头/身体搜索门之外，是全画面恢复候选。
    // 调用方可对这种大位移命中使用更严格的即时提交阈值/连续帧确认。
    bool standalone_global = false;
    cv::Mat standalone_body_box;   // [1,4] xyxy，由独立人脸 + 历史体尺寸重建
};

struct Face_Match {
    cv::Mat matched_one;
    cv::Mat matched_second;
    int conflict_delete = 0;
    bool matched_standalone = false;
    int standalone_face_idx = -1;
    float standalone_sim = 0.f;
    bool standalone_global = false;
    cv::Mat standalone_body_box;
};

class FaceRecognitionSystem {
public:

    FaceRecognitionSystem();
    ~FaceRecognitionSystem();


    int init(FaceRecogConfigDV500 config,
        float threshold = 0.50f,
        float det_thresh = 0.40f);

    void release();


    FaceRecognitionResult recognition(const cv::Mat& image,
        const std::vector<float>& kps, float faceBoxScore);

    std::string register_face(const cv::Mat& image,
        const std::vector<float>& kps,
        const std::string& user_name);

    // 先提取新特征，成功后才原子替换模板；提取失败时保留旧模板。
    std::string replace_face(const cv::Mat& image,
        const std::vector<float>& kps,
        const std::string& user_name);

    bool has_face_template() const {
        return !faces_embedding_.empty();
    }

    void reset() {
        faces_embedding_.clear();
    };

    PoseAngles calculate_face_pose(const std::vector<float>& landmarks);

    std::pair<bool, std::string> is_frontal_face(const PoseAngles& pose_angles,
        const std::map<std::string, float>& thresholds);

    std::pair<float, bool> feature_compare(const std::vector<float>& feat1,
        const std::vector<float>& feat2,
        float threshold = 0.50f);

    void update_primary_face_with_adaptive_strategy(
        const cv::Mat& normalized_face_feature,
        float recognition_confidence, float similarity);

private:
    float safe_asin(float x) {
        x = std::max(-1.0f, std::min(1.0f, x));
        return std::asin(x);
    }

    void normalize_feature(std::vector<float>& feature);

private:
    int high_quality_streak = 0;
    std::unique_ptr<CFaceRecognitionDV500> face_recognizer_;
    std::vector<FaceEntry> faces_embedding_;
    std::string face_db_path_;
    float recognition_threshold_;
    float detection_threshold_;
    cv::Size detection_size_;
    std::map<std::string, float> pose_thresholds_;
};

#endif 

#include "FaceRecognitionSystem.h"
#include <fstream>
#include <algorithm>

// （已删除 extern unsigned char DEBUG_LOG：全文件无使用；且与旧头
//   fx_wrapper_track.h 中的定义类型不符（unsigned short）——一旦被引用，
//   新 API 构建将出现装载期未定义符号（启动即死），旧构建则类型双关 UB。）

using namespace VISION_ENGINE;
//const float M_PI = 3.14159265358979323846;

FaceRecognitionSystem::FaceRecognitionSystem()
    : recognition_threshold_(0.50f)
    , detection_threshold_(0.4f) {

    pose_thresholds_ = {
        {"yaw", 25.0f},
        {"pitch", 20.0f},
        {"roll", 25.0f}
    };
} 

FaceRecognitionSystem::~FaceRecognitionSystem() {
    release();
}

int FaceRecognitionSystem::init(FaceRecogConfig config,
    float threshold,
    float det_thresh) {

    release();
    face_recognizer_ = std::make_unique<CFaceRecognition>();
    int ret = face_recognizer_->init(config);
    if (ret != 0) {
        std::cerr << "Failed to initialize face recognizer" << std::endl;
        release();
        return -1;
    }

    recognition_threshold_ = threshold;
    detection_threshold_ = det_thresh;
    // A12：此前成功路径无 return —— 非 void 函数落底是 UB（aarch64 -O2 下编译器可
    // 省略函数尾声/生成陷阱指令，init 每次启动必经此路径），且返回值为垃圾，
    // 调用方无法判断初始化成败。
    return 0;
}

void FaceRecognitionSystem::release()
{
    faces_embedding_.clear();
    if (face_recognizer_) {
        face_recognizer_->release();
        face_recognizer_.reset();
        INFO_LOG("[FaceRecognitionSystem] release");
    }
    high_quality_streak = 0;
}

void FaceRecognitionSystem::update_primary_face_with_adaptive_strategy(
    const cv::Mat& normalized_face_feature, float recognition_confidence, float similarity) {

    if (normalized_face_feature.empty()) return;

    static constexpr float HIGH_SIM_THRESHOLD = 0.95f;   
    static constexpr float MEDIUM_SIM_THRESHOLD = 0.90f; 
    static constexpr float LOW_SIM_THRESHOLD = 0.85f;    


    if (similarity < LOW_SIM_THRESHOLD) {
        return;
    }

    float alpha;
    float update_weight;  

    if (similarity > HIGH_SIM_THRESHOLD) {
        alpha = 0.97f;
        update_weight = 1.0f; 
        high_quality_streak++;
    }
    else if (similarity > MEDIUM_SIM_THRESHOLD) {
        alpha = 0.985f;
        update_weight = 0.5f;  
        high_quality_streak = 0;  
    }
    else {
        alpha = 0.995f;
        update_weight = 0.2f;
        high_quality_streak = 0;
    }

    if (high_quality_streak > 20) {
        alpha = std::max(0.95f, alpha - 0.01f); // 最多降低0.01
    }

    float confidence_factor = 1.0f;
    if (recognition_confidence > 0) {
        confidence_factor = 0.5f + 0.5f * recognition_confidence;
    }

    float effective_alpha = alpha;
    float effective_weight = update_weight * confidence_factor;

    cv::Mat main_embedding_mat(1, faces_embedding_[0].feature.size(), CV_32F);
    for (size_t i = 0; i < faces_embedding_[0].feature.size(); i++) {
        main_embedding_mat.at<float>(0, i) = faces_embedding_[0].feature[i];
    }

    main_embedding_mat = effective_alpha * main_embedding_mat +
        (1.0f - effective_alpha) * normalized_face_feature * effective_weight;

    cv::normalize(main_embedding_mat, main_embedding_mat, 1.0, 0.0, cv::NORM_L2);

    for (size_t i = 0; i < faces_embedding_[0].feature.size(); i++) {
        faces_embedding_[0].feature[i] = main_embedding_mat.at<float>(0, i);
    }


}

FaceRecognitionResult FaceRecognitionSystem::recognition(const cv::Mat& image,
    const std::vector<float>& kps, float faceBoxScore) {

    FaceRecognitionResult result;

    if (kps.empty()) {
        result.reason = "no_kps";
        return result;
    }

    if (kps.size() != 10) {
        std::cerr << "Invalid landmarks size: " << kps.size() << std::endl;
        result.reason = "invalid_kps";
        return result;
    }

    PoseAngles pose_angles = calculate_face_pose(kps);

    auto result_frontal = is_frontal_face(pose_angles, pose_thresholds_);
    bool is_frontal = result_frontal.first;

    //is_frontal = true;

    if (is_frontal) {
        std::vector<float> embedding;
        FaceInfo face_info;
        face_info.kps = kps;
        // for(int i = 0; i < 10; i++) {
        //     std::cout << "kps[" << i << "] : " << kps[i] << std::endl;
        // }
        // std::cout << "cols : " << image.cols << " rows : " << image.rows << std::endl;
        if (face_recognizer_->get(image, face_info) != 0) {
            std::cout << "Failed to extract face feature" << std::endl;
            result.reason = "feature_failed";
            return result;
        }
        embedding = face_info.embedding;

        if (embedding.empty()) {
            result.reason = "empty_feature";
            return result;
        }

        if (faces_embedding_.empty()) {
            result.reason = "no_template";
            return result;
        }

        //float max_similarity = 0.0f;
        std::string matched_name = "unknown";
        float matched_sim = 0.0f;
        float best_similarity = -1.0f;
        bool compared = false;

        for (auto& entry : faces_embedding_) {
            // 只有维度、数值和范数都有效的模板比较才有资格产生 NEGATIVE。
            // 模板损坏/维度不匹配属于 UNKNOWN，不能被下游当成明确 mismatch。
            if (entry.feature.empty() || entry.feature.size() != embedding.size())
                continue;
            float embedding_norm2 = 0.f;
            float template_norm2 = 0.f;
            bool finite = true;
            for (size_t i = 0; i < embedding.size(); ++i) {
                if (!std::isfinite(embedding[i])
                    || !std::isfinite(entry.feature[i])) {
                    finite = false;
                    break;
                }
                embedding_norm2 += embedding[i] * embedding[i];
                template_norm2 += entry.feature[i] * entry.feature[i];
            }
            if (!finite || embedding_norm2 < 1e-12f || template_norm2 < 1e-12f)
                continue;

            auto compare_result = feature_compare(embedding, entry.feature, recognition_threshold_);
            //float similarity = embedding.dot(entry.feature);
            //bool is_match = similarity > recognition_threshold_;
            float similarity = compare_result.first;
            if (!std::isfinite(similarity)) continue;
            // std::cout << "Face sim : " << similarity << " name : " << entry.user_name << std::endl;
            bool is_match = compare_result.second;
            compared = true;
            best_similarity = std::max(best_similarity, similarity);

            if (is_match && similarity > matched_sim) {
                matched_sim = similarity;
                // cv::Mat embedding_mat(1, embedding.size(), CV_32F);
                // for (size_t i = 0; i < embedding.size(); i++) {
                //     embedding_mat.at<float>(0, i) = embedding[i];
                // }

                //max_similarity = similarity;
                matched_name = entry.user_name;
                // update_primary_face_with_adaptive_strategy(embedding_mat, faceBoxScore, similarity);
            }
        }

        if (!compared) {
            result.reason = "no_comparison";
            return result;
        }

        result.comparison_completed = true;
        result.similarity = std::max(0.f, best_similarity);
        result.user_name = matched_name;
        result.reason = matched_name == "unknown" ? "mismatch" : "match";
    }
    else {
        result.reason = "non_frontal";
    }

    return result;
}

std::string FaceRecognitionSystem::register_face(const cv::Mat& image,
    const std::vector<float>& kps,
    const std::string& user_name) {

    if (kps.size() == 0) {
        return "no face detected";
    }
    if (kps.size() != 10) {
        return "invalid face landmarks";
    }

    std::vector<float> embedding;

    FaceInfo face_info;
    face_info.kps = kps;
    if (face_recognizer_->get(image, face_info) != 0) {
        return "failed to extract face feature";
    }
    embedding = face_info.embedding;
    if (embedding.empty()) {
        return "empty face feature";
    }

//    normalize_feature(embedding);

    faces_embedding_.push_back({ user_name, embedding });

    return "success";
}

std::string FaceRecognitionSystem::replace_face(const cv::Mat& image,
    const std::vector<float>& kps,
    const std::string& user_name) {

    if (kps.empty()) {
        return "no face detected";
    }
    if (kps.size() != 10) {
        return "invalid face landmarks";
    }

    std::vector<float> embedding;
    FaceInfo face_info;
    face_info.kps = kps;
    if (face_recognizer_->get(image, face_info) != 0) {
        return "failed to extract face feature";
    }
    embedding = face_info.embedding;
    if (embedding.empty()) {
        return "empty face feature";
    }

    // 新特征已经完整生成后再替换，避免 reset()+注册失败把可用旧模板清空。
    faces_embedding_.clear();
    faces_embedding_.push_back({ user_name, embedding });
    return "success";
}

PoseAngles FaceRecognitionSystem::calculate_face_pose(const std::vector<float>& landmarks) {
    PoseAngles angles = { 0.0f, 0.0f, 0.0f };

    if (landmarks.size() < 10) {
        return angles;
    }

    cv::Point2f left_eye(landmarks[0], landmarks[1]);
    cv::Point2f right_eye(landmarks[2], landmarks[3]);
    cv::Point2f nose(landmarks[4], landmarks[5]);
    cv::Point2f left_mouth(landmarks[6], landmarks[7]);
    cv::Point2f right_mouth(landmarks[8], landmarks[9]);

    float eyes_distance = cv::norm(right_eye - left_eye);
    if (eyes_distance < 1e-6f) {
        return angles;
    }
    cv::Point2f eyes_center = (left_eye + right_eye) * 0.5f;

    float nose_offset_x = nose.x - eyes_center.x;
    float ratio = nose_offset_x / (eyes_distance * 0.5f);
    ratio = std::max(-1.0f, std::min(1.0f, ratio));
    angles.yaw = safe_asin(ratio) * 180.0f / M_PI;

    cv::Point2f mouth_center = (left_mouth + right_mouth) * 0.5f;
    float face_height = mouth_center.y - eyes_center.y;
    float standard_ratio = 1.1f;
    float current_ratio = face_height / eyes_distance;
    float pitch_ratio = (current_ratio - standard_ratio) / standard_ratio;
    pitch_ratio = std::max(-0.5f, std::min(0.5f, pitch_ratio));
    angles.pitch = safe_asin(pitch_ratio) * 180.0f / M_PI;

    float dy = right_eye.y - left_eye.y;
    float dx = right_eye.x - left_eye.x;
    angles.roll = std::atan2(dy, dx) * 180.0f / M_PI;

    return angles;
}

std::pair<bool, std::string> FaceRecognitionSystem::is_frontal_face(const PoseAngles& pose_angles,
    const std::map<std::string, float>& thresholds) {

    float yaw_abs = std::abs(pose_angles.yaw);
    float pitch_abs = std::abs(pose_angles.pitch);
    float roll_abs = std::abs(pose_angles.roll);
    //  std::cout << "Pose angles - Yaw: " << pose_angles.yaw
    //       << ", Pitch: " << pose_angles.pitch
          //<< ", Roll: " << pose_angles.roll << std::endl;

    float yaw_thresh = thresholds.count("yaw") ? thresholds.at("yaw") : 25.0f;
    float pitch_thresh = thresholds.count("pitch") ? thresholds.at("pitch") : 20.0f;
    float roll_thresh = thresholds.count("roll") ? thresholds.at("roll") : 25.0f;

    if (yaw_abs <= yaw_thresh && pitch_abs <= pitch_thresh && roll_abs <= roll_thresh) {
        return { true, "frontal" };
    }
    else {
        return { false, "side face" };
    }
}

std::pair<float, bool> FaceRecognitionSystem::feature_compare(const std::vector<float>& feat1,
    const std::vector<float>& feat2,
    float threshold) {

    if (feat1.empty() || feat2.empty() || feat1.size() != feat2.size()) {
        return { 0.0f, false };
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < feat1.size(); ++i) {
        dot_product += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 < 1e-12f || norm2 < 1e-12f) {
        return { 0.0f, false };
    }

    float similarity = dot_product / (norm1 * norm2);
    bool is_match = similarity > threshold;

    return { similarity, is_match };
}

void FaceRecognitionSystem::normalize_feature(std::vector<float>& feature) {
    float norm = 0.0f;
    for (float val : feature) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 0.0f) {
        for (float& val : feature) {
            val /= norm;
        }
    }
}

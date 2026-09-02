
#include "FaceRecognition.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include "svp_acl_rt.h"
#include "svp_acl.h"
#include "svp_acl_ext.h"
#include "sample_common_svp.h"
#include "sample_common_svp_npu.h"
#include "sample_common_svp_npu_model.h"
#include "ModelProfiler.h"

using namespace VISION_ENGINE;

#define FR_LOG_I(fmt, ...) printf("[FaceRecogDV500][I] " fmt "\n", ##__VA_ARGS__)
#define FR_LOG_E(fmt, ...) printf("[FaceRecogDV500][E] " fmt "\n", ##__VA_ARGS__)

/* ================================================================
 * fp16 → float32（与模板一致）
 * ================================================================ */
static float fp16_to_f32(uint16_t h)
{
    uint32_t sign     = (h >> 15) & 0x1u;
    uint32_t exp      = (h >> 10) & 0x1fu;
    uint32_t mantissa =  h        & 0x3ffu;
    uint32_t f;
    if (exp == 0u) {
        if (mantissa == 0u) {
            f = sign << 31;
        } else {
            exp = 1u;
            while (!(mantissa & 0x400u)) { mantissa <<= 1; exp--; }
            mantissa &= 0x3ffu;
            f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1fu) {
        f = (sign << 31) | (0xffu << 23) | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

/* ================================================================
 * norm_crop：与原始 FaceRecognition.cpp 完全一致（5点 → 3点仿射）
 * ================================================================ */
cv::Mat CFaceRecognitionDV500::norm_crop(const cv::Mat& img,
                                          const std::vector<float>& landmark,
                                          int image_size)
{
    if (landmark.size() != 10) {
        FR_LOG_E("landmark size must be 10");
        return cv::Mat();
    }

    std::vector<cv::Point2f> srcPoints(5);
    for (int i = 0; i < 5; ++i)
        srcPoints[i] = { landmark[i * 2], landmark[i * 2 + 1] };

    float ratio = 1.f;
    float diff_x = 0.f;

    if (image_size % 112 == 0) { ratio = image_size / 112.f; }
    else if (image_size % 128 == 0) { ratio = image_size / 128.f; diff_x = 8.f * ratio; }
    else { ratio = image_size / 112.f; }

    // ArcFace 标准模板点（112×112）
    const cv::Point2f base_dst[5] = {
        {38.2946f, 51.6963f}, {73.5318f, 51.5014f},
        {56.0252f, 71.7366f}, {41.5493f, 92.3655f},
        {70.7299f, 92.2041f}
    };

    std::vector<cv::Point2f> dstPoints(5);
    for (int i = 0; i < 5; ++i)
        dstPoints[i] = { base_dst[i].x * ratio + diff_x, base_dst[i].y * ratio };

    // 用左眼、右眼、鼻子三点估计仿射
    std::vector<cv::Point2f> srcTri = { srcPoints[0], srcPoints[1], srcPoints[2] };
    std::vector<cv::Point2f> dstTri = { dstPoints[0], dstPoints[1], dstPoints[2] };

    cv::Mat M = cv::getAffineTransform(srcTri, dstTri);

    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(image_size, image_size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return aligned;
}

/* ================================================================
 * 构造 / 析构
 * ================================================================ */
CFaceRecognitionDV500::CFaceRecognitionDV500()
    : inited_(false)
    , net_h_(0)
    , net_w_(0)
    , embedding_dim_(0)
    , out_stride_elem_(0)
    , out_total_elem_(0)
{
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
}

CFaceRecognitionDV500::~CFaceRecognitionDV500()
{
    destroy();
}

/* ================================================================
 * init
 * ================================================================ */
int CFaceRecognitionDV500::init(const FaceRecogConfigDV500& config)
{
    if (inited_) { FR_LOG_E("already inited"); return -1; }
    destroy();
    cfg_ = config;

    // 1. 加载 .om 模型；runtime 已由 LightTracker 初始化
    if (load_model_() != 0) { destroy(); return -1; }
    model_loaded_ = true;

    // 3. 获取模型输入输出信息
    sample_svp_npu_model_info* model_info =
        sample_common_svp_npu_get_model_info(MODEL_IDX);
    if (model_info == nullptr || model_info->model_desc == nullptr) {
        FR_LOG_E("get model info failed");
        destroy(); return -1;
    }
    svp_acl_mdl_desc* desc = model_info->model_desc;

    // 3a. 输入尺寸
    svp_acl_mdl_io_dims in_dims{};
    svp_acl_error acl_ret = svp_acl_mdl_get_input_dims(desc, 0, &in_dims);
    if (acl_ret != SVP_ACL_SUCCESS) {
        FR_LOG_E("get input dims failed 0x%x", acl_ret);
        destroy(); return -1;
    }
    // 假设 NCHW 布局
    net_h_ = (int)in_dims.dims[2];
    net_w_ = (int)in_dims.dims[3];
    if (cfg_.inputSize.width != net_w_ || cfg_.inputSize.height != net_h_) {
        FR_LOG_I("adjust input size from (%d,%d) to (%d,%d)",
                 cfg_.inputSize.width, cfg_.inputSize.height, net_w_, net_h_);
        cfg_.inputSize = cv::Size(net_w_, net_h_);
    }
    FR_LOG_I("model input: N=%d C=%d H=%d W=%d",
             (int)in_dims.dims[0], (int)in_dims.dims[1], net_h_, net_w_);

    // 3b. 输出信息
    svp_acl_mdl_io_dims out_dims{};
    svp_acl_mdl_get_output_dims(desc, 0, &out_dims);
    out_total_elem_ = (int)out_dims.dims[out_dims.dim_count - 1];
    embedding_dim_ = out_total_elem_;
    FR_LOG_I("output embedding dim = %d", embedding_dim_);

    size_t default_stride = svp_acl_mdl_get_output_default_stride(desc, 0);
    size_t elem_bytes = cfg_.output_fp16 ? sizeof(uint16_t) : sizeof(float);
    out_stride_elem_ = (default_stride > 0) ? default_stride / elem_bytes : (size_t)out_total_elem_;
    FR_LOG_I("output stride = %zu bytes, stride_elem = %zu", default_stride, out_stride_elem_);

    svp_acl_data_type out_dtype = svp_acl_mdl_get_output_data_type(desc, 0);
    FR_LOG_I("output dtype = %d (0=fp32,1=fp16)", (int)out_dtype);
    if (out_dtype == SVP_ACL_FLOAT16 && !cfg_.output_fp16) {
        FR_LOG_I("auto-enable fp16 conversion");
        cfg_.output_fp16 = true;
    }

    // 4. 配置 task
    task_info_.cfg.max_batch_num     = 1;
    task_info_.cfg.dynamic_batch_num = 1;
    task_info_.cfg.total_t           = 0;
    task_info_.cfg.is_cached         = TD_TRUE;
    task_info_.cfg.model_idx         = MODEL_IDX;

    // 5. 创建 task 资源
    task_initialized_ = true;
    if (init_task_() != 0) { destroy(); return -1; }

    inited_ = true;
    FR_LOG_I("init ok: %s  size=%dx%d  embed_dim=%d",
             cfg_.model_path.c_str(), net_w_, net_h_, embedding_dim_);
    return 0;
}

/* ================================================================
 * get：输入原始图像 + 5个关键点 → 对齐 → 提特征
 * ================================================================ */
std::vector<float> CFaceRecognitionDV500::get(const cv::Mat& img,
                                              const std::vector<float>& landmarks)
{

    std::vector<float> empty;
    if (!inited_) { FR_LOG_E("not initialized"); return empty; }
    if (img.empty()) { FR_LOG_E("empty image"); return empty; }
    if (landmarks.size() != 10) {
        FR_LOG_E("need 5 keypoints (10 floats)");
        return empty;
    }

    cv::Mat aligned = norm_crop(img, landmarks, cfg_.inputSize.height);
    if (aligned.empty()) {
        FR_LOG_E("norm_crop failed");
        return empty;
    }


    return get_feat(aligned);
}

/* ================================================================
 * get_feat：输入已对齐人脸图像 → 提取特征（L2归一化后）
 * ================================================================ */
std::vector<float> CFaceRecognitionDV500::get_feat(const cv::Mat& aligned_face)
{
    // auto start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::steady_clock::now().time_since_epoch()).count();
    std::vector<float> feat;
    if (!inited_) { FR_LOG_E("not initialized"); return feat; }
    if (aligned_face.empty()) { FR_LOG_E("empty aligned face"); return feat; }


    fxprof::add_call(fxprof::Model::FaceReco);
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceReco, fxprof::Phase::Pre);

        if (preProcess(aligned_face) != 0) {
            FR_LOG_E("preProcess failed");
            return feat;
        }
    }

    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceReco, fxprof::Phase::Infer);
        if (inference() != 0) {
            FR_LOG_E("inference failed");
            return feat;
         }
    }
    
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceReco, fxprof::Phase::Post);
        if (postProcess(feat) != 0) {
            FR_LOG_E("postProcess failed");
            return feat;
        }
    }
    

    return feat;
}

/* ================================================================
 * preProcess：BGR→RGB + 拷贝到 device buffer
 * AIPP 负责归一化（mean/var_reci 已在模型转换时配置）
 * ================================================================ */
int CFaceRecognitionDV500::preProcess(const cv::Mat& src)
{
    cv::Mat dst;
    if (src.size() != cfg_.inputSize)
        cv::resize(src, dst, cfg_.inputSize, 0, 0, cv::INTER_LINEAR);
    else
        dst = src.clone();

    if (cfg_.bgr2rgb)
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    // 取 device input buffer
    td_u8  *dev_ptr = nullptr;
    td_u32  buf_size = 0, stride = 0;
    td_s32  ret = sample_common_svp_npu_get_input_data_buffer_info(
        &task_info_, 0, &dev_ptr, &buf_size, &stride);
    if (ret != TD_SUCCESS) {
        FR_LOG_E("get input buffer info failed 0x%x", ret);
        return -1;
    }

    int H = dst.rows, W = dst.cols;  // = cfg_.inputSize.height/width

    if (cfg_.input_format == InputFormatDV500::RGB_PACKAGE) {
        // HWC 交错格式
        td_u32 data_bytes = (td_u32)(H * W * 3);
        if (data_bytes > buf_size) {
            FR_LOG_E("data %u > buf %u", data_bytes, buf_size);
            return -1;
        }
        if (!dst.isContinuous()) dst = dst.clone();
        errno_t cp_ret = memcpy_s(dev_ptr, buf_size, dst.data, data_bytes);
        if (cp_ret != EOK) {
            FR_LOG_E("memcpy_s failed %d", cp_ret);
            return -1;
        }
        svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
        if (flush_ret != SVP_ACL_SUCCESS) {
            FR_LOG_E("mem flush failed 0x%x", flush_ret);
            return -1;
        }
        td_u32 packed_stride = (td_u32)(W * 3);
        ret = sample_common_svp_npu_update_input_data_buffer_info(
            dev_ptr, data_bytes, packed_stride, 0, &task_info_);
    } else { // RGB_PLANAR
        td_u32 plane_size = (td_u32)(H * W);
        td_u32 data_bytes = plane_size * 3;
        if (data_bytes > buf_size) {
            FR_LOG_E("data %u > buf %u", data_bytes, buf_size);
            return -1;
        }
        std::vector<cv::Mat> channels(3);
        cv::split(dst, channels); // 顺序 R,G,B（已 bgr2rgb）
        td_u8* ptr = dev_ptr;
        for (int c = 0; c < 3; ++c) {
            cv::Mat ch = channels[c].isContinuous() ? channels[c] : channels[c].clone();
            errno_t cp_ret = memcpy_s(ptr, buf_size - (ptr - dev_ptr),
                                      ch.data, plane_size);
            if (cp_ret != EOK) {
                FR_LOG_E("memcpy_s ch%d failed %d", c, cp_ret);
                return -1;
            }
            ptr += plane_size;
        }
        svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
        if (flush_ret != SVP_ACL_SUCCESS) {
            FR_LOG_E("mem flush failed 0x%x", flush_ret);
            return -1;
        }
        td_u32 planar_stride = (td_u32)W;
        ret = sample_common_svp_npu_update_input_data_buffer_info(
            dev_ptr, data_bytes, planar_stride, 0, &task_info_);
    }

    if (ret != TD_SUCCESS) {
        FR_LOG_E("update input buffer failed 0x%x", ret);
        return -1;
    }
    return 0;
}

/* ================================================================
 * inference
 * ================================================================ */
int CFaceRecognitionDV500::inference()
{
    td_s32 ret = sample_common_svp_npu_model_execute(&task_info_);
    if (ret != TD_SUCCESS) {
        FR_LOG_E("model execute failed 0x%x", ret);
        return -1;
    }
    return 0;
}

/* ================================================================
 * postProcess：读取输出特征，L2 归一化
 * ================================================================ */
int CFaceRecognitionDV500::postProcess(std::vector<float>& feature)
{
    // 取 output buffer
    td_u8  *out_ptr = nullptr;
    td_u32  out_size = 0, out_stride = 0;
    td_s32  ret = sample_common_svp_npu_get_output_data_buffer_info(
        &task_info_, 0, &out_ptr, &out_size, &out_stride);
    if (ret != TD_SUCCESS || out_ptr == nullptr) {
        FR_LOG_E("get output buffer failed 0x%x", ret);
        return -1;
    }

    // invalidate cache
    svp_acl_error inv_ret = svp_acl_rt_mem_invalidate(out_ptr, out_size);
    if (inv_ret != SVP_ACL_SUCCESS) {
        FR_LOG_E("mem invalidate failed 0x%x", inv_ret);
        return -1;
    }

    if (out_total_elem_ < embedding_dim_) {
        FR_LOG_E("output total elem %d < expected %d", out_total_elem_, embedding_dim_);
        return -1;
    }

    feature.resize(embedding_dim_);
    const float* data = reinterpret_cast<const float*>(out_ptr);
    for (int i = 0; i < embedding_dim_; ++i) {
        feature[i] = data[i];
    }

    // L2 归一化（与原始完全一致）
    float norm = 0.f;
    for (float v : feature) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-12f) {
        for (float& v : feature) v /= norm;
    }

    return 0;
}

/* ================================================================
 * destroy
 * ================================================================ */
void CFaceRecognitionDV500::destroy()
{
    if (task_initialized_) {
        deinit_task_();
        task_initialized_ = false;
    }
    if (model_loaded_) {
        unload_model_();
        model_loaded_ = false;
    }
    inited_ = false;
    FR_LOG_I("destroyed");
}

/* ================================================================
 * 余弦相似度（与原始一致）
 * ================================================================ */
float CFaceRecognitionDV500::compute_sim(const std::vector<float>& feat1,
                                          const std::vector<float>& feat2)
{
    if (feat1.size() != feat2.size() || feat1.empty()) return 0.f;
    float dot = 0.f, n1 = 0.f, n2 = 0.f;
    for (size_t i = 0; i < feat1.size(); ++i) {
        dot += feat1[i] * feat2[i];
        n1  += feat1[i] * feat1[i];
        n2  += feat2[i] * feat2[i];
    }
    n1 = std::sqrt(n1);
    n2 = std::sqrt(n2);
    if (n1 < 1e-12f || n2 < 1e-12f) return 0.f;
    return dot / (n1 * n2);
}

/* ================================================================
 * ACL 初始化 / 反初始化
 * ================================================================ */
/* ================================================================
 * 模型加载 / 卸载
 * ================================================================ */
int CFaceRecognitionDV500::load_model_()
{
    td_s32 ret = sample_common_svp_npu_load_model(
        cfg_.model_path.c_str(), MODEL_IDX, TD_FALSE);
    if (ret != TD_SUCCESS) {
        FR_LOG_E("load model failed %s 0x%x", cfg_.model_path.c_str(), ret);
        return -1;
    }
    // FR_LOG_I("model loaded: %s", cfg_.model_path.c_str());
    return 0;
}

void CFaceRecognitionDV500::unload_model_()
{
    sample_common_svp_npu_unload_model(MODEL_IDX);
}

/* ================================================================
 * Task 资源创建/销毁
 * ================================================================ */
int CFaceRecognitionDV500::init_task_()
{
    td_s32 ret;

    ret = sample_common_svp_npu_create_input(&task_info_);
    if (ret != TD_SUCCESS) {
        FR_LOG_E("create input failed 0x%x", ret);
        return -1;
    }

    ret = sample_common_svp_npu_create_output(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_input(&task_info_);
        FR_LOG_E("create output failed 0x%x", ret);
        return -1;
    }

    ret = sample_common_svp_npu_create_task_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_output(&task_info_);
        sample_common_svp_npu_destroy_input(&task_info_);
        FR_LOG_E("create task buf failed 0x%x", ret);
        return -1;
    }

    ret = sample_common_svp_npu_create_work_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_task_buf(&task_info_);
        sample_common_svp_npu_destroy_output(&task_info_);
        sample_common_svp_npu_destroy_input(&task_info_);
        FR_LOG_E("create work buf failed 0x%x", ret);
        return -1;
    }

    // FR_LOG_I("task init ok");
    return 0;
}

void CFaceRecognitionDV500::deinit_task_()
{
    sample_common_svp_npu_destroy_work_buf(&task_info_);
    sample_common_svp_npu_destroy_task_buf(&task_info_);
    sample_common_svp_npu_destroy_output(&task_info_);
    sample_common_svp_npu_destroy_input(&task_info_);
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
}

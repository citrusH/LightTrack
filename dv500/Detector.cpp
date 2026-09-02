
#include "Detector.h"
#include <array>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <limits>
#include "svp_acl_rt.h"
#include "svp_acl.h"
#include "svp_acl_ext.h"
#include "sample_common_svp.h"
#include "sample_common_svp_npu.h"
#include "sample_common_svp_npu_model.h"
#include "ModelProfiler.h"
using namespace VISION_ENGINE;


#define DLOGI(fmt, ...) printf("[Detector][I] " fmt "\n", ##__VA_ARGS__)
#define DLOGE(fmt, ...) printf("[Detector][E] " fmt "\n", ##__VA_ARGS__)

namespace {

constexpr int kYoloxInputW = 416;
constexpr int kYoloxInputH = 416;
constexpr int kYoloxNumClasses = 3;
constexpr int kYoloxOutputDim = 5 + kYoloxNumClasses;
constexpr size_t kYoloxPreNmsTopK = 200;
constexpr size_t kYoloxMaxDetPerClass = 100;

struct GridAndStride {
    int grid_x;
    int grid_y;
    int stride;
};

struct YoloxProposal {
    float x1;
    float y1;
    float x2;
    float y2;
    float area;
    float score;
    int label;
};

const std::vector<GridAndStride>& yolox_grids()
{
    static const std::vector<GridAndStride> grids = []() {
        static const int strides[] = {8, 16, 32};
        std::vector<GridAndStride> result;
        result.reserve(3549);
        for (int stride : strides) {
            const int grid_w = kYoloxInputW / stride;
            const int grid_h = kYoloxInputH / stride;
            for (int gy = 0; gy < grid_h; ++gy) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    result.push_back({gx, gy, stride});
                }
            }
        }
        return result;
    }();
    return grids;
}

void sort_and_limit(std::vector<YoloxProposal>& proposals)
{
    const auto score_desc = [](const YoloxProposal& a,
                               const YoloxProposal& b) {
        return a.score > b.score;
    };
    if (proposals.size() > kYoloxPreNmsTopK) {
        auto nth = proposals.begin()
                 + static_cast<std::ptrdiff_t>(kYoloxPreNmsTopK);
        std::nth_element(proposals.begin(), nth, proposals.end(), score_desc);
        proposals.resize(kYoloxPreNmsTopK);
    }
    std::sort(proposals.begin(), proposals.end(), score_desc);
}

void nms_single_class(const std::vector<YoloxProposal>& proposals,
                      float threshold,
                      std::vector<YoloxProposal>& keep)
{
    keep.clear();
    std::vector<uint8_t> removed(proposals.size(), 0);
    keep.reserve(std::min(proposals.size(), kYoloxMaxDetPerClass));
    for (size_t i = 0; i < proposals.size(); ++i) {
        if (removed[i]) continue;
        const YoloxProposal& a = proposals[i];
        keep.push_back(a);
        if (keep.size() >= kYoloxMaxDetPerClass) break;
        for (size_t j = i + 1; j < proposals.size(); ++j) {
            if (removed[j]) continue;
            const YoloxProposal& b = proposals[j];
            const float xx1 = std::max(a.x1, b.x1);
            const float yy1 = std::max(a.y1, b.y1);
            const float xx2 = std::min(a.x2, b.x2);
            const float yy2 = std::min(a.y2, b.y2);
            const float iw = xx2 - xx1;
            const float ih = yy2 - yy1;
            if (iw <= 0.0f || ih <= 0.0f) continue;
            const float inter = iw * ih;
            const float union_area = a.area + b.area - inter;
            if (union_area > 0.0f && inter / union_area >= threshold)
                removed[j] = 1;
        }
    }
}

} // namespace


int Detector::init(const DetectorConfig& cfg)
{
    if (inited_) { DLOGE("already inited"); return -1; }
    destroy();
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
    cfg_ = cfg;
    thresh_iou_ = cfg.threshold_iou;

    if (load_model_() != 0) { destroy(); return -1; }
    model_loaded_ = true;

    sample_svp_npu_model_info* model_info =
        sample_common_svp_npu_get_model_info(MODEL_IDX);
    if (model_info == nullptr || model_info->model_desc == nullptr) {
        DLOGE("get model info failed");
        destroy(); return -1;
    }
    svp_acl_mdl_desc* desc = model_info->model_desc;

    svp_acl_mdl_io_dims in_dims{};
    svp_acl_error acl_ret = svp_acl_mdl_get_input_dims(desc, 0, &in_dims);
    if (acl_ret != SVP_ACL_SUCCESS) {
        DLOGE("get input dims failed 0x%x", acl_ret);
        destroy(); return -1;
    }

    net_h_ = static_cast<int>(in_dims.dims[2]);
    net_w_ = static_cast<int>(in_dims.dims[3]);

    svp_acl_mdl_io_dims out_dims{};
    acl_ret = svp_acl_mdl_get_output_dims(desc, 0, &out_dims);
    if (acl_ret != SVP_ACL_SUCCESS || out_dims.dim_count < 2) {
        DLOGE("YOLOX get output dims failed 0x%x", acl_ret);
        destroy(); return -1;
    }
    const int anchor_dim = static_cast<int>(out_dims.dim_count) - 2;
    const int value_dim = static_cast<int>(out_dims.dim_count) - 1;
    size_t output_elements = 1;
    for (size_t d = 0; d < out_dims.dim_count; ++d) {
        if (out_dims.dims[d] <= 0) {
            DLOGE("YOLOX invalid output dim[%zu]=%ld", d,
                  (long)out_dims.dims[d]);
            destroy(); return -1;
        }
        output_elements *= static_cast<size_t>(out_dims.dims[d]);
    }
    if (out_dims.dims[anchor_dim] != cfg_.num_anchor
        || out_dims.dims[value_dim] != cfg_.dim_anchor
        || output_elements != static_cast<size_t>(cfg_.num_anchor)
                            * static_cast<size_t>(cfg_.dim_anchor)
        || cfg_.num_anchor != 3549 || cfg_.dim_anchor != kYoloxOutputDim
        || svp_acl_mdl_get_output_data_type(desc, 0) != SVP_ACL_FLOAT) {
        DLOGE("YOLOX output contract mismatch: rank=%zu tail=[%ld,%ld] "
              "elements=%zu dtype=%d expected=[...,3549,8] FP32",
              out_dims.dim_count, (long)out_dims.dims[anchor_dim],
              (long)out_dims.dims[value_dim], output_elements,
              (int)svp_acl_mdl_get_output_data_type(desc, 0));
        destroy(); return -1;
    }
    DLOGI("YOLOX output anchors=%d dim=%d classes=%d",
          cfg_.num_anchor, cfg_.dim_anchor, kYoloxNumClasses);

    task_info_.cfg.max_batch_num     = 1;
    task_info_.cfg.dynamic_batch_num = 1;
    task_info_.cfg.total_t           = 0;
    task_info_.cfg.is_cached         = TD_TRUE;
    task_info_.cfg.model_idx         = MODEL_IDX;

    task_initialized_ = true;
    if (init_task_() != 0) { destroy(); return -1; }

    inited_ = true;
    DLOGI("init ok: %s", cfg_.model_path.c_str());
    return 0;
}


int Detector::run(const cv::Mat& bgr_image, float thresh_score,
                  std::vector<ObjDetInfo>& result)
{
    // auto start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::steady_clock::now().time_since_epoch()).count();
    if (!inited_) { DLOGE("not inited"); return -1; }
    result.clear();

    if (bgr_image.empty() || !std::isfinite(thresh_score)
        || thresh_score <= 0.0f || thresh_score > 1.0f) {
        DLOGE("invalid YOLOX run input: empty=%d threshold=%f",
              bgr_image.empty() ? 1 : 0, thresh_score);
        return -1;
    }

    thresh_conf_ = thresh_score;


    float scale_w = 1.f, scale_h = 1.f;

    int ret = 0;
    fxprof::add_call(fxprof::Model::Detector);

    // float scale_w = 1, scale_h = 1;
    //--------------------------------
    //		     pre-precess
    //--------------------------------
    {
        fxprof::ScopedPhase _p(fxprof::Model::Detector, fxprof::Phase::Pre);
        ret = preProcess(bgr_image, scale_w, scale_h);
    }
    if (ret < 0) {
        ERROR_LOG("pre-precess fail! ret=%d\n", ret);
        return -1;
    }

    //--------------------------------
    //		  network forward
    //--------------------------------
    {
        fxprof::ScopedPhase _p(fxprof::Model::Detector, fxprof::Phase::Infer);
        ret = inference();
    }
    if (ret < 0) {
        ERROR_LOG("inference fail! ret=%d\n", ret);
        return -1;
    }

    //--------------------------------
    //			post-process
    //--------------------------------
    {
        fxprof::ScopedPhase _p(fxprof::Model::Detector, fxprof::Phase::Post);
        ret = postProcess(bgr_image.size(), scale_w, scale_h, thresh_conf_, result);
    }
    if (ret < 0) {
        ERROR_LOG("post-precess fail! ret=%d\n", ret);
        return -1;
    }

    return 0;
}

int Detector::preProcess(const cv::Mat& image, float& scale_w, float& scale_h)
{
    if (image.empty() || image.type() != CV_8UC3) {
        DLOGE("YOLOX expects non-empty CV_8UC3 input, type=%d", image.type());
        return -1;
    }

    const float resize_ratio = std::min(
        static_cast<float>(net_h_) / static_cast<float>(image.rows),
        static_cast<float>(net_w_) / static_cast<float>(image.cols));
    if (!std::isfinite(resize_ratio) || resize_ratio <= 0.0f) {
        DLOGE("YOLOX invalid resize ratio: %f", resize_ratio);
        return -1;
    }
    const int new_w = static_cast<int>(image.cols * resize_ratio);
    const int new_h = static_cast<int>(image.rows * resize_ratio);
    if (new_w <= 0 || new_h <= 0 || new_w > net_w_ || new_h > net_h_) {
        DLOGE("YOLOX invalid resized size: %dx%d", new_w, new_h);
        return -1;
    }
    scale_w = 1.0f / resize_ratio;
    scale_h = 1.0f / resize_ratio;

    td_u8  *dev_ptr  = nullptr;
    td_u32  buf_size = 0;
    td_u32  stride   = 0;
    td_s32  ret = sample_common_svp_npu_get_input_data_buffer_info(
        &task_info_, 0, &dev_ptr, &buf_size, &stride);
    if (ret != TD_SUCCESS || dev_ptr == nullptr) {
        DLOGE("get input buffer info failed 0x%x", ret);
        return -1;
    }

    const int row_bytes = net_w_ * 3;
    const td_u32 valid_size = stride * net_h_;
    if (static_cast<td_u32>(row_bytes) > stride) {
        DLOGE("stride too small: stride=%u need=%d", stride, row_bytes);
        return -1;
    }
    if (valid_size > buf_size) {
        DLOGE("buffer not enough: stride=%u buf=%u", stride, buf_size);
        return -1;
    }

    cv::Mat dev_img(net_h_, net_w_, CV_8UC3, dev_ptr, static_cast<size_t>(stride));
    // YOLOX 使用左上对齐，右侧/下侧 padding 固定为 114。
    memset(dev_ptr, 0, valid_size);
    dev_img.setTo(cv::Scalar(114, 114, 114));
    cv::resize(image, dev_img(cv::Rect(0, 0, new_w, new_h)),
               cv::Size(new_w, new_h), 0, 0, cv::INTER_NEAREST);

    // 当前同名 OM/AIPP 契约接收 OpenCV BGR；仅配置明确要求时才交换通道。
    if (cfg_.bgr2rgb)
        cv::cvtColor(dev_img, dev_img, cv::COLOR_BGR2RGB);

    svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, valid_size);
    if (flush_ret != SVP_ACL_SUCCESS) {
        DLOGE("mem flush failed 0x%x", flush_ret);
        return -1;
    }

    ret = sample_common_svp_npu_update_input_data_buffer_info(
        dev_ptr, valid_size, stride, 0, &task_info_);
    if (ret != TD_SUCCESS) {
        DLOGE("update input buffer failed 0x%x", ret);
        return -1;
    }
    return 0;
}


/* 原生 SVP-NPU 同步推理。 */
int Detector::inference()
{
    td_s32 ret = sample_common_svp_npu_model_execute(&task_info_);
    if (ret != TD_SUCCESS) {
        DLOGE("model execute failed 0x%x", ret);
        return -1;
    }
    return 0;
}

/* YOLOX output: FP32 [1,3549,8], row = x/y/w/h/obj/face/body/head. */
int Detector::postProcess(cv::Size src_img_size,
                          float scale_w, float scale_h,
                          float thresh_conf,
                          std::vector<ObjDetInfo>& result)
{
    result.clear();

    if (src_img_size.width <= 0 || src_img_size.height <= 0
        || !std::isfinite(scale_w) || !std::isfinite(scale_h)
        || scale_w <= 0.0f || scale_h <= 0.0f
        || !std::isfinite(thresh_conf) || thresh_conf <= 0.0f
        || thresh_conf > 1.0f || thresh_iou_ <= 0.0f || thresh_iou_ > 1.0f) {
        DLOGE("YOLOX invalid postprocess arguments");
        return -1;
    }

    td_u8* out_ptr    = nullptr;
    td_u32 out_size   = 0;
    td_u32 out_stride = 0;

    if (sample_common_svp_npu_get_output_data_buffer_info(
            &task_info_, 0, &out_ptr, &out_size, &out_stride) != TD_SUCCESS) {
        DLOGE("get output buffer info failed");
        return -1;
    }

    if (out_ptr == nullptr
        || svp_acl_rt_mem_invalidate(out_ptr, out_size) != SVP_ACL_SUCCESS) {
        DLOGE("invalidate output buffer failed");
        return -1;
    }
    sample_svp_npu_model_info* model_info =
        sample_common_svp_npu_get_model_info(MODEL_IDX);
    if (model_info == nullptr || model_info->model_desc == nullptr) {
        DLOGE("YOLOX model descriptor unavailable");
        return -1;
    }
    svp_acl_mdl_desc* desc = model_info->model_desc;
    size_t stride_bytes = out_stride;
    if (stride_bytes == 0)
        stride_bytes = svp_acl_mdl_get_output_default_stride(desc, 0);
    if (stride_bytes == 0)
        stride_bytes = static_cast<size_t>(kYoloxOutputDim) * sizeof(float);
    if (stride_bytes % sizeof(float) != 0
        || stride_bytes < static_cast<size_t>(kYoloxOutputDim) * sizeof(float)) {
        DLOGE("YOLOX invalid output stride: %zu", stride_bytes);
        return -1;
    }
    const size_t required_bytes =
        static_cast<size_t>(cfg_.num_anchor - 1) * stride_bytes
        + static_cast<size_t>(kYoloxOutputDim) * sizeof(float);
    if (required_bytes > out_size) {
        DLOGE("YOLOX output buffer too small: actual=%u required=%zu stride=%zu",
              out_size, required_bytes, stride_bytes);
        return -1;
    }

    const size_t stride_elem = stride_bytes / sizeof(float);
    const float* output = reinterpret_cast<const float*>(out_ptr);
    const std::vector<GridAndStride>& grids = yolox_grids();
    if (grids.size() != static_cast<size_t>(cfg_.num_anchor)) {
        DLOGE("YOLOX grid count mismatch: %zu", grids.size());
        return -1;
    }

    std::array<std::vector<YoloxProposal>, kYoloxNumClasses> class_proposals;
    for (auto& proposals : class_proposals) proposals.reserve(128);
    for (int anchor_idx = 0; anchor_idx < cfg_.num_anchor; ++anchor_idx) {
        const float* pred = output + static_cast<size_t>(anchor_idx) * stride_elem;
        const float objectness = pred[4];
        if (!std::isfinite(objectness) || objectness < thresh_conf) continue;

        int label = 0;
        float class_score = std::isfinite(pred[5]) ? pred[5] : 0.0f;
        for (int cls = 1; cls < kYoloxNumClasses; ++cls) {
            const float value = pred[5 + cls];
            if (std::isfinite(value) && value > class_score) {
                class_score = value;
                label = cls;
            }
        }
        const float score = objectness * class_score;
        if (!std::isfinite(score) || score < thresh_conf) continue;

        const float raw_x = pred[0];
        const float raw_y = pred[1];
        const float raw_w = pred[2];
        const float raw_h = pred[3];
        if (!std::isfinite(raw_x) || !std::isfinite(raw_y)
            || !std::isfinite(raw_w) || !std::isfinite(raw_h)) continue;

        const GridAndStride& gs = grids[anchor_idx];
        const float grid_stride = static_cast<float>(gs.stride);
        const float cx = (raw_x + gs.grid_x) * grid_stride;
        const float cy = (raw_y + gs.grid_y) * grid_stride;
        const float box_w = std::exp(raw_w) * grid_stride;
        const float box_h = std::exp(raw_h) * grid_stride;
        if (!std::isfinite(cx) || !std::isfinite(cy)
            || !std::isfinite(box_w) || !std::isfinite(box_h)
            || box_w <= 0.0f || box_h <= 0.0f) continue;

        YoloxProposal proposal;
        proposal.x1 = std::max(0.0f, std::min(cx - box_w * 0.5f,
                                              static_cast<float>(net_w_)));
        proposal.y1 = std::max(0.0f, std::min(cy - box_h * 0.5f,
                                              static_cast<float>(net_h_)));
        proposal.x2 = std::max(0.0f, std::min(cx + box_w * 0.5f,
                                              static_cast<float>(net_w_)));
        proposal.y2 = std::max(0.0f, std::min(cy + box_h * 0.5f,
                                              static_cast<float>(net_h_)));
        const float width = proposal.x2 - proposal.x1;
        const float height = proposal.y2 - proposal.y1;
        if (width <= 0.0f || height <= 0.0f) continue;
        proposal.area = width * height;
        proposal.score = score;
        proposal.label = label;
        class_proposals[label].push_back(proposal);
    }

    std::vector<YoloxProposal> keep;
    keep.reserve(kYoloxPreNmsTopK);
    for (auto& proposals : class_proposals) sort_and_limit(proposals);
    for (int cls = 0; cls < kYoloxNumClasses; ++cls) {
        nms_single_class(class_proposals[cls], thresh_iou_, keep);
        for (const YoloxProposal& proposal : keep) {
            float x1 = proposal.x1 * scale_w;
            float y1 = proposal.y1 * scale_h;
            float x2 = proposal.x2 * scale_w;
            float y2 = proposal.y2 * scale_h;
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(src_img_size.width)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(src_img_size.height)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(src_img_size.width)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(src_img_size.height)));
            if (x2 <= x1 || y2 <= y1) continue;

            const int left = std::max(0, std::min(static_cast<int>(std::floor(x1)),
                                                  src_img_size.width - 1));
            const int top = std::max(0, std::min(static_cast<int>(std::floor(y1)),
                                                 src_img_size.height - 1));
            const int right = std::max(left + 1,
                std::min(static_cast<int>(std::ceil(x2)), src_img_size.width));
            const int bottom = std::max(top + 1,
                std::min(static_cast<int>(std::ceil(y2)), src_img_size.height));

            ObjDetInfo obj;
            obj.label = proposal.label;
            obj.score = proposal.score;
            // 项目跟踪层约定：cv::Rect 的 width/height 字段承载 x2/y2。
            obj.box = cv::Rect(left, top, right, bottom);
            result.push_back(obj);
        }
    }

    std::sort(result.begin(), result.end(),
              [](const ObjDetInfo& a, const ObjDetInfo& b) {
                  return a.score > b.score;
              });

    return 0;
}


/* 模型 task 资源释放。 */
void Detector::destroy()
{
    DLOGI("destroy called, inited=%d", inited_);  // 加这行

    if (task_initialized_) {
        deinit_task_();
        task_initialized_ = false;
    }
    if (model_loaded_) {
        unload_model_();
        model_loaded_ = false;
    }

    inited_ = false;
    DLOGI("destroyed");

    DLOGI("deinit_task_ done");
    DLOGI("unload_model_ done");
}

/* ================================================================== *
 * 模型加载 / 卸载  —— 无需修改
 * ================================================================== */
int Detector::load_model_()
{
    td_s32 ret = sample_common_svp_npu_load_model(
        cfg_.model_path.c_str(), MODEL_IDX, TD_FALSE);
    if (ret != TD_SUCCESS) {
        DLOGE("load model failed %s 0x%x", cfg_.model_path.c_str(), ret);
        return -1;
    }
    // DLOGI("model loaded: %s", cfg_.model_path.c_str());
    return 0;
}

void Detector::unload_model_()
{
    sample_common_svp_npu_unload_model(MODEL_IDX);
}

/* ================================================================== *
 * Task 初始化 / 反初始化  —— 无需修改
 * ================================================================== */
int Detector::init_task_()
{
    td_s32 ret;

    ret = sample_common_svp_npu_create_input(&task_info_);
    if (ret != TD_SUCCESS) {
        DLOGE("create input failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_output(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_input(&task_info_);
        DLOGE("create output failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_task_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_output(&task_info_);
        sample_common_svp_npu_destroy_input (&task_info_);
        DLOGE("create task buf failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_work_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_task_buf(&task_info_);
        sample_common_svp_npu_destroy_output  (&task_info_);
        sample_common_svp_npu_destroy_input   (&task_info_);
        DLOGE("create work buf failed 0x%x", ret); return -1;
    }

    // DLOGI("task init ok");
    return 0;
}

void Detector::deinit_task_()
{
    sample_common_svp_npu_destroy_work_buf(&task_info_);
    sample_common_svp_npu_destroy_task_buf(&task_info_);
    sample_common_svp_npu_destroy_output  (&task_info_);
    sample_common_svp_npu_destroy_input   (&task_info_);
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
}

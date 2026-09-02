#include "Facekps.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

#include "svp_acl_rt.h"
#include "svp_acl.h"
#include "svp_acl_ext.h"
#include "sample_common_svp.h"
#include "sample_common_svp_npu.h"
#include "sample_common_svp_npu_model.h"
#include "ModelProfiler.h"
using namespace VISION_ENGINE;


#define FKLOG_I(fmt, ...) printf("[FaceKps][I] " fmt "\n", ##__VA_ARGS__)
#define FKLOG_E(fmt, ...) printf("[FaceKps][E] " fmt "\n", ##__VA_ARGS__)

/* ================================================================
 * fp16 → float32
 * 与 Detector_dv500 完全一致，集中在此定义
 * ================================================================ */
/*static*/ float CFaceKeypoint106::fp16_to_f32(uint16_t h)
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
        f = (sign << 31) | (0xffu << 23) | (mantissa << 13);  /* inf / nan */
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

/* ================================================================
 * GetAffineTransform  —  与原始完全一致
 * 构造从原图（以 center 为中心，scale 缩放，rotation 旋转）到
 * 边长为 output_size 的正方形画布的 2×3 仿射矩阵
 * ================================================================ */
cv::Mat CFaceKeypoint106::GetAffineTransform(const cv::Point2f& center,
                                              int   output_size,
                                              float scale,
                                              float rotation)
{
    cv::Mat M(2, 3, CV_64F);

    double rad = rotation * CV_PI / 180.0;
    double sn  = std::sin(rad);
    double cs  = std::cos(rad);

    M.at<double>(0, 0) =  scale * cs;
    M.at<double>(0, 1) =  scale * -sn;
    M.at<double>(1, 0) =  scale * sn;
    M.at<double>(1, 1) =  scale * cs;

    M.at<double>(0, 2) = output_size * 0.5
                       - scale * cs * center.x
                       + scale * sn * center.y;
    M.at<double>(1, 2) = output_size * 0.5
                       - scale * sn * center.x
                       - scale * cs * center.y;
    return M;
}


std::vector<cv::Point2f> CFaceKeypoint106::TransformPoints(
    const std::vector<cv::Point2f>& pts,
    const cv::Mat& M)
{
    std::vector<cv::Point2f> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        float x = static_cast<float>(
            M.at<double>(0, 0) * pts[i].x +
            M.at<double>(0, 1) * pts[i].y +
            M.at<double>(0, 2));
        float y = static_cast<float>(
            M.at<double>(1, 0) * pts[i].x +
            M.at<double>(1, 1) * pts[i].y +
            M.at<double>(1, 2));
        out[i] = { x, y };
    }
    return out;
}


int CFaceKeypoint106::init(const FaceKeypointConfig& config)
{
    if (inited_) { FKLOG_E("already inited"); return -1; }
    destroy();
    cfg_ = config;

    /* ---- 1. 加载 .om 模型；runtime 已由 LightTracker 初始化 ---- */
    if (load_model_() != 0) { destroy(); return -1; }
    model_loaded_ = true;

    /* ---- 3. 从模型描述读取输入/输出元信息 ---- */
    sample_svp_npu_model_info* model_info =
        sample_common_svp_npu_get_model_info(MODEL_IDX);
    if (model_info == nullptr || model_info->model_desc == nullptr) {
        FKLOG_E("get model info failed");
        destroy(); return -1;
    }
    svp_acl_mdl_desc* desc = model_info->model_desc;

    /* ---- 3a. 输入尺寸（对应原始 config.inputSize） ---- */
    svp_acl_mdl_io_dims in_dims{};
    svp_acl_error acl_ret = svp_acl_mdl_get_input_dims(desc, 0, &in_dims);
    if (acl_ret != SVP_ACL_SUCCESS) {
        FKLOG_E("get input dims failed 0x%x", acl_ret);
        destroy(); return -1;
    }
    /*
     * NCHW layout: dims[0]=N, dims[1]=C, dims[2]=H, dims[3]=W
     * 以模型实际尺寸为准，覆盖 config 中的 inputSize
     */
    net_h_ = (int)in_dims.dims[2];
    net_w_ = (int)in_dims.dims[3];
    cfg_.inputSize = cv::Size(net_w_, net_h_);
    FKLOG_I("model input: N=%d C=%d H=%d W=%d",
            (int)in_dims.dims[0], (int)in_dims.dims[1], net_h_, net_w_);

    /* ---- 3b. 输出元信息（shape + stride）---- */
    svp_acl_mdl_io_dims out_dims{};
    svp_acl_mdl_get_output_dims(desc, 0, &out_dims);

    // /* 打印输出shape，方便调试确认 */
    FKLOG_I("output shape:");
    for (int d = 0; d < (int)out_dims.dim_count; ++d)
        FKLOG_I("  dim[%d] = %ld", d, (long)out_dims.dims[d]);

    /*
     * 期望输出 shape：[1, numPoints*2] 或 [1, 1, numPoints*2]
     * 取最后一个维度作为总元素数
     */
    out_total_elem_ = (int)out_dims.dims[out_dims.dim_count - 1];
    FKLOG_I("out_total_elem = %d  (expected = %d)",
            out_total_elem_, cfg_.numPoints * 2);

    /* stride：含 padding 的每行实际字节数 */
    size_t default_stride = svp_acl_mdl_get_output_default_stride(desc, 0);
    size_t elem_bytes     = cfg_.output_fp16 ? sizeof(uint16_t) : sizeof(float);
    out_stride_elem_      = (default_stride > 0)
                          ? default_stride / elem_bytes
                          : (size_t)out_total_elem_;
    FKLOG_I("output stride = %zu bytes, stride_elem = %zu, elem_bytes = %zu",
            default_stride, out_stride_elem_, elem_bytes);

    /* ---- 3c. 确认输出数据类型（打印供调试） ---- */
    svp_acl_data_type out_dtype = svp_acl_mdl_get_output_data_type(desc, 0);
    FKLOG_I("output dtype = %d  (0=fp32, 1=fp16)", (int)out_dtype);
    if (out_dtype == SVP_ACL_FLOAT16 && !cfg_.output_fp16) {
        FKLOG_I("WARNING: model dtype is fp16 but config.output_fp16=false, auto-correcting");
        cfg_.output_fp16 = true;
    }

    /* ---- 4. task cfg ---- */
    task_info_.cfg.max_batch_num     = 1;
    task_info_.cfg.dynamic_batch_num = 1;
    task_info_.cfg.total_t           = 0;
    task_info_.cfg.is_cached         = TD_TRUE;
    task_info_.cfg.model_idx         = MODEL_IDX;

    /* ---- 5. 创建 input/output/task_buf/work_buf ---- */
    task_initialized_ = true;
    if (init_task_() != 0) { destroy(); return -1; }

    inited_ = true;
    FKLOG_I("init ok: %s  input=%dx%d  points=%d",
            cfg_.model_path.c_str(), net_w_, net_h_, cfg_.numPoints);
    return 0;
}

/* ================================================================
 * run
 * 对应原始 CFaceKeypoint106::run()，接口完全一致
 * ================================================================ */
FaceKeypointResult CFaceKeypoint106::run(const cv::Mat& src_img,
                                          const cv::Rect& face_bbox)
{
    // auto start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    //     std::chrono::steady_clock::now().time_since_epoch()).count();
    FaceKeypointResult result;
    result.face_rect = face_bbox;

    if (!inited_) { FKLOG_E("not initialized"); return result; }
    if (src_img.empty()) { FKLOG_E("empty image");  return result; }

    cv::Mat dst_dbg, M;
    fxprof::add_call(fxprof::Model::FaceKps);
    int ret = 0;
    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceKps, fxprof::Phase::Pre);
        ret = preProcess(src_img, face_bbox, dst_dbg, M);
    }
    if (ret != 0) {
        FKLOG_E("preProcess failed");
        return result;
    }

    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceKps, fxprof::Phase::Infer);
        ret = inference();
    }
    if (ret != 0) {
        FKLOG_E("inference failed");
        return result;
    }

    {
        fxprof::ScopedPhase _p(fxprof::Model::FaceKps, fxprof::Phase::Post);
        ret = postProcess(M, result.points);
    }
    if (ret != 0) {
        FKLOG_E("postProcess failed");
    }

    return result;
}

/* ================================================================
 * preProcess
 *
 * 与原始逻辑完全一致的部分：
 *   GetAffineTransform → warpAffine → bgr2rgb
 *
 * DV500 替换 cv::dnn::blobFromImage + net_.setInput 的部分：
 *   根据 cfg_.input_format 选择 HWC 或 CHW 写入 device buffer
 *
 * 注：AIPP 负责归一化（var_reci），此处送 uint8 原始值
 * ================================================================ */
int CFaceKeypoint106::preProcess(const cv::Mat& src,
                                  const cv::Rect& bbox,
                                  cv::Mat& dst_dbg,
                                  cv::Mat& M)
{
    /* ---- 仿射变换（与原始完全一致） ---- */
    float w = static_cast<float>(bbox.width) - static_cast<float>(bbox.x);
    float h = static_cast<float>(bbox.height) - static_cast<float>(bbox.y);
    cv::Point2f center(bbox.x + w * 0.5f, bbox.y + h * 0.5f);

    /*
     * scale：使人脸在 inputSize 内居中，留 1.5x 的上下文区域
     * 与原始 preProcess 完全一致
     */
    float scale = cfg_.inputSize.width / (std::max(w, h) * 1.5f);
    M = GetAffineTransform(center, cfg_.inputSize.width, scale, 0.f);

    cv::Mat dst;
    cv::warpAffine(src, dst, M, cfg_.inputSize,
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    // INTER_NEAREST

    /* ---- BGR → RGB（与原始一致，在写入设备前完成） ---- */
    if (cfg_.bgr2rgb)
        cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

    dst_dbg = dst;  /* 保存一份供调试，不影响流程 */

    /* ---- 取 device 侧 input buffer ---- */
    td_u8  *dev_ptr  = nullptr;
    td_u32  buf_size = 0, stride = 0;
    td_s32  ret = sample_common_svp_npu_get_input_data_buffer_info(
        &task_info_, 0, &dev_ptr, &buf_size, &stride);
    if (ret != TD_SUCCESS) {
        FKLOG_E("get input buffer info failed 0x%x", ret);
        return -1;
    }

    int H = dst.rows, W = dst.cols;   /* inputSize.height, inputSize.width */

    if (cfg_.input_format == InputFormat::RGB_PACKAGE) {
        /* ============================================================
         * RGB_PACKAGE：HWC 交错格式，直接 memcpy
         * AIPP: input_format=RGB_PACKAGE
         * 对应原始: blobFromImage(swapRB=false) 之前的 uint8 数据
         * ============================================================ */
        td_u32 data_bytes = (td_u32)(H * W * 3);
        if (data_bytes > buf_size) {
            FKLOG_E("data %u > buf %u", data_bytes, buf_size);
            return -1;
        }

        if (!dst.isContinuous()) dst = dst.clone();
        errno_t cp_ret = memcpy_s(dev_ptr, buf_size, dst.data, data_bytes);
        if (cp_ret != EOK) { FKLOG_E("memcpy_s failed %d", cp_ret); return -1; }

        svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
        if (flush_ret != SVP_ACL_SUCCESS) {
            FKLOG_E("mem flush failed 0x%x", flush_ret); return -1;
        }

        /* stride = W * C（RGB_PACKAGE 行宽） */
        td_u32 packed_stride = (td_u32)(W * 3);
        ret = sample_common_svp_npu_update_input_data_buffer_info(
            dev_ptr, data_bytes, packed_stride, 0, &task_info_);

    } else {
        /* ============================================================
         * RGB_PLANAR：CHW 平面格式，需 split 后按通道写入
         * AIPP: input_format=RGB_PLANAR
         * 对应原始: blobFromImage 内部的 HWC→CHW 转换
         * ============================================================ */
        td_u32 plane_size  = (td_u32)(H * W);
        td_u32 data_bytes  = plane_size * 3;
        if (data_bytes > buf_size) {
            FKLOG_E("data %u > buf %u", data_bytes, buf_size);
            return -1;
        }

        std::vector<cv::Mat> channels(3);
        cv::split(dst, channels);   /* [0]=R, [1]=G, [2]=B（已 bgr2rgb） */

        td_u8 *ptr = dev_ptr;
        for (int c = 0; c < 3; ++c) {
            cv::Mat ch = channels[c].isContinuous()
                       ? channels[c] : channels[c].clone();
            errno_t cp_ret = memcpy_s(ptr,
                                      buf_size - (td_u32)(ptr - dev_ptr),
                                      ch.data, plane_size);
            if (cp_ret != EOK) {
                FKLOG_E("memcpy_s ch%d failed %d", c, cp_ret); return -1;
            }
            ptr += plane_size;
        }

        svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
        if (flush_ret != SVP_ACL_SUCCESS) {
            FKLOG_E("mem flush failed 0x%x", flush_ret); return -1;
        }

        /* stride = W（RGB_PLANAR 单通道行宽） */
        td_u32 planar_stride = (td_u32)W;
        ret = sample_common_svp_npu_update_input_data_buffer_info(
            dev_ptr, data_bytes, planar_stride, 0, &task_info_);
    }

    if (ret != TD_SUCCESS) {
        FKLOG_E("update input buffer failed 0x%x", ret);
        return -1;
    }

    return 0;
}

/* ================================================================
 * inference
 * 对应原始：net_.forward(outs, ...)
 * DV500：sample_common_svp_npu_model_execute
 * ================================================================ */
int CFaceKeypoint106::inference()
{
    td_s32 ret = sample_common_svp_npu_model_execute(&task_info_);
    if (ret != TD_SUCCESS) {
        FKLOG_E("model execute failed 0x%x", ret);
        return -1;
    }
    return 0;
}

/* ================================================================
 * postProcess
 *
 * 与原始逻辑完全一致的部分：
 *   (data[i*2] + 1) * half  → 像素坐标
 *   invertAffineTransform   → 逆仿射
 *   TransformPoints         → 映射回原图
 *
 * DV500 替换 output_.data 读取的部分：
 *   get_output_data_buffer_info + invalidate
 *   + fp16_to_f32（如需要）+ stride 步进
 * ================================================================ */
int CFaceKeypoint106::postProcess(const cv::Mat& M,
                                   std::vector<FaceKeypoint>& result)
{
    result.clear();

    /* ---- 取 output buffer ---- */
    td_u8  *out_ptr  = nullptr;
    td_u32  out_size = 0, out_stride = 0;
    td_s32  ret = sample_common_svp_npu_get_output_data_buffer_info(
        &task_info_, 0, &out_ptr, &out_size, &out_stride);
    if (ret != TD_SUCCESS || out_ptr == nullptr) {
        FKLOG_E("get output buffer failed 0x%x", ret);
        return -1;
    }

    /*
     * ★ invalidate：NPU 写完后 CPU 必须先 invalidate cache 再读
     *   否则读到的是 CPU cache 中的旧数据
     */
    svp_acl_error inv_ret = svp_acl_rt_mem_invalidate(out_ptr, out_size);
    if (inv_ret != SVP_ACL_SUCCESS) {
        FKLOG_E("mem invalidate failed 0x%x", inv_ret);
        return -1;
    }

    /*
     * 输出形状：[1, numPoints*2] 或 [1, 1, numPoints*2]
     * 坐标排列：[x0, y0, x1, y1, ... x105, y105]
     * 取值范围：[-1, 1]（与原始 postProcess 假设完全一致）
     *
     * stride 步进：out_stride_elem_（init 时从模型 desc 读取，含 padding）
     */
    int expected = cfg_.numPoints * 2;
    if (out_total_elem_ < expected) {
        FKLOG_E("output elem %d < expected %d", out_total_elem_, expected);
        return -1;
    }

    const float half = cfg_.inputSize.width * 0.5f;  /* 与原始完全一致 */

    std::vector<cv::Point2f> pts;
    pts.reserve(cfg_.numPoints);

    if (cfg_.output_fp16) {
        /* ---- fp16 输出读取 ---- */
        const uint16_t *data = reinterpret_cast<const uint16_t *>(out_ptr);
        for (int i = 0; i < cfg_.numPoints; ++i) {
            /*
             * 注意：如果 stride_elem > total_elem，说明有行 padding
             * 但 numPoints*2 通常在同一行，直接按 i*2 偏移即可
             * 如果模型输出是 [numPoints, 2] 则需要 stride 步进
             */
            float x = (fp16_to_f32(data[i * 2    ]) + 1.f) * half;
            float y = (fp16_to_f32(data[i * 2 + 1]) + 1.f) * half;
            pts.emplace_back(x, y);
        }
    } else {
        /* ---- fp32 输出读取（与原始 data[i*2] 完全一致） ---- */
        const float *data = reinterpret_cast<const float *>(out_ptr);
        for (int i = 0; i < cfg_.numPoints; ++i) {
            float x = (data[i * 2    ] + 1.f) * half;  /* 与原始完全一致 */
            float y = (data[i * 2 + 1] + 1.f) * half;  /* 与原始完全一致 */
            pts.emplace_back(x, y);
        }
    }

    /* ---- 逆仿射映射回原图坐标（与原始完全一致） ---- */
    cv::Mat IM;
    cv::invertAffineTransform(M, IM);
    auto world_pts = TransformPoints(pts, IM);

    result.resize(world_pts.size());
    for (size_t i = 0; i < world_pts.size(); ++i) {
        result[i].x = world_pts[i].x;
        result[i].y = world_pts[i].y;
    }
    return 0;
}

/* ================================================================
 * destroy
 * 对应原始：net_ = cv::dnn::Net()
 * ================================================================ */
void CFaceKeypoint106::destroy()
{
    if (task_initialized_) {
        deinit_task_();
        task_initialized_ = false;
    }
    if (model_loaded_) {
        unload_model_();
        model_loaded_ = false;
    }

    inited_          = false;
    out_stride_elem_ = 0;
    out_total_elem_  = 0;
    FKLOG_I("destroyed");
}

/* ================================================================
 * ACL 初始化 / 反初始化
 * 与 Detector_dv500 完全一致
 * ================================================================ */
/* ================================================================
 * 模型加载 / 卸载
 * 与 Detector_dv500 完全一致
 * ================================================================ */
int CFaceKeypoint106::load_model_()
{
    td_s32 ret = sample_common_svp_npu_load_model(
        cfg_.model_path.c_str(), MODEL_IDX, TD_FALSE);
    if (ret != TD_SUCCESS) {
        FKLOG_E("load model failed %s 0x%x", cfg_.model_path.c_str(), ret);
        return -1;
    }
    // FKLOG_I("model loaded: %s", cfg_.model_path.c_str());
    return 0;
}

void CFaceKeypoint106::unload_model_()
{
    sample_common_svp_npu_unload_model(MODEL_IDX);
}

/* ================================================================
 * Task 初始化 / 反初始化
 * 与 Detector_dv500 完全一致
 * ================================================================ */
int CFaceKeypoint106::init_task_()
{
    td_s32 ret;

    ret = sample_common_svp_npu_create_input(&task_info_);
    if (ret != TD_SUCCESS) {
        FKLOG_E("create input failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_output(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_input(&task_info_);
        FKLOG_E("create output failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_task_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_output(&task_info_);
        sample_common_svp_npu_destroy_input (&task_info_);
        FKLOG_E("create task buf failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_work_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_task_buf(&task_info_);
        sample_common_svp_npu_destroy_output  (&task_info_);
        sample_common_svp_npu_destroy_input   (&task_info_);
        FKLOG_E("create work buf failed 0x%x", ret); return -1;
    }

    // FKLOG_I("task init ok");
    return 0;
}

void CFaceKeypoint106::deinit_task_()
{
    sample_common_svp_npu_destroy_work_buf(&task_info_);
    sample_common_svp_npu_destroy_task_buf(&task_info_);
    sample_common_svp_npu_destroy_output  (&task_info_);
    sample_common_svp_npu_destroy_input   (&task_info_);
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
}

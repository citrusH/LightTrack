
#include "PersonReID.h"
#include <cstring>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "svp_acl_rt.h"
#include "svp_acl.h"
#include "svp_acl_ext.h"
#include "sample_common_svp.h"
#include "sample_common_svp_npu.h"
#include "sample_common_svp_npu_model.h"
#include "ModelProfiler.h"

using namespace VISION_ENGINE;

#define PCBLOGI(fmt, ...) printf("[PCBReID][I] " fmt "\n", ##__VA_ARGS__)
#define PCBLOGE(fmt, ...) printf("[PCBReID][E] " fmt "\n", ##__VA_ARGS__)

/* ================================================================
 * fp16 → float32（与其他 DV500 推理类完全一致）
 * ================================================================ */
/*static*/ float PersonReID_PCB::fp16_to_f32(uint16_t h)
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
 * init
 * 对应原始 PersonReID::init()
 * ================================================================ */
int PersonReID_PCB::init(const PCBReIDConfig& config)
{
    if (inited_) { PCBLOGE("already inited"); return -1; }
    destroy();
    cfg_ = config;

    /* 归一化参数长度校验 */
    // if (cfg_.means.size() < 3 || cfg_.norms.size() < 3) {
    //     PCBLOGE("means/norms must have at least 3 elements");
    //     return -1;
    // }

    /* ---- 1. 加载 .om 模型；runtime 已由 LightTracker 初始化 ---- */
    if (load_model_() != 0) { destroy(); return -1; }
    model_loaded_ = true;

    /* ---- 3. 从模型描述读取输入/输出元信息 ---- */
    sample_svp_npu_model_info* model_info =
        sample_common_svp_npu_get_model_info(MODEL_IDX);
    if (model_info == nullptr || model_info->model_desc == nullptr) {
        PCBLOGE("get model info failed");
        destroy(); return -1;
    }
    svp_acl_mdl_desc* desc = model_info->model_desc;

    /* ---- 3a. 输入尺寸
     * 原始：net_wh_ = Size(input_dim_[3], input_dim_[2])
     *       MNN CAFFE(NCHW): shape=[N,C,H,W] → dims[2]=H, dims[3]=W
     * DV500 NCHW: in_dims.dims[2]=H, in_dims.dims[3]=W  ← 与原始一致
     * ---- */
    svp_acl_mdl_io_dims in_dims{};
    svp_acl_error acl_ret = svp_acl_mdl_get_input_dims(desc, 0, &in_dims);
    if (acl_ret != SVP_ACL_SUCCESS) {
        PCBLOGE("get input dims failed 0x%x", acl_ret);
        destroy(); return -1;
    }
    PCBLOGI("input dims: N=%ld C=%ld H=%ld W=%ld",
            (long)in_dims.dims[0], (long)in_dims.dims[1],
            (long)in_dims.dims[2], (long)in_dims.dims[3]);

    /*
     * 原始：net_wh_ = Size(input_dim_[3], input_dim_[2])  → W=dim[3], H=dim[2]
     * NCHW layout，与 MNN CAFFE tensor shape 一致
     */
    net_wh_ = cv::Size((int)in_dims.dims[3], (int)in_dims.dims[2]);
    PCBLOGI("net_wh = %dx%d (WxH)", net_wh_.width, net_wh_.height);

    /* ---- 3b. 输出维度（特征向量长度）
     * 原始：output_dim_ = output_tensor_->shape()，然后在 postProcess 里连乘
     * ---- */
    svp_acl_mdl_io_dims out_dims{};
    svp_acl_mdl_get_output_dims(desc, 0, &out_dims);

    PCBLOGI("output dims:");
    feat_dim_ = 1;
    for (int d = 0; d < (int)out_dims.dim_count; ++d) {
        PCBLOGI("  dim[%d] = %ld", d, (long)out_dims.dims[d]);
        feat_dim_ *= (int)out_dims.dims[d];
    }
    PCBLOGI("feature dim = %d", feat_dim_);

    /* ---- 3c. 输出 stride ---- */
    size_t default_stride = svp_acl_mdl_get_output_default_stride(desc, 0);
    size_t elem_bytes     = cfg_.output_fp16 ? sizeof(uint16_t) : sizeof(float);
    out_stride_elem_      = (default_stride > 0)
                          ? default_stride / elem_bytes
                          : (size_t)feat_dim_;
    PCBLOGI("output stride=%zu bytes, stride_elem=%zu", default_stride, out_stride_elem_);

    /* ---- 3d. 自动检测输出 dtype ---- */
    svp_acl_data_type out_dtype = svp_acl_mdl_get_output_data_type(desc, 0);
    PCBLOGI("output dtype = %d  (0=fp32, 1=fp16)", (int)out_dtype);
    if (out_dtype == SVP_ACL_FLOAT16 && !cfg_.output_fp16) {
        PCBLOGI("WARNING: model dtype is fp16, auto-correcting config");
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
    PCBLOGI("init ok: %s  input=%dx%d  feat_dim=%d",
            cfg_.model_path.c_str(), net_wh_.width, net_wh_.height, feat_dim_);
    return 0;
}

/* ================================================================
 * run
 * 对应原始 PersonReID::run()，接口完全一致
 * ================================================================ */
int PersonReID_PCB::run(const cv::Mat& image, cv::Rect box, cv::Mat& result)
{
    // auto ts = std::chrono::steady_clock::now();

    if (!inited_) { PCBLOGE("not initialized"); return -1; }
    if (image.empty()) { PCBLOGE("input image is empty"); return -1; }
    fxprof::add_call(fxprof::Model::ReID);

    cv::Mat img;
    int ret = 0;

    {
        fxprof::ScopedPhase _p(fxprof::Model::ReID, fxprof::Phase::Pre);
        ret = preProcess(image, box, img);
    }
    if (ret < 0) {
        PCBLOGE("preProcess failed"); return -1;
    }

    {
        fxprof::ScopedPhase _p(fxprof::Model::ReID, fxprof::Phase::Infer);
        ret = inference(img);
    }
    if (ret < 0) {
        PCBLOGE("inference failed"); return -1;
    }

    {
        fxprof::ScopedPhase _p(fxprof::Model::ReID, fxprof::Phase::Post);
        ret = postProcess(result);
    }
    if (ret < 0) {
        PCBLOGE("postProcess failed"); return -1;
    }

    // auto now = std::chrono::steady_clock::now();
    // double t = std::chrono::duration<double>(now - ts).count();
    // printf("reid time: %.4f ms\n", t * 1000);  /* 与原始 cout 格式对齐 */

    return 0;
}

// int PersonReID_PCB::preProcess(const cv::Mat& src, cv::Rect box, cv::Mat& dst)
// {
//     /* ---- ROI 坐标修正（与原始完全一致）
//      * 原始用 box.width/height 当右下角坐标，严格保留
//      * ---- */
//     int x1 = std::max(box.x,      0);
//     int y1 = std::max(box.y,      0);
//     int x2 = std::min(box.width,  src.cols - 1);
//     int y2 = std::min(box.height, src.rows - 1);

//     if (x1 >= x2 || y1 >= y2) {
//         PCBLOGE("invalid box: (%d,%d)-(%d,%d)", x1, y1, x2, y2);
//         return -1;
//     }

//     /* ---- ROI 裁剪 + resize（与原始完全一致） ---- */
//     dst = src(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();
//     cv::resize(dst, dst, net_wh_, 0, 0, cv::INTER_LINEAR);
//     // cv::INTER_LINEAR cv::INTER_NEAREST

//     /* ---- BGR → RGB（与原始 bConvertBGR_ 逻辑一致） ---- */
//     if (cfg_.bgr2rgb)
//         cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);

//    td_u8  *dev_ptr  = nullptr;
//     td_u32  buf_size = 0, stride = 0;
//     td_s32  ret = sample_common_svp_npu_get_input_data_buffer_info(
//         &task_info_, 0, &dev_ptr, &buf_size, &stride);
//     if (ret != TD_SUCCESS) {
//         PCBLOGE("get input buffer info failed 0x%x", ret);
//         return -1;
//     }

//     int H = dst.rows, W = dst.cols;

//     if (cfg_.input_format == ReIDInputFormat::RGB_PACKAGE) {
//         /* ---- RGB_PACKAGE：HWC 交错，一次 memcpy ---- */
//         td_u32 data_bytes = (td_u32)(H * W * 3);
//         if (data_bytes > buf_size) {
//             PCBLOGE("data %u > buf %u", data_bytes, buf_size);
//             return -1;
//         }

//         if (!dst.isContinuous()) dst = dst.clone();
//         errno_t cp_ret = memcpy_s(dev_ptr, buf_size, dst.data, data_bytes);
//         if (cp_ret != EOK) { PCBLOGE("memcpy_s failed %d", cp_ret); return -1; }

//         svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
//         if (flush_ret != SVP_ACL_SUCCESS) {
//             PCBLOGE("mem flush failed 0x%x", flush_ret); return -1;
//         }

//         /* stride = W * C */
//         td_u32 packed_stride = (td_u32)(W * 3);
//         ret = sample_common_svp_npu_update_input_data_buffer_info(
//             dev_ptr, data_bytes, packed_stride, 0, &task_info_);

//     } else {
//         /* ---- RGB_PLANAR：CHW 平面，按通道分三次 memcpy ---- */
//         td_u32 plane_size = (td_u32)(H * W);
//         td_u32 data_bytes = plane_size * 3;
//         if (data_bytes > buf_size) {
//             PCBLOGE("data %u > buf %u", data_bytes, buf_size);
//             return -1;
//         }

//         std::vector<cv::Mat> channels(3);
//         cv::split(dst, channels);   /* [0]=R, [1]=G, [2]=B（已 bgr2rgb） */

//         td_u8 *ptr = dev_ptr;
//         for (int c = 0; c < 3; ++c) {
//             cv::Mat ch = channels[c].isContinuous()
//                        ? channels[c] : channels[c].clone();
//             errno_t cp_ret = memcpy_s(ptr,
//                                       buf_size - (td_u32)(ptr - dev_ptr),
//                                       ch.data, plane_size);
//             if (cp_ret != EOK) {
//                 PCBLOGE("memcpy_s ch%d failed %d", c, cp_ret); return -1;
//             }
//             ptr += plane_size;
//         }

//         svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
//         if (flush_ret != SVP_ACL_SUCCESS) {
//             PCBLOGE("mem flush failed 0x%x", flush_ret); return -1;
//         }

//         /* stride = W（单通道行宽） */
//         td_u32 planar_stride = (td_u32)W;
//         ret = sample_common_svp_npu_update_input_data_buffer_info(
//             dev_ptr, data_bytes, planar_stride, 0, &task_info_);
//     }

//     if (ret != TD_SUCCESS) {
//         PCBLOGE("update input buffer failed 0x%x", ret);
//         return -1;
//     }
//     return 0;
// }


int PersonReID_PCB::preProcess(const cv::Mat& src, cv::Rect box, cv::Mat& dst)
{
    /* ---- ROI 坐标修正（与原始完全一致） ---- */
    int x1 = std::max(box.x,      0);
    int y1 = std::max(box.y,      0);
    int x2 = std::min(box.width,  src.cols - 1);
    int y2 = std::min(box.height, src.rows - 1);

    if (x1 >= x2 || y1 >= y2) {
        PCBLOGE("invalid box: (%d,%d)-(%d,%d)", x1, y1, x2, y2);
        return -1;
    }

    /* 源必须是 8UC3：否则 resize 输出类型与设备包装 Mat 不符，create() 会另分配
     * 内存而非写进 dev_ptr，导致下发陈旧数据（静默 bug），这里提前拦掉。 */
    if (src.type() != CV_8UC3) {
        PCBLOGE("preProcess expects CV_8UC3 src, got type=%d", src.type());
        return -1;
    }

    /* ---- 先拿设备输入缓冲（cached MMZ，CPU 可直接寻址）：
     *      让 resize/cvtColor 直接写进去，省掉 ROI clone 与一次整图 memcpy_s ---- */
    td_u8  *dev_ptr  = nullptr;
    td_u32  buf_size = 0, stride = 0;
    td_s32  ret = sample_common_svp_npu_get_input_data_buffer_info(
        &task_info_, 0, &dev_ptr, &buf_size, &stride);
    if (ret != TD_SUCCESS) {
        PCBLOGE("get input buffer info failed 0x%x", ret);
        return -1;
    }

    const int W = net_wh_.width;
    const int H = net_wh_.height;

    /* ROI 视图：不 clone（resize 支持非连续/带 ROI 的源） */
    cv::Mat roi = src(cv::Rect(x1, y1, x2 - x1, y2 - y1));

    if (cfg_.input_format == ReIDInputFormat::RGB_PACKAGE) {
        /* ---- RGB_PACKAGE：HWC 交错，紧凑行宽 stride = W*3 ---- */
        const td_u32 data_bytes    = (td_u32)(H * W * 3);
        const td_u32 packed_stride = (td_u32)(W * 3);
        if (data_bytes > buf_size) {
            PCBLOGE("data %u > buf %u", data_bytes, buf_size);
            return -1;
        }

        /* 把设备缓冲直接包成 HWC Mat（行宽 = W*3），resize 原地写入设备：
         * dev_hwc 尺寸/类型与 resize 目标完全一致 → create() 命中、不再另分配，
         * 像素直接落在 dev_ptr 上。cv::Mat 用外部指针构造，不持有/不释放该内存。 */
        cv::Mat dev_hwc(H, W, CV_8UC3, dev_ptr);
        cv::resize(roi, dev_hwc, net_wh_, 0, 0, cv::INTER_LINEAR);
        if (cfg_.bgr2rgb)
            cv::cvtColor(dev_hwc, dev_hwc, cv::COLOR_BGR2RGB);   // 原地，仍在设备缓冲

        /* cached MMZ：CPU 写完必须刷 cache，NPU 才能读到新数据（与原始一致） */
        svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
        if (flush_ret != SVP_ACL_SUCCESS) {
            PCBLOGE("mem flush failed 0x%x", flush_ret); return -1;
        }

        ret = sample_common_svp_npu_update_input_data_buffer_info(
            dev_ptr, data_bytes, packed_stride, 0, &task_info_);

        dst = dev_hwc;   // 预处理结果就在设备缓冲（别名，下次调用前消费完）

    } else {
        /* ---- RGB_PLANAR：CHW 平面。resize 到复用暂存后，split 直接落进设备三平面 ---- */
        const td_u32 plane_size    = (td_u32)(H * W);
        const td_u32 data_bytes    = plane_size * 3;
        const td_u32 planar_stride = (td_u32)W;
        if (data_bytes > buf_size) {
            PCBLOGE("data %u > buf %u", data_bytes, buf_size);
            return -1;
        }

        /* 复用暂存：避免每候选重新分配；split 的源需要一块连续 HWC 图 */
        thread_local cv::Mat stage;
        cv::resize(roi, stage, net_wh_, 0, 0, cv::INTER_LINEAR);
        if (cfg_.bgr2rgb)
            cv::cvtColor(stage, stage, cv::COLOR_BGR2RGB);

        /* 三个单通道 Mat 覆盖在设备缓冲的连续平面上，split 写入即落设备
         * （用 Mat* 数组重载：split 对已匹配尺寸/类型的目标 create() 命中、不另分配） */
        cv::Mat dev_planes[3] = {
            cv::Mat(H, W, CV_8UC1, dev_ptr + 0 * plane_size),
            cv::Mat(H, W, CV_8UC1, dev_ptr + 1 * plane_size),
            cv::Mat(H, W, CV_8UC1, dev_ptr + 2 * plane_size),
        };
        cv::split(stage, dev_planes);

        svp_acl_error flush_ret = svp_acl_rt_mem_flush(dev_ptr, data_bytes);
        if (flush_ret != SVP_ACL_SUCCESS) {
            PCBLOGE("mem flush failed 0x%x", flush_ret); return -1;
        }

        ret = sample_common_svp_npu_update_input_data_buffer_info(
            dev_ptr, data_bytes, planar_stride, 0, &task_info_);

        dst = stage;
    }

    if (ret != TD_SUCCESS) {
        PCBLOGE("update input buffer failed 0x%x", ret);
        return -1;
    }
    return 0;
}

/* ================================================================
 * inference
 *
 * 与原始完全一致的部分：
 *   split(img, channels)
 *   按通道 memcpy 到 CHW buffer（对应原始 nchw_tensor_）
 *
 * DV500 替换 MNN 写入部分：
 *   nchw_tensor_->host<float>() → dev_ptr（float32 CHW）
 *   input_tensor_->copyFromHostTensor → flush + update_input_data_buffer_info
 *   interpreter_->runSession → sample_common_svp_npu_model_execute
 * ================================================================ */
int PersonReID_PCB::inference(const cv::Mat& img)
{
    /* ---- 执行推理（对应原始 interpreter_->runSession）---- */
    td_s32 ret = sample_common_svp_npu_model_execute(&task_info_);
    if (ret != TD_SUCCESS) {
        PCBLOGE("model execute failed 0x%x", ret);
        return -1;
    }

    return 0;
}

/* ================================================================
 * postProcess
 *
 * 与原始完全一致的部分：
 *   len = 连乘 output_dim_（已在 init 时存入 feat_dim_）
 *   Mat feature(1, len, CV_32FC1)
 *   memcpy 特征数据
 *   result = feature / fnorm（L2 归一化）
 *
 * DV500 替换 MNN copyToHostTensor 的部分：
 *   get_output_data_buffer_info + invalidate → 直接读 device buffer
 * ================================================================ */
// int PersonReID_PCB::postProcess(cv::Mat& result)
// {
//     /* ---- 取 output buffer ---- */
//     td_u8  *out_ptr  = nullptr;
//     td_u32  out_size = 0, out_stride = 0;
//     td_s32  ret = sample_common_svp_npu_get_output_data_buffer_info(
//         &task_info_, 0, &out_ptr, &out_size, &out_stride);
//     if (ret != TD_SUCCESS || out_ptr == nullptr) {
//         PCBLOGE("get output buffer failed 0x%x", ret);
//         return -1;
//     }

//     /* ★ invalidate：NPU 写完后 CPU 必须先 invalidate cache 再读 */
//     svp_acl_error inv_ret = svp_acl_rt_mem_invalidate(out_ptr, out_size);
//     if (inv_ret != SVP_ACL_SUCCESS) {
//         PCBLOGE("mem invalidate failed 0x%x", inv_ret);
//         return -1;
//     }

//     if (feat_dim_ <= 0) { PCBLOGE("feat_dim_ not set"); return -1; }

//     /* ---- 读取特征数据（对应原始 memcpy(feature.data, host_output.host<float>(), ...)）---- */
//     cv::Mat feature(1, feat_dim_, CV_32FC1);
//     float  *dst_data = reinterpret_cast<float *>(feature.data);

//     if (cfg_.output_fp16) {
//         /* fp16 输出：逐元素转换 */
//         const uint16_t *src = reinterpret_cast<const uint16_t *>(out_ptr);
//         for (int i = 0; i < feat_dim_; ++i)
//             dst_data[i] = fp16_to_f32(src[i]);
//     } else {
//         /* fp32 输出：直接 memcpy（与原始完全一致） */
//         const float *src = reinterpret_cast<const float *>(out_ptr);
//         if (out_stride_elem_ == (size_t)feat_dim_) {
//             /* 无 padding，一次拷贝 */
//             memcpy(dst_data, src, (size_t)feat_dim_ * sizeof(float));
//         } else {
//             /* 有 padding，只取有效数据 */
//             for (int i = 0; i < feat_dim_; ++i)
//                 dst_data[i] = src[i];
//         }
//     }

//     /* ---- L2 归一化（与原始完全一致）
//      * 原始：float fnorm = norm(feature, NORM_L2); result = feature / fnorm;
//      * ---- */
//     float fnorm = (float)cv::norm(feature, cv::NORM_L2);
//     if (fnorm < 1e-6f) {
//         // PCBLOGE("feature norm near zero");
//         result = feature.clone();
//         return -1;
//     }
//     result = feature / fnorm;   /* 与原始写法完全一致 */

//     return 0;
// }

// int PersonReID_PCB::postProcess(cv::Mat& result)
// {
//     /* ---- 取 output buffer ---- */
//     td_u8  *out_ptr  = nullptr;
//     td_u32  out_size = 0, out_stride = 0;
//     td_s32  ret = sample_common_svp_npu_get_output_data_buffer_info(
//         &task_info_, 0, &out_ptr, &out_size, &out_stride);
//     if (ret != TD_SUCCESS || out_ptr == nullptr) {
//         PCBLOGE("get output buffer failed 0x%x", ret);
//         return -1;
//     }

//     /* ★ invalidate cache */
//     svp_acl_error inv_ret = svp_acl_rt_mem_invalidate(out_ptr, out_size);
//     if (inv_ret != SVP_ACL_SUCCESS) {
//         PCBLOGE("mem invalidate failed 0x%x", inv_ret);
//         return -1;
//     }

//     /* ---- PCB 输出形状: float32[1, 256, 3, 1]
//      * 内存布局(NCHW): [batch][channel][stripe][width]
//      *   C=256 (每个stripe的特征维度)
//      *   H=3   (stripe数量)
//      *   W=1
//      * 内存顺序: stripe0_ch0, stripe0_ch1, ..., stripe0_ch255,
//      *           stripe1_ch0, ..., stripe1_ch255,
//      *           stripe2_ch0, ..., stripe2_ch255
//      * 但 NCHW 实际排列是: ch0_stripe0, ch0_stripe1, ch0_stripe2,
//      *                     ch1_stripe0, ch1_stripe1, ch1_stripe2, ...
//      * 即 out[c][h][0] = out_ptr[c * 3 + h]
//      * ---- */
//     constexpr int STRIPE_NUM  = 3;
//     constexpr int STRIPE_DIM  = 256;
//     constexpr int TOTAL_DIM   = STRIPE_NUM * STRIPE_DIM;  // 768

//     const float *src = reinterpret_cast<const float *>(out_ptr);

//     /* 检查 buffer 大小是否合理 */
//     if (out_size < TOTAL_DIM * sizeof(float)) {
//         PCBLOGE("output buffer too small: %u < %zu", out_size,
//                 TOTAL_DIM * sizeof(float));
//         return -1;
//     }

//     /* ---- 按 stripe 顺序 concat 成 1×768 向量
//      * NCHW 布局下 out[c][h] = src[c * H + h]
//      * 我们需要的 concat 顺序：
//      *   result[h*STRIPE_DIM + c] = src[c * STRIPE_NUM + h]
//      * ---- */
//     cv::Mat feature(1, TOTAL_DIM, CV_32FC1);
//     float  *dst = reinterpret_cast<float *>(feature.data);

//     for (int h = 0; h < STRIPE_NUM; ++h) {
//         for (int c = 0; c < STRIPE_DIM; ++c) {
//             // NCHW: src[n=0][c][h][w=0] = src[c * STRIPE_NUM + h]
//             dst[h * STRIPE_DIM + c] = src[c * STRIPE_NUM + h];
//         }
//     }

//     /* ---- L2 归一化（与原始一致）---- */
//     float fnorm = (float)cv::norm(feature, cv::NORM_L2);
//     if (fnorm < 1e-6f) {
//         result = feature.clone();
//         return -1;
//     }
//     result = feature / fnorm;

//     return 0;
// }


int PersonReID_PCB::postProcess(cv::Mat& result)
{
    td_u8  *out_ptr  = nullptr;
    td_u32  out_size = 0, out_stride = 0;
    td_s32  ret = sample_common_svp_npu_get_output_data_buffer_info(
        &task_info_, 0, &out_ptr, &out_size, &out_stride);
    if (ret != TD_SUCCESS || out_ptr == nullptr) {
        PCBLOGE("get output buffer failed 0x%x", ret);
        return -1;
    }

    svp_acl_error inv_ret = svp_acl_rt_mem_invalidate(out_ptr, out_size);
    if (inv_ret != SVP_ACL_SUCCESS) {
        PCBLOGE("mem invalidate failed 0x%x", inv_ret);
        return -1;
    }

    /* DSR模型输出: float32[1, 512]，简单一维向量，无stripe结构 */
    if (feat_dim_ <= 0) { PCBLOGE("feat_dim_ not set"); return -1; }

    cv::Mat feature(1, feat_dim_, CV_32FC1);
    float *dst_data = reinterpret_cast<float *>(feature.data);

    if (cfg_.output_fp16) {
        const uint16_t *src = reinterpret_cast<const uint16_t *>(out_ptr);
        for (int i = 0; i < feat_dim_; ++i)
            dst_data[i] = fp16_to_f32(src[i]);
    } else {
        const float *src = reinterpret_cast<const float *>(out_ptr);
        if (out_stride_elem_ == (size_t)feat_dim_) {
            memcpy(dst_data, src, (size_t)feat_dim_ * sizeof(float));
        } else {
            for (int i = 0; i < feat_dim_; ++i)
                dst_data[i] = src[i];
        }
    }

    /* L2归一化，配合余弦相似度比对 */
    float fnorm = (float)cv::norm(feature, cv::NORM_L2);
    if (fnorm < 1e-6f) {
        result = feature.clone();
        return -1;
    }
    result = feature / fnorm;

    return 0;
}

/* ================================================================
 * destroy
 * 对应原始析构：releaseModel + releaseSession + delete nchw_tensor_
 * ================================================================ */
void PersonReID_PCB::destroy()
{
    if (task_initialized_) {
        deinit_task_();
        task_initialized_ = false;
    }
    if (model_loaded_) {
        unload_model_();
        model_loaded_ = false;
    }

    inited_   = false;
    feat_dim_ = 0;
    // PCBLOGI("destroyed");
}

/* ================================================================
 * 模型加载 / 卸载
 * ================================================================ */
int PersonReID_PCB::load_model_()
{
    td_s32 ret = sample_common_svp_npu_load_model(
        cfg_.model_path.c_str(), MODEL_IDX, TD_FALSE);
    if (ret != TD_SUCCESS) {
        PCBLOGE("load model failed %s 0x%x", cfg_.model_path.c_str(), ret);
        return -1;
    }
    // PCBLOGI("model loaded: %s", cfg_.model_path.c_str());
    return 0;
}

void PersonReID_PCB::unload_model_()
{
    sample_common_svp_npu_unload_model(MODEL_IDX);
}

/* ================================================================
 * Task 初始化 / 反初始化
 * ================================================================ */
int PersonReID_PCB::init_task_()
{
    td_s32 ret;

    ret = sample_common_svp_npu_create_input(&task_info_);
    if (ret != TD_SUCCESS) {
        PCBLOGE("create input failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_output(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_input(&task_info_);
        PCBLOGE("create output failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_task_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_output(&task_info_);
        sample_common_svp_npu_destroy_input (&task_info_);
        PCBLOGE("create task buf failed 0x%x", ret); return -1;
    }

    ret = sample_common_svp_npu_create_work_buf(&task_info_);
    if (ret != TD_SUCCESS) {
        sample_common_svp_npu_destroy_task_buf(&task_info_);
        sample_common_svp_npu_destroy_output  (&task_info_);
        sample_common_svp_npu_destroy_input   (&task_info_);
        PCBLOGE("create work buf failed 0x%x", ret); return -1;
    }

    // PCBLOGI("task init ok");
    return 0;
}

void PersonReID_PCB::deinit_task_()
{
    sample_common_svp_npu_destroy_work_buf(&task_info_);
    sample_common_svp_npu_destroy_task_buf(&task_info_);
    sample_common_svp_npu_destroy_output  (&task_info_);
    sample_common_svp_npu_destroy_input   (&task_info_);
    memset_s(&task_info_, sizeof(task_info_), 0, sizeof(task_info_));
}

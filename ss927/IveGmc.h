/* ============================================================================
 * IveGmc.h —— GMC 前半段的海思 IVE 硬件实现（SS92x / ot_/ss_ API）
 *
 * 用途：把 GMC 里最吃 CPU 的"灰度下采样 → 金字塔 → Shi-Tomasi 角点 → LK 金字塔
 *       光流"整段卸载到 IVE 硬件；仿射拟合(estimateAffinePartial2D)仍在 CPU。
 *       任一步失败返回 false，调用方自动回退到原 OpenCV 实现。
 *
 * 仅当定义 USE_HISI_IVE 时编译。需把本文件的 .cpp 加入构建，并链接
 * ss_mpi_ive / ss_mpi_sys，且应用已完成 MPP 系统初始化（ss_mpi_sys_init 等）。
 *
 * 接口约定：输出的匹配点对在"工作分辨率"坐标系；调用方对其跑
 * estimateAffinePartial2D 得到 M_work，再把平移分量除以 work_scale 还原到原图
 * （与现有 OpenCV 实现的 `M.at(0,2)/=scale` 完全一致）。
 * ==========================================================================*/
#ifndef IVE_GMC_H
#define IVE_GMC_H

#ifdef USE_HISI_IVE

#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>

class IveGmc {
public:
    IveGmc();
    ~IveGmc();

    // 分配持久 MMZ 缓冲（金字塔/角点/点数组等）。work_w/h 为工作分辨率
    // （建议 ≈ 现有 kGmcWorkWidth=640 对应的宽高，16 对齐内部自动处理）。
    // 幂等：重复调用先 release 再分配。失败返回 false。
    bool init(int work_w, int work_h);
    void release();
    bool ready() const { return inited_; }

    // —— BGR 输入路径（与现有 cv::Mat 流程兼容，无需改取流）——
    //   CPU 转灰度+缩放到工作分辨率后上传，IVE 跑金字塔/角点/光流。
    //   exclude_boxes_xyxy：原图坐标的前景框（落入其中的角点会被剔除），可空。
    //   prev/cur_work_pts：工作分辨率坐标的匹配点对；work_scale：work = orig*scale。
    //   返回 false = 本帧无可用结果（首帧/特征不足/失败）→ 调用方回退。
    bool match_bgr(const cv::Mat& bgr,
                   const std::vector<cv::Rect>& exclude_boxes_xyxy,
                   std::vector<cv::Point2f>& prev_work_pts,
                   std::vector<cv::Point2f>& cur_work_pts,
                   float& work_scale);

    // —— YUV420SP 零拷贝快路（推荐）——
    //   直接以帧缓冲 Y 平面（已在 MMZ 物理连续内存）作灰度，省掉转灰度+上传。
    //   y_phys/y_virt：Y 平面物理/虚拟地址；w/h：原图尺寸；y_stride：Y 行跨距。
    bool match_yuv420sp(uint64_t y_phys, uint64_t y_virt, int w, int h, int y_stride,
                        const std::vector<cv::Rect>& exclude_boxes_xyxy,
                        std::vector<cv::Point2f>& prev_work_pts,
                        std::vector<cv::Point2f>& cur_work_pts,
                        float& work_scale);

private:
    struct Mmz { uint64_t phys = 0; uint64_t virt = 0; uint32_t size = 0; };

    bool alloc_mmz(Mmz& b, uint32_t size, const char* tag);
    void free_mmz(Mmz& b);

    // cur_pyr[0] 已填好后：构建 1..max_level 层 + （有 prev 时）角点→LK→配对。
    bool process_after_level0(const std::vector<cv::Rect>& exclude_boxes_xyxy,
                              std::vector<cv::Point2f>& prev_work_pts,
                              std::vector<cv::Point2f>& cur_work_pts);

    bool ive_resize_u8c1(const Mmz& src, int sw, int sh, int sstride,
                         const Mmz& dst, int dw, int dh, int dstride);

    // 金字塔层数 = kMaxLevel+1。取 2（3 层：640×360→320×180→160×90）使每层宽高
    //   均为偶数且严格半分，同时满足 IVE resize/st_cand_corner（偶数）与 LK（严格半分）
    //   两类约束；work_h=360 无法支撑 4 层偶数半分（需被 16 整除）。lk.max_level 随此变。
    static const int kMaxLevel = 2;          // 金字塔层数 = kMaxLevel+1
    static const int kPyrLevels = kMaxLevel + 1;
    static const int kMaxCorners = 500;      // OT_IVE_ST_MAX_CORNER_NUM

    bool inited_ = false;
    bool has_prev_ = false;
    int work_w_ = 0, work_h_ = 0;            // 工作分辨率（实际宽高）
    int work_stride_ = 0;                    // 16 对齐后的行跨距
    float work_scale_ = 1.f;                 // work = orig * work_scale_（按宽计算）

    // 金字塔每层尺寸/跨距（level 0 = work_w_/h_，逐层 /2）
    int lvl_w_[kPyrLevels] = {0};
    int lvl_h_[kPyrLevels] = {0};
    int lvl_stride_[kPyrLevels] = {0};

    Mmz prev_pyr_[kPyrLevels];
    Mmz cur_pyr_[kPyrLevels];
    Mmz full_gray_;     // BGR 路径：工作分辨率灰度上传缓冲（= cur_pyr[0] 直接用，见 .cpp）
    Mmz eig_;           // st_cand_corner 输出（工作分辨率 U8C1）
    Mmz st_assist_;     // st_cand_corner 辅助内存
    Mmz resize_assist_; // resize 辅助内存
    Mmz corner_mem_;    // ot_ive_st_corner_info
    Mmz prev_pts_;      // ot_svp_point_s25q7 × kMaxCorners
    Mmz next_pts_;      // ot_svp_point_s25q7 × kMaxCorners
    Mmz status_;        // u8 × kMaxCorners
    Mmz err_;           // u16 × kMaxCorners
};

#endif /* USE_HISI_IVE */
#endif /* IVE_GMC_H */

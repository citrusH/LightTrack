/* ============================================================================
 * IveGmc.cpp —— GMC 前半段的海思 IVE 实现（SS92x / ss_/ot_ API）
 *
 * 前提：应用已完成 MPP 系统初始化（ss_mpi_sys_init / VB 池）。链接 ss_mpi_ive、
 *       ss_mpi_sys。仅在 USE_HISI_IVE 时编译。
 *
 * 设计与正确性说明：
 *  - 内存统一用"非缓存" ss_mpi_sys_mmz_alloc：CPU 与 IVE 共享时无需手动 flush/
 *    invalidate（正确优先）。若要提速可改 ss_mpi_sys_mmz_alloc_cached + 在 CPU 读
 *    设备产物前 invalidate、CPU 写设备输入后 flush（需对应 cache MPI）。
 *  - LK 点为定点 ot_svp_point_s25q7：像素值 ×128 写入，回读 /128。
 *  - 角点检测在"上一帧"金字塔 level0 上做（与 OpenCV goodFeaturesToTrack(prev) 一致），
 *    再 LK prev→cur；处理完把 cur 金字塔交换为下一帧的 prev。
 *  - 输出点对在"工作分辨率"坐标系；调用方对其做 estimateAffinePartial2D，再把平移
 *    分量除以 work_scale 还原到原图（与现有 OpenCV 路径完全一致）。
 *
 * ⚠ 需在真机按 IVE 开发指南/实测核对的少量点（已就近标注 ADAPT）：
 *    1) 各辅助内存(st_assist_/resize_assist_)的"精确"大小公式（这里按工作分辨率给了
 *       较宽裕的估值，过小会返回错误码）。
 *    2) 几个 ctrl 阈值（IVE_ST_QUALITY/MIN_DIST/LK_*）按角点数量与跟踪质量微调。
 *    3) td_bool/TD_TRUE/TD_SUCCESS 等基础宏名（应在 ot_type.h/ot_errno.h）。
 * ==========================================================================*/
#include "IveGmc.h"

#ifdef USE_HISI_IVE

#include "ot_common_svp.h"
#include "ot_common_ive.h"
#include "ss_mpi_ive.h"
#include "ss_mpi_sys.h"        // ss_mpi_sys_mmz_alloc/_free 在此声明（grep 已确认 ss_mpi_sys.h:49）
#include "ModelProfiler.h"     // GMC 分阶段计时（ScopedGmc，纯头文件，与帧统计同开关）

#include <cstring>
#include <cstdint>
#include <cstddef>     // offsetof（DIAG2 探测 mem_info 布局/ABI）
#include <cmath>
#include <algorithm>
#include <iostream>

// ── 可调参数（器件调参起点）────────────────────────────────────────────
#define IVE_ST_QUALITY     ((td_u0q8)13)   // ADAPT: 角点质量(≈0.05*256)，越大越严
#define IVE_ST_MIN_DIST    ((td_u16)8)      // 角点最小间距（对应 OpenCV minDistance）
#define IVE_LK_MIN_EIG     ((td_u0q8)2)     // LK 最小特征值阈值
#define IVE_LK_ITER_CNT    ((td_u8)10)      // LK 最大迭代次数(≤20)
#define IVE_LK_EPS         ((td_u0q8)3)     // LK 收敛 eps(≈0.01*256)
#define IVE_MIN_TRACK_PTS  8                // 少于此对匹配点 → 视为失败，回退

static inline int align16(int x) { return (x + 15) & ~15; }

static inline void fill_u8c1(ot_svp_img& img, uint64_t phys, uint64_t virt,
                             int w, int h, int stride) {
    std::memset(&img, 0, sizeof(img));
    img.type        = OT_SVP_IMG_TYPE_U8C1;
    img.width       = (td_u32)w;
    img.height      = (td_u32)h;
    img.stride[0]   = (td_u32)stride;
    img.phys_addr[0] = (td_u64)phys;
    img.virt_addr[0] = (td_u64)virt;
}

static inline void fill_mem(ot_svp_mem_info& m, uint64_t phys, uint64_t virt, uint32_t size) {
    m.phys_addr = (td_u64)phys;
    m.virt_addr = (td_u64)virt;
    m.size      = (td_u32)size;
}

static bool ive_wait(ot_ive_handle h) {
    td_bool fin = TD_FALSE;
    // 阻塞查询直到该任务完成
    td_s32 ret = ss_mpi_ive_query(h, &fin, TD_TRUE);
    return (ret == TD_SUCCESS) && (fin == TD_TRUE);
}

// ────────────────────────────────────────────────────────────────────────
IveGmc::IveGmc() {}
IveGmc::~IveGmc() { release(); }

bool IveGmc::alloc_mmz(Mmz& b, uint32_t size, const char* tag) {
    td_phys_addr_t phys = 0;
    td_void* virt = nullptr;
    td_s32 ret = ss_mpi_sys_mmz_alloc(&phys, &virt, nullptr, nullptr, size);
    if (ret != TD_SUCCESS || virt == nullptr) {
        std::cout << "[IVE_GMC] mmz_alloc failed tag=" << (tag ? tag : "?")
                  << " size=" << size << " ret=" << ret << std::endl;
        return false;
    }
    b.phys = (uint64_t)phys;
    b.virt = (uint64_t)(uintptr_t)virt;
    b.size = size;
    std::memset((void*)(uintptr_t)b.virt, 0, size);
    return true;
}

void IveGmc::free_mmz(Mmz& b) {
    if (b.virt) {
        ss_mpi_sys_mmz_free((td_phys_addr_t)b.phys, (td_void*)(uintptr_t)b.virt);
    }
    b = Mmz{};
}

bool IveGmc::init(int work_w, int work_h) {
    release();
    if (work_w < 32 || work_h < 32) return false;

    // 探针/兜底：确保 MPP 系统已初始化（IVE 依赖之）。IveGmc 设计上假定 app 已
    //   调用 ss_mpi_sys_init；此处再调一次是安全的——若已初始化通常返回成功/已存在。
    //   IVE resize 报 NOT_CONFIG(errid 0x7) 正是"MPP 未初始化"的特征，用本探针判定：
    //     ret==0 且之前 resize 失败 → 说明 app 此前没 init，本调用补上了 → 应转好；
    //     ret!=0（已存在等）→ MPP 早已初始化，NOT_CONFIG 另有原因（多半 VB 池未配）。
    td_s32 sys_ret = ss_mpi_sys_init();
    std::cout << "[IVE_GMC] ss_mpi_sys_init ret=" << sys_ret << std::endl;

    // 工作分辨率强制偶数（IVE 偶数对齐；非 16:9 宽高比可能算出奇数高）。
    //   level0 = work 尺寸，且 match_bgr 上传按 work_w_/work_h_ 逐行拷贝 → 必须与
    //   lvl_w_[0]/lvl_h_[0] 一致，故在源头取偶，避免 1 行/列错位。
    work_w_ = work_w & ~1;
    work_h_ = work_h & ~1;
    work_stride_ = align16(work_w_);

    // 金字塔每层尺寸（逐层 /2，下限 16）
    //   ⚠ IVE resize 要求宽/高均为偶数（块/色度对齐）：逐层右移会产生奇数高
    //   （360→180→90→45，最深层 45 为奇）→ ss_mpi_ive_resize 返回 ILLEGAL_PARAM(0x7)。
    //   故每层 w/h 向下取偶（& ~1）；下限 16 本身为偶。
    for (int l = 0; l < kPyrLevels; ++l) {
        int w = std::max(16, work_w_ >> l) & ~1;
        int h = std::max(16, work_h_ >> l) & ~1;
        lvl_w_[l] = w; lvl_h_[l] = h; lvl_stride_[l] = align16(w);
    }

    bool ok = true;
    for (int l = 0; l < kPyrLevels && ok; ++l) {
        uint32_t sz = (uint32_t)lvl_stride_[l] * lvl_h_[l];
        ok = ok && alloc_mmz(prev_pyr_[l], sz, "prev_pyr");
        ok = ok && alloc_mmz(cur_pyr_[l],  sz, "cur_pyr");
    }
    uint32_t work_sz = (uint32_t)work_stride_ * work_h_;
    ok = ok && alloc_mmz(eig_, work_sz, "eig");
    // st_cand_corner 辅助内存（强制公式，否则 ILLEGAL_PARAM 0x7）：
    //   mem.size = 4 * align16(src_w) * src_h + sizeof(ot_ive_st_max_eig_val)
    //   src = prev_pyr_[0] = level0 = work_w_×work_h_；align16(work_w_)=work_stride_。
    //   原代码只给 4*work_sz，漏了 + sizeof(ot_ive_st_max_eig_val) 尾部 → 偏小且不匹配。
    uint32_t st_mem = 4u * (uint32_t)work_stride_ * (uint32_t)work_h_
                    + (uint32_t)sizeof(ot_ive_st_max_eig_val);
    ok = ok && alloc_mmz(st_assist_,     st_mem,   "st_assist");
    ok = ok && alloc_mmz(resize_assist_, work_sz,  "resize_assist");
    // st_corner 输出缓冲：精确所需大小的公式只在驱动二进制里（头文件/doc 均未给出）。
    //   实测 sizeof(ot_ive_st_corner_info)=2002 与 ~4080 都被拒（依次报 size(0)/size(1)），
    //   说明驱动要求的容量更大且按"分段"校验，单纯 >=2002 解释不通。故先给足量缓冲（64KB，
    //   远超 500 角点任何合理布局）做一次性诊断兼修复：跨过容量校验 → 打通 st_corner→LK→
    //   仿射全链路；若 64KB 仍报 size(1)，则证明并非容量问题（另寻设置/传参根因）。
    //   链路确认后再按实测最小值右值化。size>= 校验下缓冲偏大无副作用。
    uint32_t corner_info_sz = 64u * 1024u;
    // 诊断：打印本编译单元里实际的类型大小。厂商样例用 sizeof(ot_ive_st_corner_info)（≈2002）
    //   即可，而我方原先同样用 sizeof 却被拒 size(0)<2002 → 强烈怀疑本 .cpp 包含的头里
    //   ot_svp_point_u16 / OT_IVE_ST_MAX_CORNER_NUM 与 grep 的那份不一致（多 SDK 树）。
    std::cout << "[IVE_GMC] DIAG sizeof(ot_ive_st_corner_info)=" << sizeof(ot_ive_st_corner_info)
              << " sizeof(ot_svp_point_u16)=" << sizeof(ot_svp_point_u16)
              << " OT_IVE_ST_MAX_CORNER_NUM=" << (int)OT_IVE_ST_MAX_CORNER_NUM
              << " kMaxCorners=" << kMaxCorners
              << " corner_buf=" << corner_info_sz << std::endl;
    ok = ok && alloc_mmz(corner_mem_, corner_info_sz, "corner");
    ok = ok && alloc_mmz(prev_pts_, kMaxCorners * (uint32_t)sizeof(ot_svp_point_s25q7), "prev_pts");
    ok = ok && alloc_mmz(next_pts_, kMaxCorners * (uint32_t)sizeof(ot_svp_point_s25q7), "next_pts");
    ok = ok && alloc_mmz(status_,   kMaxCorners,                       "status");
    ok = ok && alloc_mmz(err_,      kMaxCorners * (uint32_t)sizeof(td_u16), "err");

    if (!ok) { release(); return false; }
    has_prev_ = false;
    inited_ = true;
    return true;
}

void IveGmc::release() {
    for (int l = 0; l < kPyrLevels; ++l) { free_mmz(prev_pyr_[l]); free_mmz(cur_pyr_[l]); }
    free_mmz(full_gray_); free_mmz(eig_); free_mmz(st_assist_); free_mmz(resize_assist_);
    free_mmz(corner_mem_); free_mmz(prev_pts_); free_mmz(next_pts_);
    free_mmz(status_); free_mmz(err_);
    inited_ = false; has_prev_ = false;
}

// 按官方文档计算 ss_mpi_ive_resize 所需辅助内存（ctrl->mem）字节数。
//   total_mem = total_num * assist_unit_size，total_num = comp_num * hor_block * ver_block。
//   板端 assist_unit_size = 48（PC 仿真 = 56）；U8C1：comp_num=1, metric=2032。
//   放大/等长：block = ceil(dim/metric)；缩小：按文档分块公式（dst_tile 推导）。
//   ⚠ IVE 要求 ctrl->mem.size 恰为此值；传整缓冲大小等不匹配值 → ILLEGAL_PARAM(errid 0x7)。
static uint32_t ive_resize_assist_size(int src_w, int src_h, int dst_w, int dst_h,
                                       int comp_num, int metric, int assist_unit_size) {
    auto ceil_div = [](int a, int b) -> int { return b > 0 ? (a / b + (a % b ? 1 : 0)) : 1; };
    auto blocks = [&](int sdim, int ddim) -> int {
        if (ddim >= sdim)                                    // 放大/等长
            return ceil_div(ddim, metric);
        int src_tile_num = ceil_div(sdim, metric);           // 缩小（文档公式）
        int src_tmp_tile = ceil_div(sdim, src_tile_num);
        int scale        = ceil_div(sdim, ddim);
        // & 优先级低于 +：掩码作用于整个和（与文档/HiSilicon 写法一致）
        int dst_tile     = ((ddim / src_tile_num + (metric - src_tmp_tile) / scale) & ~0x1) - 2;
        if (dst_tile < 1) dst_tile = 1;
        return ceil_div(ddim, dst_tile);
    };
    uint32_t total_num = (uint32_t)comp_num
                       * (uint32_t)blocks(src_w, dst_w)
                       * (uint32_t)blocks(src_h, dst_h);
    if (total_num < 1) total_num = 1;
    return total_num * (uint32_t)assist_unit_size;
}

bool IveGmc::ive_resize_u8c1(const Mmz& src, int sw, int sh, int sstride,
                             const Mmz& dst, int dw, int dh, int dstride) {
    fxprof::ScopedGmc _g(fxprof::GmcStage::Resize);   // 计入 IVE resize（提交+阻塞等待）
    ot_svp_src_img s[1]; ot_svp_dst_img d[1];
    fill_u8c1(s[0], src.phys, src.virt, sw, sh, sstride);
    fill_u8c1(d[0], dst.phys, dst.virt, dw, dh, dstride);
    ot_ive_resize_ctrl ctrl;
    std::memset(&ctrl, 0, sizeof(ctrl));
    // 本芯片 IVE 只有 OT_IVE_RESIZE_MODE_LINEAR 一种模式（见 ot_common_ive.h 的
    //   ot_ive_resize_mode 枚举：仅 LINEAR=0x0 + BUTT，无 AREA）。原代码用 AREA（非法
    //   枚举值）→ 引擎返回 ILLEGAL_PARAM(errid 0x7)。金字塔 /2 下采样用双线性完全够用。
    ctrl.mode = OT_IVE_RESIZE_MODE_LINEAR;
    ctrl.num  = 1;
    // ── 辅助内存大小必须恰为文档公式值（U8C1: comp_num=1, metric=2032, 板端 unit=48）──
    //   原先传整缓冲大小(work_stride*work_h≈230400) ≠ 公式值 → ILLEGAL_PARAM(0x7)。
    uint32_t need = ive_resize_assist_size(sw, sh, dw, dh, /*comp*/1, /*metric*/2032, /*unit*/48);
    if (need > resize_assist_.size) {                 // 缓冲不足（理论上不会，公式值很小）
        std::cout << "[IVE_GMC] resize assist under-alloc: need=" << need
                  << " have=" << resize_assist_.size << std::endl;
        return false;
    }
    fill_mem(ctrl.mem, resize_assist_.phys, resize_assist_.virt, need);
    ot_ive_handle h;
    td_s32 ret = ss_mpi_ive_resize(&h, s, d, &ctrl, TD_TRUE);
    if (ret != TD_SUCCESS) {
        std::cout << "[IVE_GMC] resize ret=" << ret
                  << " src(w=" << sw << " h=" << sh << " stride=" << sstride
                  << " phys=0x" << std::hex << src.phys << std::dec << ")"
                  << " dst(w=" << dw << " h=" << dh << " stride=" << dstride
                  << " phys=0x" << std::hex << dst.phys << std::dec << ")"
                  << " mem(need=" << need << " have=" << resize_assist_.size
                  << " phys=0x" << std::hex << resize_assist_.phys << std::dec << ")"
                  << " mode=" << (int)ctrl.mode << " num=" << ctrl.num << std::endl;
        return false;
    }
    return ive_wait(h);
}

bool IveGmc::process_after_level0(const std::vector<cv::Rect>& boxes,
                                  std::vector<cv::Point2f>& prev_work_pts,
                                  std::vector<cv::Point2f>& cur_work_pts) {
    prev_work_pts.clear();
    cur_work_pts.clear();

    // 1) 构建 cur 金字塔 1..max_level
    for (int l = 1; l < kPyrLevels; ++l) {
        if (!ive_resize_u8c1(cur_pyr_[l - 1], lvl_w_[l - 1], lvl_h_[l - 1], lvl_stride_[l - 1],
                             cur_pyr_[l],     lvl_w_[l],     lvl_h_[l],     lvl_stride_[l]))
            return false;
    }

    // 首帧：无 prev，存为 prev 后返回 false（与 OpenCV 首帧不补偿一致）
    if (!has_prev_) {
        for (int l = 0; l < kPyrLevels; ++l) std::swap(prev_pyr_[l], cur_pyr_[l]);
        has_prev_ = true;
        return false;
    }

    // 2) 角点：st_cand_corner(prev level0) → eig，st_corner → corner_info
    {
        ot_svp_src_img src; fill_u8c1(src, prev_pyr_[0].phys, prev_pyr_[0].virt,
                                      lvl_w_[0], lvl_h_[0], lvl_stride_[0]);
        ot_svp_dst_img cand; fill_u8c1(cand, eig_.phys, eig_.virt,
                                       work_w_, work_h_, work_stride_);
        ot_ive_st_cand_corner_ctrl cc;
        std::memset(&cc, 0, sizeof(cc));
        cc.quality_level = IVE_ST_QUALITY;
        fill_mem(cc.mem, st_assist_.phys, st_assist_.virt, st_assist_.size);
        ot_ive_handle h;
        td_s32 ret;
        bool waited;
        {
            fxprof::ScopedGmc _g(fxprof::GmcStage::StCand);   // IVE 提交+阻塞等待
            ret = ss_mpi_ive_st_cand_corner(&h, &src, &cand, &cc, TD_TRUE);
            waited = (ret == TD_SUCCESS) && ive_wait(h);      // 短路：ret!=SUCCESS 时不 wait（同原行为）
        }
        if (!waited) {
            std::cout << "[IVE_GMC] st_cand_corner ret=" << ret << std::endl;
            for (int l = 0; l < kPyrLevels; ++l) std::swap(prev_pyr_[l], cur_pyr_[l]);
            return false;
        }
        ot_svp_dst_mem_info cmem; fill_mem(cmem, corner_mem_.phys, corner_mem_.virt, corner_mem_.size);
        ot_ive_st_corner_ctrl sc;
        std::memset(&sc, 0, sizeof(sc));
        sc.max_corner_num = (td_u16)kMaxCorners;
        sc.min_dist       = IVE_ST_MIN_DIST;
        // DIAG2：容量已排除（64KB 仍报 corner->size<2002）。打印调用点 cmem 实际字段 + mem_info
        //   布局 + 候选图，定位驱动究竟读到什么。若此处 cmem.size=65536 而驱动仍说 0/1，
        //   即证实 ot_svp_mem_info 头/.so ABI 不一致（.size 偏移错位）。仅打印前 2 帧避免刷屏。
        {
            static int diag2 = 0;
            if (diag2++ < 2) {
                std::cout << "[IVE_GMC] DIAG2 cmem.size=" << cmem.size
                          << " phys=0x" << std::hex << (uint64_t)cmem.phys_addr
                          << " virt=0x" << (uint64_t)cmem.virt_addr << std::dec
                          << " | sizeof(mem_info)=" << (int)sizeof(ot_svp_dst_mem_info)
                          << " off(phys/virt/size)=" << (int)offsetof(ot_svp_mem_info, phys_addr)
                          << "/" << (int)offsetof(ot_svp_mem_info, virt_addr)
                          << "/" << (int)offsetof(ot_svp_mem_info, size)
                          << " | cand(w=" << cand.width << " h=" << cand.height
                          << " stride=" << cand.stride[0] << " type=" << (int)cand.type << ")"
                          << std::endl;
            }
        }
        {
            fxprof::ScopedGmc _g(fxprof::GmcStage::StCorner);
            ret = ss_mpi_ive_st_corner(&cand, &cmem, &sc);   // 同步
        }
        if (ret != TD_SUCCESS) {
            std::cout << "[IVE_GMC] st_corner ret=" << ret << std::endl;
            for (int l = 0; l < kPyrLevels; ++l) std::swap(prev_pyr_[l], cur_pyr_[l]);
            return false;
        }
    }

    // 3) 读角点 + 前景框过滤 + 写 prev_pts(s25q7)
    ot_ive_st_corner_info* ci = (ot_ive_st_corner_info*)(uintptr_t)corner_mem_.virt;
    int n = (int)ci->corner_num;
    if (n > kMaxCorners) n = kMaxCorners;

    // 前景框（原图 xyxy）映射到工作坐标
    struct B { float x1, y1, x2, y2; };
    std::vector<B> wb; wb.reserve(boxes.size());
    for (const auto& r : boxes) {
        wb.push_back({ r.x * work_scale_, r.y * work_scale_,
                       r.width * work_scale_, r.height * work_scale_ });
    }
    auto in_box = [&](float x, float y) {
        for (const auto& b : wb)
            if (x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2) return true;
        return false;
    };

    ot_svp_point_s25q7* pp = (ot_svp_point_s25q7*)(uintptr_t)prev_pts_.virt;
    std::vector<cv::Point2f> kept;  // 工作坐标
    kept.reserve(n);
    {
        fxprof::ScopedGmc _g(fxprof::GmcStage::CornerRead);   // CPU 读角点+前景过滤+写定点
        for (int i = 0; i < n; ++i) {
            float x = (float)ci->corner[i].x;
            float y = (float)ci->corner[i].y;
            if (in_box(x, y)) continue;
            int k = (int)kept.size();
            pp[k].x = (td_s25q7)std::lround(x * 128.0);   // ×2^7
            pp[k].y = (td_s25q7)std::lround(y * 128.0);
            kept.push_back(cv::Point2f(x, y));
            if (k + 1 >= kMaxCorners) break;
        }
    }
    int pts_num = (int)kept.size();
    if (pts_num < IVE_MIN_TRACK_PTS) {
        for (int l = 0; l < kPyrLevels; ++l) std::swap(prev_pyr_[l], cur_pyr_[l]);
        return false;
    }

    // 4) LK 金字塔光流 prev→cur
    {
        fxprof::ScopedGmc _g(fxprof::GmcStage::LK);   // 计入 LK（设置+提交+阻塞等待）
        ot_svp_src_img prev_arr[kPyrLevels], cur_arr[kPyrLevels];
        for (int l = 0; l < kPyrLevels; ++l) {
            fill_u8c1(prev_arr[l], prev_pyr_[l].phys, prev_pyr_[l].virt,
                      lvl_w_[l], lvl_h_[l], lvl_stride_[l]);
            fill_u8c1(cur_arr[l],  cur_pyr_[l].phys,  cur_pyr_[l].virt,
                      lvl_w_[l], lvl_h_[l], lvl_stride_[l]);
        }
        ot_svp_src_mem_info ppm; fill_mem(ppm, prev_pts_.phys, prev_pts_.virt,
                                          pts_num * (uint32_t)sizeof(ot_svp_point_s25q7));
        ot_svp_mem_info     npm; fill_mem(npm, next_pts_.phys, next_pts_.virt,
                                          pts_num * (uint32_t)sizeof(ot_svp_point_s25q7));
        ot_svp_dst_mem_info stm; fill_mem(stm, status_.phys, status_.virt, (uint32_t)pts_num);
        ot_svp_dst_mem_info erm; fill_mem(erm, err_.phys, err_.virt,
                                          pts_num * (uint32_t)sizeof(td_u16));
        ot_ive_lk_optical_flow_pyr_ctrl lk;
        std::memset(&lk, 0, sizeof(lk));
        lk.out_mode             = OT_IVE_LK_OPTICAL_FLOW_PYR_OUT_MODE_STATUS;
        lk.use_init_flow        = TD_FALSE;
        lk.points_num           = (td_u16)pts_num;
        lk.max_level            = (td_u8)kMaxLevel;
        lk.min_eig_val_threshold = IVE_LK_MIN_EIG;
        lk.iter_cnt             = IVE_LK_ITER_CNT;
        lk.eps                  = IVE_LK_EPS;
        ot_ive_handle h;
        td_s32 ret = ss_mpi_ive_lk_optical_flow_pyr(&h, prev_arr, cur_arr, &ppm, &npm,
                                                    &stm, &erm, &lk, TD_TRUE);
        if (ret != TD_SUCCESS || !ive_wait(h)) {
            std::cout << "[IVE_GMC] lk_optical_flow ret=" << ret
                      << " | ctrl: out_mode=" << (int)lk.out_mode
                      << " use_init=" << (int)lk.use_init_flow
                      << " points_num=" << (int)lk.points_num
                      << " max_level=" << (int)lk.max_level
                      << " min_eig=" << (int)lk.min_eig_val_threshold
                      << " iter=" << (int)lk.iter_cnt
                      << " eps=" << (int)lk.eps
                      << " | pts_num=" << pts_num
                      << " sizeof(pt)=" << (int)sizeof(ot_svp_point_s25q7)
                      << " | lv0=" << lvl_w_[0] << "x" << lvl_h_[0] << " s=" << lvl_stride_[0]
                      << " lv1=" << lvl_w_[1] << "x" << lvl_h_[1] << " s=" << lvl_stride_[1]
                      << " lv2=" << lvl_w_[2] << "x" << lvl_h_[2] << " s=" << lvl_stride_[2]
                      << " | mem: ppm=" << ppm.size << " npm=" << npm.size
                      << " stm=" << stm.size << " erm=" << erm.size
                      << " | phys(align16?): p0=0x" << std::hex << prev_pyr_[0].phys
                      << " p1=0x" << prev_pyr_[1].phys << " p2=0x" << prev_pyr_[2].phys
                      << " c0=0x" << cur_pyr_[0].phys
                      << " c1=0x" << cur_pyr_[1].phys << " c2=0x" << cur_pyr_[2].phys
                      << " pp=0x" << prev_pts_.phys << " np=0x" << next_pts_.phys
                      << " stt=0x" << status_.phys << " er=0x" << err_.phys
                      << std::dec << std::endl;
            for (int l = 0; l < kPyrLevels; ++l) std::swap(prev_pyr_[l], cur_pyr_[l]);
            return false;
        }
    }

    // 5) 回读匹配点（status!=0），输出工作坐标点对
    ot_svp_point_s25q7* np = (ot_svp_point_s25q7*)(uintptr_t)next_pts_.virt;
    uint8_t* st = (uint8_t*)(uintptr_t)status_.virt;
    for (int i = 0; i < pts_num; ++i) {
        if (st[i] == 0) continue;
        prev_work_pts.push_back(kept[i]);
        cur_work_pts.push_back(cv::Point2f(np[i].x / 128.0f, np[i].y / 128.0f));
    }

    // 6) cur → 下一帧 prev
    for (int l = 0; l < kPyrLevels; ++l) std::swap(prev_pyr_[l], cur_pyr_[l]);
    has_prev_ = true;

    return (int)prev_work_pts.size() >= IVE_MIN_TRACK_PTS;
}

bool IveGmc::match_bgr(const cv::Mat& bgr,
                       const std::vector<cv::Rect>& boxes,
                       std::vector<cv::Point2f>& prev_work_pts,
                       std::vector<cv::Point2f>& cur_work_pts,
                       float& work_scale) {
    if (!inited_ || bgr.empty()) return false;
    work_scale_ = (float)work_w_ / (float)bgr.cols;
    work_scale  = work_scale_;

    // CPU 转灰度 + 缩放到工作分辨率（只此一步在 CPU；角点/光流全在 IVE）
    {
        fxprof::ScopedGmc _g(fxprof::GmcStage::Front);   // CPU 前段：cvtColor+resize+上传
        cv::Mat gray, work;
        if (bgr.channels() == 3)       cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        else if (bgr.channels() == 4)  cv::cvtColor(bgr, gray, cv::COLOR_BGRA2GRAY);
        else                           gray = bgr;
        cv::resize(gray, work, cv::Size(work_w_, work_h_), 0, 0, cv::INTER_AREA);

        // 上传到 cur_pyr_[0]（按 work_stride_ 逐行拷贝）
        uint8_t* dst = (uint8_t*)(uintptr_t)cur_pyr_[0].virt;
        for (int r = 0; r < work_h_; ++r)
            std::memcpy(dst + (size_t)r * work_stride_, work.ptr(r), work_w_);
    }

    return process_after_level0(boxes, prev_work_pts, cur_work_pts);
}

bool IveGmc::match_yuv420sp(uint64_t y_phys, uint64_t y_virt, int w, int h, int y_stride,
                            const std::vector<cv::Rect>& boxes,
                            std::vector<cv::Point2f>& prev_work_pts,
                            std::vector<cv::Point2f>& cur_work_pts,
                            float& work_scale) {
    if (!inited_ || y_virt == 0 || w <= 0 || h <= 0) return false;
    work_scale_ = (float)work_w_ / (float)w;
    work_scale  = work_scale_;

    // Y 平面直接当 U8C1 全分辨率灰度 → IVE 缩放到 cur_pyr_[0]（零 CPU 拷贝）
    Mmz ymm; ymm.phys = y_phys; ymm.virt = y_virt; ymm.size = (uint32_t)y_stride * h;
    if (!ive_resize_u8c1(ymm, w, h, y_stride,
                         cur_pyr_[0], lvl_w_[0], lvl_h_[0], lvl_stride_[0]))
        return false;

    return process_after_level0(boxes, prev_work_pts, cur_work_pts);
}

#endif /* USE_HISI_IVE */

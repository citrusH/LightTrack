#ifndef MODULE_PERSON_DETECT_H
#define MODULE_PERSON_DETECT_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "types.h"
#include "svp_acl_rt.h"
#include "svp_acl.h"
#include "svp_acl_ext.h"
#include "extern_c_wrapper.h"


namespace VISION_ENGINE
{
	/* ---------------------------------------------------------------
	* Detector 配置（对应文件2 Config）
	* --------------------------------------------------------------- */
	struct DetectorConfig {
		std::string model_path;   /* .om 文件路径 */
		bool        bgr2rgb      = false;
		float       threshold_iou = 0.5f;
		int         device_id     = 0;
		int         num_anchor    = 3549;
		int         dim_anchor    = 8;     /* x/y/w/h + obj + 3 classes */
	};

	/* ---------------------------------------------------------------
	* DV500 YOLOX Detector
	* 使用原生 sample_common_svp_npu task，向跟踪层输出 xyxy。
	* --------------------------------------------------------------- */
	class Detector {
	public:
		Detector()  = default;
		~Detector() { destroy(); }
		Detector(const Detector&)            = delete;
		Detector& operator=(const Detector&) = delete;

		int  init(const DetectorConfig& cfg);
		int  run(const cv::Mat& bgr_image, float thresh_score,
				std::vector<ObjDetInfo>& result);
		void destroy();

	private:
		/* ---- 推理三阶段（对应文件2） ---- */
		int preProcess (const cv::Mat& image, float& scale_w, float& scale_h);
		int inference  ();
		int postProcess(cv::Size src_img_size, float scale_w, float scale_h,
						float thresh_conf, std::vector<ObjDetInfo>& result);

		/* ---- DV500 模型与 task 资源；SVP ACL runtime 由 LightTracker 持有 ---- */
		int  load_model_  ();
		void unload_model_();
		int  init_task_   ();
		void deinit_task_ ();
		static inline float fp16_to_f32(uint16_t h);

		/* ---- 状态 ---- */
		static const unsigned MODEL_IDX = 0;

		DetectorConfig              cfg_;
		bool                        inited_   = false;
		bool                        model_loaded_ = false;
		bool                        task_initialized_ = false;
		int                         net_w_    = 416;
		int                         net_h_    = 416;
		float                       thresh_conf_ = 0.5f;
		float                       thresh_iou_  = 0.5f;
		sample_svp_npu_task_info task_info_;
	};
}

#endif


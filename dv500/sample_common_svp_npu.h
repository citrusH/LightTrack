/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */

#ifndef SAMPLE_COMMON_SVP_NPU_H
#define SAMPLE_COMMON_SVP_NPU_H
#include "svp_acl.h"
#include "svp_acl_mdl.h"
#include "ot_type.h"

#define SAMPLE_SVP_NPU_MAX_THREAD_NUM    16
#define SAMPLE_SVP_NPU_MAX_TASK_NUM      16
#define SAMPLE_SVP_NPU_MAX_STREAM_NUM    16
#define SAMPLE_SVP_NPU_MAX_MODEL_NUM     6
#define SAMPLE_SVP_NPU_EXTRA_INPUT_NUM   2
#define SAMPLE_SVP_NPU_BYTE_BIT_NUM      8
#define SAMPLE_SVP_NPU_SHOW_TOP_NUM      5
#define SAMPLE_SVP_NPU_MAX_NAME_LEN      32
#define SAMPLE_SVP_NPU_MAX_MEM_SIZE      0xFFFFFFFF
#define SAMPLE_SVP_NPU_RECT_LEFT_TOP     0
#define SAMPLE_SVP_NPU_RECT_RIGHT_TOP    1
#define SAMPLE_SVP_NPU_RECT_RIGHT_BOTTOM 2
#define SAMPLE_SVP_NPU_RECT_LEFT_BOTTOM  3
#define SAMPLE_SVP_NPU_THRESHOLD_NUM     4

#define SAMPLE_SVP_NPU_RFCN_THRESHOLD_NUM      2
#define SAMPLE_SVP_NPU_AICPU_WAIT_TIME         1000
#define SAMPLE_SVP_NPU_RECT_COLOR              0x0000FF00
#define SAMPLE_SVP_NPU_MILLIC_SEC              20000
#define SAMPLE_SVP_NPU_IMG_THREE_CHN           3
#define SAMPLE_SVP_NPU_DOUBLE                  2

#define SAMPLE_SVP_NPU_PRIOR_TIMEOUT    1200
#define SAMPLE_SVP_NPU_TIMESTAMP_PER_SECOND 1000

typedef enum {
    SAMPLE_SVP_NPU_PRIORITY_0 = 0,
    SAMPLE_SVP_NPU_PRIORITY_1 = 1,
    SAMPLE_SVP_NPU_PRIORITY_2 = 2,
    SAMPLE_SVP_NPU_PRIORITY_3 = 3,
    SAMPLE_SVP_NPU_PRIORITY_4 = 4,
    SAMPLE_SVP_NPU_PRIORITY_5 = 5,
    SAMPLE_SVP_NPU_PRIORITY_6 = 6,
    SAMPLE_SVP_NPU_PRIORITY_7 = 7,
    SAMPLE_SVP_NPU_PRIORITY_BUTT,
} sample_svp_npu_model_priority;

typedef struct {
    td_u32 priority;
    td_u32 prior_preemp_en;
    td_u32 prior_up_step_timeout;
    td_u32 prior_up_top_timeout;
} sample_svp_npu_model_prior_cfg;

typedef struct {
    td_u32 model_id;
    td_bool is_load_flag;
    td_ulong model_mem_size;
    td_void *model_mem_ptr;
    svp_acl_mdl_desc *model_desc;
    svp_acl_mdl_config_handle *handle;
    sample_svp_npu_model_prior_cfg prior_cfg;
    size_t input_num;
    size_t output_num;
    size_t dynamic_batch_idx;
} sample_svp_npu_model_info;

typedef struct {
    td_u32 max_batch_num;
    td_u32 dynamic_batch_num;
    td_u32 total_t;
    td_bool is_cached;
    td_u32 model_idx;
} sample_svp_npu_task_cfg;

typedef struct {
    sample_svp_npu_task_cfg cfg;
    svp_acl_mdl_dataset *input_dataset;
    svp_acl_mdl_dataset *output_dataset;
    td_void *task_buf_ptr;
    size_t task_buf_size;
    size_t task_buf_stride;
    td_void *work_buf_ptr;
    size_t work_buf_size;
    size_t work_buf_stride;
} sample_svp_npu_task_info;

typedef struct {
    td_void *work_buf_ptr;
    size_t work_buf_size;
    size_t work_buf_stride;
} sample_svp_npu_shared_work_buf;

typedef struct {
    td_float score;
    td_u32 class_id;
} sample_svp_npu_top_n_result;

typedef struct {
    td_char *num_name;
    td_char *roi_name;
    td_bool has_background;
    td_u32 roi_offset;
} sample_svp_npu_detection_info;

typedef enum {
    SAMPLE_SVP_NPU_RFCN = 0,
    SAMPLE_SVP_NPU_YOLO = 1,
    SAMPLE_SVP_NPU_HRNET = 2,
    SAMPLE_SVP_NPU_MODEL_NAME_BUTT,
} svp_npu_model_name;

typedef struct {
    svp_npu_model_name model_name;
    sample_svp_npu_detection_info *detection_info;
} sample_svp_npu_thread_args;

typedef struct {
    td_char roi_num_name[SAMPLE_SVP_NPU_MAX_NAME_LEN];
    td_char roi_class_name[SAMPLE_SVP_NPU_MAX_NAME_LEN];
} sample_svp_npu_roi_info;

typedef struct {
    td_float nms_threshold;
    td_float score_threshold;
    td_float min_height;
    td_float min_width;
    td_char *name;
} sample_svp_npu_threshold;

typedef struct {
    td_u8 crop_switch;
    td_u8 padding_switch;
    td_u8 scf_switch;
    td_u8 preproc_switch;

    td_s32 crop_pos_x;
    td_s32 crop_pos_y;
    td_s32 crop_width;
    td_s32 crop_height;

    td_s32 padding_top;
    td_s32 padding_bottom;
    td_s32 padding_left;
    td_s32 padding_right;

    td_s32 scf_input_width;
    td_s32 scf_input_height;
    td_s32 scf_output_width;
    td_s32 scf_output_height;

    td_float reci_chn0;
    td_float reci_chn1;
    td_float reci_chn2;
    td_float reci_chn3;

    td_float mean_chn0;
    td_float mean_chn1;
    td_float mean_chn2;
    td_float mean_chn3;

    td_float min_chn0;
    td_float min_chn1;
    td_float min_chn2;
    td_float min_chn3;
} sample_aipp_dynamic_batch_param;

typedef struct {
    svp_acl_aipp_input_format input_format;
    td_s8 csc_switch;
    td_s8 rbuv_swap_switch;
    td_s8 ax_swap_switch;
    td_s32 image_width;
    td_s32 image_height;
    td_s32 csc_matrix_r0c0;
    td_s32 csc_matrix_r0c1;
    td_s32 csc_matrix_r0c2;
    td_s32 csc_matrix_r1c0;
    td_s32 csc_matrix_r1c1;
    td_s32 csc_matrix_r1c2;
    td_s32 csc_matrix_r2c0;
    td_s32 csc_matrix_r2c1;
    td_s32 csc_matrix_r2c2;
    td_u8 csc_output_bias_r0;
    td_u8 csc_output_bias_r1;
    td_u8 csc_output_bias_r2;
    td_u8 csc_input_bias_r0;
    td_u8 csc_input_bias_r1;
    td_u8 csc_input_bias_r2;
} sample_aipp_dynamic_param;

typedef struct {
    sample_aipp_dynamic_param dynamic_param;
    sample_aipp_dynamic_batch_param dynamic_batch_param;
} sample_svp_npu_model_aipp;

#endif

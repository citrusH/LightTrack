#pragma once

#include "svp_acl.h"
#include "svp_acl_rt.h"

// LightTracker 内 SVP ACL global/device 生命周期的唯一所有者。
// 第一阶段只支持库内独占、单线程、单个已初始化 Tracker。
class SvpAclRuntime {
public:
    SvpAclRuntime() = default;
    ~SvpAclRuntime();

    SvpAclRuntime(const SvpAclRuntime&) = delete;
    SvpAclRuntime& operator=(const SvpAclRuntime&) = delete;

    int init(int device_id = 0);
    void release();

    bool is_initialized() const { return initialized_; }
    int device_id() const { return device_id_; }
    svp_acl_rt_run_mode run_mode() const { return run_mode_; }

private:
    static SvpAclRuntime* active_owner_;

    int device_id_ = 0;
    svp_acl_rt_run_mode run_mode_{};
    bool acl_initialized_ = false;
    bool device_set_ = false;
    bool initialized_ = false;
};

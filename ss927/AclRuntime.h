#pragma once

#include "acl/acl.h"
#include <cstdint>

// ACL global runtime 在进程内只初始化一次；每个 LightTracker 实例独占其
// device/context 生命周期。第一阶段只允许一个已初始化实例，且所有调用必须位于同一线程。
class AclRuntime {
public:
    AclRuntime() = default;
    ~AclRuntime();

    AclRuntime(const AclRuntime&) = delete;
    AclRuntime& operator=(const AclRuntime&) = delete;

    int init(int32_t device_id = 0);
    void release();

    bool is_initialized() const { return initialized_; }
    bool is_device() const { return run_mode_ == ACL_DEVICE; }
    aclrtRunMode run_mode() const { return run_mode_; }
    aclrtContext context() const { return context_; }

private:
    static AclRuntime* active_owner_;
    static bool acl_process_initialized_;

    int32_t device_id_ = 0;
    aclrtRunMode run_mode_{};
    aclrtContext context_ = nullptr;
    bool device_set_ = false;
    bool context_created_ = false;
    bool initialized_ = false;
};

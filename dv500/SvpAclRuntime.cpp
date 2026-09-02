#include "SvpAclRuntime.h"

#include <cstdio>

SvpAclRuntime* SvpAclRuntime::active_owner_ = nullptr;

SvpAclRuntime::~SvpAclRuntime()
{
    release();
}

int SvpAclRuntime::init(int device_id)
{
    if (initialized_) return 0;
    if (active_owner_ != nullptr && active_owner_ != this) {
        std::fprintf(stderr,
            "[SVP_ACL][E] init rejected: only one active tracker is supported\n");
        return -1;
    }

    active_owner_ = this;
    device_id_ = device_id;

    svp_acl_error ret = svp_acl_init(nullptr);
    if (ret != SVP_ACL_SUCCESS) {
        std::fprintf(stderr, "[SVP_ACL][E] init failed, error=0x%x\n", ret);
        active_owner_ = nullptr;
        return -1;
    }
    acl_initialized_ = true;
    std::fprintf(stdout, "[SVP_ACL][I] init\n");

    ret = svp_acl_rt_set_device(device_id_);
    if (ret != SVP_ACL_SUCCESS) {
        std::fprintf(stderr, "[SVP_ACL][E] set device %d failed, error=0x%x\n",
                     device_id_, ret);
        release();
        return -1;
    }
    device_set_ = true;
    std::fprintf(stdout, "[SVP_ACL][I] set device %d\n", device_id_);

    ret = svp_acl_rt_get_run_mode(&run_mode_);
    if (ret != SVP_ACL_SUCCESS || run_mode_ != SVP_ACL_DEVICE) {
        std::fprintf(stderr,
            "[SVP_ACL][E] invalid run mode, error=0x%x mode=%d\n",
            ret, static_cast<int>(run_mode_));
        release();
        return -1;
    }

    initialized_ = true;
    std::fprintf(stdout, "[SVP_ACL][I] ready\n");
    return 0;
}

void SvpAclRuntime::release()
{
    initialized_ = false;

    if (device_set_) {
        svp_acl_error ret = svp_acl_rt_reset_device(device_id_);
        if (ret != SVP_ACL_SUCCESS) {
            std::fprintf(stderr,
                "[SVP_ACL][E] reset device %d failed, error=0x%x\n",
                device_id_, ret);
        } else {
            std::fprintf(stdout, "[SVP_ACL][I] reset device %d\n", device_id_);
        }
    }
    device_set_ = false;

    if (acl_initialized_) {
        svp_acl_error ret = svp_acl_finalize();
        if (ret != SVP_ACL_SUCCESS) {
            std::fprintf(stderr, "[SVP_ACL][E] finalize failed, error=0x%x\n", ret);
        } else {
            std::fprintf(stdout, "[SVP_ACL][I] finalize\n");
        }
    }
    acl_initialized_ = false;
    run_mode_ = svp_acl_rt_run_mode{};
    if (active_owner_ == this) active_owner_ = nullptr;
}

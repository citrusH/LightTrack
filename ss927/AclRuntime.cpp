#include "AclRuntime.h"

#include <cstdio>

#define ACL_INFO_LOG(fmt, ...) std::fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define ACL_ERROR_LOG(fmt, ...) std::fprintf(stderr, "[ERROR]  " fmt "\n", ##__VA_ARGS__)

AclRuntime* AclRuntime::active_owner_ = nullptr;
bool AclRuntime::acl_process_initialized_ = false;

AclRuntime::~AclRuntime()
{
    release();
}

int AclRuntime::init(int32_t device_id)
{
    if (initialized_) return 0;
    if (active_owner_ != nullptr && active_owner_ != this) {
        ACL_ERROR_LOG("[ACL] init rejected: only one active tracker runtime is supported");
        return -1;
    }

    active_owner_ = this;
    device_id_ = device_id;

    aclError ret = ACL_SUCCESS;
    if (!acl_process_initialized_) {
        ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            ACL_ERROR_LOG("[ACL] aclInit failed, errorCode=%d", static_cast<int32_t>(ret));
            active_owner_ = nullptr;
            return -1;
        }
        acl_process_initialized_ = true;
        ACL_INFO_LOG("[ACL] process init");
    } else {
        ACL_INFO_LOG("[ACL] reuse process runtime");
    }

    ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        ACL_ERROR_LOG("[ACL] set device %d failed, errorCode=%d",
                  device_id_, static_cast<int32_t>(ret));
        release();
        return -1;
    }
    device_set_ = true;
    ACL_INFO_LOG("[ACL] set device %d", device_id_);

    ret = aclrtGetRunMode(&run_mode_);
    if (ret != ACL_SUCCESS) {
        ACL_ERROR_LOG("[ACL] get run mode failed, errorCode=%d",
                  static_cast<int32_t>(ret));
        release();
        return -1;
    }

    ret = aclrtCreateContext(&context_, device_id_);
    if (ret != ACL_SUCCESS) {
        ACL_ERROR_LOG("[ACL] create context failed, device=%d, errorCode=%d",
                  device_id_, static_cast<int32_t>(ret));
        context_ = nullptr;
        release();
        return -1;
    }
    context_created_ = true;

    ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_SUCCESS) {
        ACL_ERROR_LOG("[ACL] set current context failed, errorCode=%d",
                  static_cast<int32_t>(ret));
        release();
        return -1;
    }

    initialized_ = true;
    ACL_INFO_LOG("[ACL] create context (runMode=%s)",
             is_device() ? "device" : "host");
    return 0;
}

void AclRuntime::release()
{
    initialized_ = false;

    if (context_created_ && context_ != nullptr) {
        aclError ret = aclrtSetCurrentContext(context_);
        if (ret != ACL_SUCCESS) {
            ACL_ERROR_LOG("[ACL] set current context before release failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        }
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_SUCCESS) {
            ACL_ERROR_LOG("[ACL] destroy context failed, errorCode=%d",
                      static_cast<int32_t>(ret));
        } else {
            ACL_INFO_LOG("[ACL] destroy context");
        }
    }
    context_ = nullptr;
    context_created_ = false;

    if (device_set_) {
        aclError ret = aclrtResetDevice(device_id_);
        if (ret != ACL_SUCCESS) {
            ACL_ERROR_LOG("[ACL] reset device %d failed, errorCode=%d",
                      device_id_, static_cast<int32_t>(ret));
        } else {
            ACL_INFO_LOG("[ACL] reset device %d", device_id_);
        }
    }
    device_set_ = false;

    run_mode_ = aclrtRunMode{};

    if (active_owner_ == this) active_owner_ = nullptr;
    if (acl_process_initialized_) {
        ACL_INFO_LOG("[ACL] instance release complete; process runtime retained");
    }
}

#undef ACL_INFO_LOG
#undef ACL_ERROR_LOG

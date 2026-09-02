#ifndef _GSF_RGBCAP_H_
#define _GSF_RGBCAP_H_

#ifdef __cplusplus
extern "C" {
#endif

int aiav_video_start(int s32Width, int s32Height,int src_rate, int dst_rate);
int aiav_video_stop();
unsigned char *aiav_get_bgr_frame_addr(int s32MilliSec);
int aiav_bgr_release_frame();

#ifdef __cplusplus
}
#endif

#endif 




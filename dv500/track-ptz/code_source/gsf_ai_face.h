#ifndef _GSF_AI_FACE_H_
#define _GSF_AI_FACE_H_


// #ifdef SUPPORT_AI_FACE

//#include "gsf_struct.h"
//#include "gsf_ive_interface.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct _GSF_AIFR_MAN_
{
    unsigned char       u8Init;
    unsigned char       u8Res[10];
	pthread_t  thIdReinit;
	pthread_mutex_t mutexReinit;
}GSF_AIFR_MAN, *LPGSF_AIFR_MAN;

int gsf_aifr_init();

int gsf_aifr_uninit();

int gsf_aifr_refresh(int nCh, void *pCfg);




#ifdef __cplusplus
}
#endif

// #endif // #ifdef SUPPORT_AI_FACE

#endif


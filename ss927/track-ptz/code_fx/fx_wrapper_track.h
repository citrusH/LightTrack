#ifndef TRACK_C_INTERFACE_H
#define TRACK_C_INTERFACE_H

#define MAX_PERSON_COUNT 5

unsigned short DEBUG_LOG = 0;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;
    float score;
    int x;
    int y;
    int w;
    int h;
} PersonInfo;
typedef struct {
    PersonInfo infos[MAX_PERSON_COUNT];
    int count;
} OutResult;

typedef struct {
    int x;
    int y;
    int x2;
    int y2;
} MainTarget;

typedef enum {
    TRACK_PTZ_BLIND_IDLE = 0,
    TRACK_PTZ_BLIND_SLIDING = 1,
    TRACK_PTZ_BLIND_STOPPED = 2
} TrackPtzBlindPhase;

// 不透明指针类型，隐藏C++实现细节
typedef struct TrackHandle TrackHandle;


// 创建Track对象
TrackHandle* track_create();


// 初始化跟踪器
void track_init(TrackHandle* handle);

// 销毁跟踪器实例（NULL 安全）。
void track_release(TrackHandle* handle);

typedef struct {
    unsigned char* pBGRData;   // 输入图像数据（BGR格式）
    int width;                 // 图像宽度
    int height;                // 图像高度
    MainTarget* mainTarget;    // 主目标（可选，可能用于引导跟踪）
    // char flipValue;  
    float zoomValue;            // 1 就开了翻转
    int ptz_blind_phase;        // 外部 PTZ 实际 blind 坐标系状态
} TrackInput;


// int track_run(TrackHandle* handle, unsigned char* pBGRData, OutResult* out_result);

// 形参：MainTarget& mainTarget  是c++ 的方式， 函数里可以直接使用 mainTarget.xx
        // c语言是： 形参：MainTarget* mainTarget ，函数里指针使用：mainTarget->xx
// int track_run(TrackHandle* handle, unsigned char* pBGRData, OutResult* out_result, MainTarget* mainTarget);


int track_run(TrackHandle* handle, TrackInput* input, OutResult* out_result);

#ifdef __cplusplus
}
#endif


#endif // TRACK_C_INTERFACE_H

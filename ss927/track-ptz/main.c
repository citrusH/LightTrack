#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <syslog.h>
#include <stdbool.h>

#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>
#include <math.h>
#include <time.h> 
#include "cJSON.h"

#include "gsf_ai_face.h"
#include "gsf_rgbcap.h"

#include "common.h"
#include "fx_wrapper_track.h"


#define FX_DEBUG 0
#define AIFR_VPSS_CH 2
#define AIFR_VIDEO_WIDTH 768//416
#define AIFR_VIDEO_HEIGHT 432//416
#define AIFR_DST_RATE 60//60//25 // -1 输出的帧率
#define pCenter (AIFR_VIDEO_WIDTH / 2)
int AIFR_SRC_RATE  = 30;//60; // -1 输入的帧率, 是多少就是多少 改不了
static const float wScale = 1.9f; // 768 w：416 ,  1.9

extern int get_humen_tracking_mode();
extern void gsf_uart_send_visca(unsigned char * pMsg);

static GSF_AIFR_MAN gAifrMan;
static pthread_t gAifrPid;
static int taskRunFlag = 0;

static TrackHandle* track = NULL;

typedef int (*readMode)(void);
typedef void (*sendViscaCmd)(unsigned char *pCmd);
// fx060627
#if FX_CHIP == 2 // dv500
typedef int32_t  GSF_S32;
typedef uint32_t GSF_U32;
typedef struct _GSF_RECT_S
{
	GSF_S32		s32X;							//起点x坐标
	GSF_S32		s32Y;							//起点y坐标
	GSF_U32		u32Width;						//区域宽度
	GSF_U32		u32Height;						//区域高度
}GSF_RECT_S, *LPGSF_RECT_S;
GSF_RECT_S gRect;
typedef int (*start_draw_rect)(void);
typedef int (*stop_draw_rect)(void);
typedef int (*set_rect)(int num, GSF_RECT_S *pRect);
float drawScale = 1920.0f / AIFR_VIDEO_WIDTH;
#endif

static short personCenter = 0;
static short speedRecord = 2;

#if FX_CHIP == 2 // dv500
typedef struct {
    readMode read_mode;
    sendViscaCmd send_visca_cmd;
    start_draw_rect pfn_start_rect; // fx060627
    stop_draw_rect pfn_stop_rect; // fx060627
    set_rect pfn_set_rect; // fx060627
} CbFuncs;
#else // ss927
typedef struct {
    readMode read_mode;
    sendViscaCmd send_visca_cmd;
} CbFuncs;
#endif

static CbFuncs gCb = {0};

void gsf_set_ai_function(CbFuncs *Cb)
{
	memcpy(&gCb, Cb, sizeof(gCb));
	printf("sucess set_ai_function\n");
}

unsigned char* get_safe_bgr_copy(int s32MilliSec) {
    unsigned char *src = NULL;
    unsigned char *dst = NULL;
    int size = 0;

    // 获取原始指针
    src = aiav_get_bgr_frame_addr(s32MilliSec);
    if (!src) return NULL;

    // 计算大小 (宽 * 高 * 3字节)
    size = AIFR_VIDEO_WIDTH * AIFR_VIDEO_HEIGHT * 3;

    // 分配新内存
    dst = (unsigned char*)malloc(size);
    if (!dst) {
        aiav_bgr_release_frame(); // 分配失败也要释放原帧
        return NULL;
    }

    memcpy(dst, src, size); // 拷贝数据
 
    aiav_bgr_release_frame(); // 释放原始硬件帧，归还资源

    return dst; //  返回安全的新指针 (调用者用完需 free)
}

#pragma region fxw

int stop_face();

unsigned long get_millis() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (unsigned long)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
    }
    return 0;
}

// static const uint64_t s_timeout_ms = 1500;  // Connect timeout in milliseconds


#pragma region 对接

// 速度
// unsigned char u8Speed = 2;
// unsigned char speed1 = 0x05;
// unsigned char speed2 = 0x08;
// unsigned char speed3 = 0x0b;
// unsigned char speed4 = 0x0d;
// unsigned char speed5 = 0x0f;
// 灵敏度
// unsigned short sensitivity1L = 0;
// unsigned short sensitivity1R = 0;

// 这里顺序请保持一样，以免强转出问题：GSF_AI_CFG_S *pAiCfg = (GSF_AI_CFG_S *)pCfg; 
typedef struct _GSF_AI_CFG_S
{
    unsigned char       u8Enable;
    unsigned char       u8Mode;
    unsigned char       u8Speed;
    unsigned char 		u8Zoom;
    unsigned int		u32Px;
    unsigned int		u32Py;
    unsigned int 		u32TopSpace;
    unsigned int 		u32Align;
    // unsigned int		u32Rev1;
    // unsigned int		u32Rev2;
}GSF_AI_CFG_S, *LPGSF_AI_CFG_S;

GSF_AI_CFG_S gCfg;

int gsf_aifr_set_frameRate(int frameRate){

    AIFR_SRC_RATE = frameRate;

    return 0;
}

static float gZoomValue = 1;
static float gFieldAngle = 60;
static char gFlipValue = 0;
static char gMirrorValue = 0; 

static uint16_t gLensZoomMultiple = 0;

int gsf_aifr_set_lens(uint16_t lensZoomMultiple){
    
    gLensZoomMultiple = lensZoomMultiple;
    
    return 0;
}

int gsf_aifr_set_ZFM(uint16_t zoomValue, char flipValue, char mirrorValue){

    // 20 倍镜头
    // float current_zoom = 1.0f + ((float)zoomValue / 0x4000) * (20.0f - 1.0f);

    // if (zoomValue > 0x4000) zoomValue = 0x4000;
    // gZoomValue = zoomValue;
    if(zoomValue < gLensZoomMultiple) gZoomValue = gLensZoomMultiple / zoomValue; 
    else gZoomValue = zoomValue / gLensZoomMultiple; //717.0f;

    gZoomValue = roundf(gZoomValue * 100.0f) / 100.0f; // 四舍五入保留两位小数

    gFlipValue = flipValue;

    gMirrorValue = mirrorValue;


    return 0;
}

int gsf_aifr_get_param(int nCh, void *pCfg){

    GSF_AI_CFG_S *pAiCfg = (GSF_AI_CFG_S *)pCfg;
    if (pAiCfg == NULL){printf("[gsf_aifr_get_param] error 1\n");return 1;}

    FILE *file = fopen("/oem/aitrack_cfg.json", "r");  
    if(!file){
        printf("[gsf_aifr_get_param] error 2\n");

        // 没有就创建一个
        file = fopen("/oem/aitrack_cfg.json", "w");
        if(file){
            fprintf(file, "{\n"
                     "    \"u8Enable\": 0,\n"
                     "    \"u8Zoom\": 0,\n"
                     "    \"u32Px\": 0,\n"
                     "    \"u32Py\": 0,\n"
                     "    \"u8Speed\": 1,\n"
                     "    \"u8Mode\": 1,\n"
                     "    \"u32TopSpace\": 2,\n"
                     "    \"u32Align\": 2\n"
                     "}");

            fflush(file); int fd = fileno(file); fsync(fd);
            fclose(file); 
        }

        // 初始化参数，其它参数都是 0
        gCfg.u8Speed = 2;
        gCfg.u32TopSpace = 2;
        gCfg.u32Align = 2;

        return 2;
    }

    fseek(file, 0, SEEK_END); 
    int length = ftell(file); 
    fseek(file, 0, SEEK_SET);
    char *buffer = (char *)malloc(length + 1); // 突然想使用动态分配的方式
    if(!buffer){printf("[gsf_aifr_get_param] error 3\n");fclose(file); return 3;}
    fread(buffer, 1, length, file);
    buffer[length] = '\0';
    fclose(file);

    cJSON *root = cJSON_Parse(buffer);
    if (!root) {printf("[gsf_aifr_get_param] error 4\n"); free(buffer); return 4;}

    cJSON* cjsonN;

    unsigned int error = 0;
    cjsonN = cJSON_GetObjectItem(root, "u8Enable");
    if(cjsonN != NULL) pAiCfg->u8Enable = cjsonN->valueint;
    else error++;

    cjsonN = cJSON_GetObjectItem(root, "u8Zoom");
    if(cjsonN != NULL) pAiCfg->u8Zoom = cjsonN->valueint;
    else error++;

    pAiCfg->u32Px = 0;

    pAiCfg->u32Py = 0;

    cjsonN = cJSON_GetObjectItem(root, "u8Speed");
    if(cjsonN != NULL) pAiCfg->u8Speed = cjsonN->valueint;
    else error++;
    
    cjsonN = cJSON_GetObjectItem(root, "u8Mode");
    if(cjsonN != NULL) pAiCfg->u8Mode = cjsonN->valueint;
    else error++;

    cjsonN = cJSON_GetObjectItem(root, "u32TopSpace");
    if(cjsonN != NULL) pAiCfg->u32TopSpace = cjsonN->valueint;
    else error++;

    cjsonN = cJSON_GetObjectItem(root, "u32Align");
    if(cjsonN != NULL) pAiCfg->u32Align = cjsonN->valueint;
    else error++;

    if(error) printf("read json %d error \n", error);

    free(buffer);
    cJSON_Delete(root);
    
    return 0;
}

int gsf_aifr_set_param_enable(int value){
    FILE *file = fopen("/oem/aitrack_cfg.json", "r");
    if(!file){ printf("[gsf_aifr_set_param] error 2\n");return 2;}
    fseek(file, 0, SEEK_END);
    int length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buffer = (char *)malloc(length + 1);
    if(!buffer){printf("[gsf_aifr_set_param] error 3\n");fclose(file);return 3;}
    fread(buffer, 1, length, file);
    buffer[length] = '\0';
    fclose(file);

    cJSON *root = cJSON_Parse(buffer);
    if(!root) {printf("[gsf_aifr_set_param] error 4\n");free(buffer);return 4;}

    // 仅修改u8Enable为0，其余字段完全保留原有数据
    cJSON_ReplaceItemInObject(root, "u8Enable", cJSON_CreateNumber(value));

    char *new_json_string = cJSON_Print(root);
    if (!new_json_string || strlen(new_json_string) < 5) {
        printf("[gsf_aifr_set_param] error 10\n");
        cJSON_Delete(root);
        free(buffer);
        return 10;
    }
    free(buffer);

    file = fopen("/oem/aitrack_cfg.json", "w");
    if (!file) {
        printf("[gsf_aifr_set_param] error 5\n");
        cJSON_Delete(root);
        free(new_json_string);
        return 5;
    }
    fputs(new_json_string, file);

    fflush(file); int fd = fileno(file); fsync(fd);

    fclose(file);
    cJSON_Delete(root);
    free(new_json_string);

    return 0;
}

int gsf_aifr_set_param(int nCh, void *pCfg){

    GSF_AI_CFG_S *pAiCfg = (GSF_AI_CFG_S *)pCfg;
    if (pAiCfg == NULL){printf("[gsf_aifr_set_param] error 1\n");return 1;}

    // printf("[gsf_aifr_set_param]:  u8Enable=%hhu u8Zoom=%hhu,u32Px=%d,u32Py=%d,u8Speed=%hhu,u8Mode=%hhu, u32TopSpace=%d u32Align=%d\n", pAiCfg->u8Enable, pAiCfg->u8Zoom,  pAiCfg->u32Px, pAiCfg->u32Py, pAiCfg->u8Speed, pAiCfg->u8Mode, pAiCfg->u32TopSpace, pAiCfg->u32Align);

    // 1. 这里朴工会调用，在这里就可以判断数据的改变
    gCfg.u8Enable = pAiCfg->u8Enable; // 🚩
    gCfg.u8Zoom = pAiCfg->u8Zoom; // 🚩
    gCfg.u32Px = pAiCfg->u32Px; // 🚩
    gCfg.u32Py = pAiCfg->u32Py; // 🚩
    gCfg.u8Speed = pAiCfg->u8Speed;  // ⭕ 
    gCfg.u8Mode = pAiCfg->u8Mode; // 🚩
    gCfg.u32TopSpace = pAiCfg->u32TopSpace;
    gCfg.u32Align = pAiCfg->u32Align;
    


    // 2. 把参数保存到 json 文件里面去
    FILE *file = fopen("/oem/aitrack_cfg.json", "r");  
    if(!file){printf("[gsf_aifr_set_param] error 2\n");fclose(file); return 2;}
    fseek(file, 0, SEEK_END); 
    int length = ftell(file); 
    fseek(file, 0, SEEK_SET);
    char *buffer = (char *)malloc(length + 1); // 突然想使用动态分配的方式
    if(!buffer){printf("[gsf_aifr_set_param] error 3\n");fclose(file); return 3;}
    fread(buffer, 1, length, file);
    buffer[length] = '\0';
    fclose(file);

    cJSON *root = cJSON_Parse(buffer);
    if (!root) {printf("[gsf_aifr_set_param] error 4\n"); free(buffer); return 4;}

    cJSON_ReplaceItemInObject(root, "u8Enable", cJSON_CreateNumber(pAiCfg->u8Enable));
    cJSON_ReplaceItemInObject(root, "u8Zoom", cJSON_CreateNumber(pAiCfg->u8Zoom));
    // cJSON_ReplaceItemInObject(root, "u32Px", cJSON_CreateNumber(pAiCfg->u32Px));
    // cJSON_ReplaceItemInObject(root, "u32Py", cJSON_CreateNumber(pAiCfg->u32Py));
    cJSON_ReplaceItemInObject(root, "u8Speed", cJSON_CreateNumber(pAiCfg->u8Speed));
    cJSON_ReplaceItemInObject(root, "u8Mode", cJSON_CreateNumber(pAiCfg->u8Mode));
    cJSON_ReplaceItemInObject(root, "u32TopSpace", cJSON_CreateNumber(pAiCfg->u32TopSpace));
    cJSON_ReplaceItemInObject(root, "u32Align", cJSON_CreateNumber(pAiCfg->u32Align));

    char *new_json_string = cJSON_Print(root);
    if (!new_json_string || strlen(new_json_string) < 5) {
        printf("[gsf_aifr_set_param] error 10\n");
        cJSON_Delete(root);
        free(buffer);
        return 10;
    }
    free(buffer); 
    file = fopen("/oem/aitrack_cfg.json", "w");
    if (!file) {  
        printf("[gsf_aifr_set_param] error 5\n");  
        cJSON_Delete(root);  
        free(new_json_string);  
        return 5;  
    }

    fputs(new_json_string, file); 

    fflush(file); int fd = fileno(file); fsync(fd);

    fclose(file);  
    cJSON_Delete(root);  
    free(new_json_string); 

    
    return 0;
}

int gsf_aifr_reset_param(){

    FILE *file = fopen("/oem/aitrack_cfg.json", "r");  
    if(!file){printf("[gsf_aifr_set_param] error 2\n");fclose(file); return 2;}
    fseek(file, 0, SEEK_END); 
    int length = ftell(file); 
    fseek(file, 0, SEEK_SET);
    char *buffer = (char *)malloc(length + 1); // 突然想使用动态分配的方式
    if(!buffer){printf("[gsf_aifr_set_param] error 3\n");fclose(file); return 3;}
    fread(buffer, 1, length, file);
    buffer[length] = '\0';
    fclose(file);

    cJSON *root = cJSON_Parse(buffer);
    if (!root) {printf("[gsf_aifr_set_param] error 4\n"); free(buffer); return 4;}

    cJSON_ReplaceItemInObject(root, "u8Enable", cJSON_CreateNumber(0));
    cJSON_ReplaceItemInObject(root, "u8Zoom", cJSON_CreateNumber(0));
    cJSON_ReplaceItemInObject(root, "u32Px", cJSON_CreateNumber(0));
    cJSON_ReplaceItemInObject(root, "u32Py", cJSON_CreateNumber(0));
    cJSON_ReplaceItemInObject(root, "u8Speed", cJSON_CreateNumber(2));
    cJSON_ReplaceItemInObject(root, "u8Mode", cJSON_CreateNumber(1));
    cJSON_ReplaceItemInObject(root, "u32TopSpace", cJSON_CreateNumber(2));
    cJSON_ReplaceItemInObject(root, "u32Align", cJSON_CreateNumber(2));

    char *new_json_string = cJSON_Print(root);
    if (!new_json_string || strlen(new_json_string) < 5) {
        printf("[gsf_aifr_set_param] error 10\n");
        cJSON_Delete(root);
        return 10;
    }
    free(buffer); 
    file = fopen("/oem/aitrack_cfg.json", "w");
    if (!file) {  
        printf("[gsf_aifr_set_param] error 5\n");  
        cJSON_Delete(root);  
        free(new_json_string);  
        return 5;  
    }

    fputs(new_json_string, file); 

    fflush(file); int fd = fileno(file); fsync(fd);

    fclose(file);  
    cJSON_Delete(root);  
    free(new_json_string); 

    
    return 0;
}


int hexchar_to_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1; // 非法字符
}
int hexstr_to_bytes(const char *hexstr, unsigned char *data, size_t len) {
    // if (strlen(hexstr) != 2 * len) return -1;
    for (size_t i = 0; i < len; i++) {

        // 直接用自己的hexchar_to_val 只检测16进制效果高一些,  strtol 会慢一些
        int high = hexchar_to_val(hexstr[2 * i]);
        int low = hexchar_to_val(hexstr[2 * i + 1]);
        if (high == -1 || low == -1) return -1;
        data[i] = (high << 4) | low;
    }
    return 0;
}
void bytes_to_hexstr(const unsigned char *data, size_t len, char *hexstr) { // 将16进制数据转为字符串
    for (size_t i = 0; i < len; i++) {
        sprintf(hexstr + 2 * i, "%02X", data[i]);
    }
    hexstr[2 * len] = '\0';
}
int gsf_aifr_get_uvcKey(unsigned char input_data[43]){ // 这个函数给朴工调用, 激活的时候朴工会调用一次,  然后我会把数据写到 json 文件里 做持久化存储

    if(!input_data) return 1;

    for (int i = 0; i < 43; i++) {
        printf("%02X ", input_data[i]);
    }
    printf("\n");

    // 转换为十六进制字符串
    char hexstr[43 * 2 + 1];
    bytes_to_hexstr(input_data, 43, hexstr);
    printf("Hex string: %s\n", hexstr);

    // 数据写到 json 文件 做持久化存储
    FILE *file = fopen("/dss/aitrack_uvcKey.json", "w");
    if (file) {
        fprintf(file, "{\n"
                      "    \"uvcKey\": \"%s\"\n"
                      "}", hexstr);
        
        fflush(file); int fd = fileno(file); fsync(fd);
        // sync(); // 全局
        
        fclose(file);
        printf("Successfully wrote hexstr to JSON file.\n");
    } else { perror("Failed to open file"); return 2; } 
        
    return 0;
}


#pragma endregion

#pragma region 追踪
// unsigned short readJson = 0; // json 转态
unsigned char debugJson = 0;
unsigned short lightStatus = 0; // 灯转态
unsigned short positionLog = 0; // 是否输出log, 看看是否放到 /oem/aitrack.json 比较合适

// visca cmd 
// 垂直方向的速度是第6个字节决定的
unsigned char vU1[9] =      {0x81,0x02,0x06,0x01,0x0c,0x0c,0x03,0x01,0xff};
unsigned char vD2[9] =      {0x81,0x02,0x06,0x01,0x0c,0x0c,0x03,0x02,0xff};
unsigned char vSt3[9] =     {0x81,0x02,0x06,0x01,0x64,0x64,0x10,0x03,0xff};
// 水平方向的速度是第5个字节决定的
unsigned char vL5[9] =      {0x81,0x02,0x06,0x01,0x0b,0x0b,0x01,0x03,0xff};
unsigned char vR6[9] =      {0x81,0x02,0x06,0x01,0x0b,0x0b,0x02,0x03,0xff};
unsigned char vSp7[9] =     {0x81,0x02,0x06,0x01,0x64,0x64,0x03,0x10,0xff};

unsigned char vS8[9] =      {0x81,0x02,0x06,0x01,0x64,0x64,0x03,0x03,0xff};

unsigned char vROn12[9] =   {0x81,0x01,0x7e,0x01,0x0a,0x00,0x02,0x00,0xff}; 
unsigned char vROff13[9] =  {0x81,0x01,0x7e,0x01,0x0a,0x00,0x03,0x00,0xff};
unsigned char vGOn14[9] =   {0x81,0x01,0x7e,0x01,0x0a,0x00,0x00,0x02,0xff};
unsigned char vGOff15[9] =  {0x81,0x01,0x7e,0x01,0x0a,0x00,0x00,0x03,0xff};

// 20 倍镜头
  // 1.1: 81 01 04 47 00 01 0c 04 ff
  // 1.2: 81 01 04 47 00 05 04 05 ff
  // 1.3: 81 01 04 47 00 07 03 05 ff
  // 1.4: 81 01 04 47 00 09 00 03 ff
unsigned char vZoom20[9] =  {0x81,0x01,0x04,0x47,0x00,0x00,0x00,0x00,0xff}; // 1 倍
unsigned char vZoom21[9] =  {0x81,0x01,0x04,0x47,0x00,0x00,0x00,0x00,0xff}; // 1 倍
// unsigned char vZoom22[9] =  {0x81,0x01,0x04,0x47,0x02,0x03,0x0a,0x0c,0xff}; // 3 倍
// unsigned char vZoom23[9] =  {0x81,0x01,0x04,0x47,0x02,0x03,0x0a,0x0c,0xff}; // 3 倍
// unsigned char vZoom24[9] =  {0x81,0x01,0x04,0x47,0x02,0x03,0x0a,0x0c,0xff}; // 3 倍
// unsigned char vZoom24[9] =  {0x81,0x01,0x04,0x47,0x02,0x03,0x0a,0x0c,0xff};


// 设置预置位和调用预置位, 都用 0x35 预置位  8x 01 04 3F 01 pq FF 
unsigned char vPreset0[9] =  {0x81,0x01,0x04,0x3f,0x00,0x35,0xff}; // 清除预置位
unsigned char vPreset1[9] =  {0x81,0x01,0x04,0x3f,0x01,0x35,0xff}; // 设置预置位
unsigned char vPreset2[9] =  {0x81,0x01,0x04,0x3f,0x02,0x35,0xff}; // 转到预置位



int scaleLevelRecord = 1;
unsigned short vPStatus = 100, vTStatus = 100, vZoomStatus = 100; // speedStatus = 0;  // pan 和 tilt 状态肯定是要分开的
unsigned short moveC = 0, sotpC = 0; // 多发几次移动，不然可能都不动

// 移动限制区域
// unsigned short p1L = pCenter-45, p1R = pCenter+45; //p2L = 240, p2R = 400;
// unsigned short t1U = 50, t1D = 220; // t2U = 15, t2D = 135;
//unsigned short oldpersonCenter = 0;
// unsigned short intervalFps = 0, oldY = 0, oldW = 0, oldX = 0, oldH = 0;


// fx260826
unsigned short faceMatchOK = 0; // 是否匹配成功
unsigned long lose_start_ms = 0;   // 人脸丢失起始毫秒时间
unsigned long g_lost_duration_ms = 0;  // 当前累计丢失时长(ms)
#define TOTAL_LOSE_TIMEOUT_MS  8000UL    // 总计超时， 8秒超时, 回主场
#define AUTO_PICK_TIMEOUT_MS   7000UL    // 丢失7秒自动拾取人脸,如果没有人脸继续往 LOSE_TIMEOUT_MS 走
unsigned char onefaceMatchOK = 0; // fx260328_1 主场
unsigned char pressRemoteControl = 0;  // fx260328_1 按下遥控器, 初始为1 ，也就是第一帧设置主目标后面由遥控器控制

// 水平方向系数
float KP_LINEAR = 0.04f; // 0.02f; // 主要影响中小误差时的速度,每偏离 1 个像素，基础速度增加多少
float KP_SQUARE = 0.0005f; // 0.0005f; // 主要影响大误差时的速度爆发力,速度随着距离的增加呈指数级上升
float SMOOTH_ALPHA = 0.40f; // 决定速度变化的惯性和平滑度,调大 接近 1.0, 极快，几乎立即达到目标速度
static float lastSmoothedSpeed = 0.0f;
int TARGET_X = AIFR_VIDEO_WIDTH / 2;
int H_OFFSET = 0; // 水平补偿
// int deadZoneX = 0; // 死区中心

// 垂直方向系数
// V_BASE_TOP_SPACE_RATIO   V_TOP_SPACE_REDUCTION_FACTOR
const float V_KP_LINEAR = 0.08f;
const float V_KP_SQUARE = 0.0012f;
const float V_STRONG_SMOOTH_ALPHA = 0.25f;
const int   V_BASE_ALIGN_HEIGHT = 35; //60;  
static float V_BASE_TOP_SPACE_RATIO = 0.15f;
const float V_TOP_SPACE_REDUCTION_FACTOR = 0.008f;
static float lastVSmoothedSpeed = 0.0f;
static bool isTargetLocked = false; // 锁定目标
static int lockedTargetY = 0; // 锁定目标Y
static int V_VERTICAL_OFFSET = 0;  // 上下偏移


// ============================================================================
// [MOD][PTZ_BLIND_DISTANCE_BY_SPEED]
// 主目标短时丢失后的 PTZ “固定最后真实框 + 速度积分限距”控制
//
// 最终策略：
// 1. 主目标真实存在：正常 PTZ 跟随，并保存最后一次真实 PersonInfo；
// 2. 主目标丢失：从头到尾都使用最后一次真实 PersonInfo 作为 blind 控制框；
//    blind box 的 x/y/w/h 均不改变，不再做 HOLD->DECAY，也不做 KF/速度位置外推；
// 3. 固定 box 继续给 PTZ 提供遮挡前的控制误差，因此 PTZ 延续原运动趋势；
// 4. PTZ 层记录当前真正下发的水平 speed code，并换算为“图像等效 px/s”；
// 5. blind 期间按 distance += speed_px_s * dt 累积估计滑动距离；
// 6. 累积距离达到最后真实 BODY box 宽度 × ratio 后，立即停止 blind 输出并 stop pan；
// 7. 时间仅作为 failsafe，绝不是正常 blind 终止条件；
// 8. blind 框只存在于 PTZ 层，不回写 tracker/KF/ReID；
// 9. 第一版 blind 只继续水平 pan：禁止自动变倍，并停止 tilt。
// ============================================================================

typedef enum {
    PTZ_BLIND_IDLE = 0,
    PTZ_BLIND_SLIDING,
    PTZ_BLIND_STOPPED
} PTZ_BLIND_PHASE_E;

typedef enum {
    PTZ_FOLLOW_NONE = 0,
    PTZ_FOLLOW_REAL,
    PTZ_FOLLOW_BLIND,
    PTZ_FOLLOW_STOPPED
} PTZ_FOLLOW_RESULT_E;

typedef struct {
    bool valid_real;                  // 是否已经保存过可靠真实主目标
    bool active;                      // 当前是否处于 blind slide episode
    PTZ_BLIND_PHASE_E phase;

    unsigned long start_ms;           // 本次真实目标丢失起始时间
    unsigned long last_integrate_ms;  // 上一次速度积分时间

    PersonInfo last_real;             // 最后一次真实主目标框
    PersonInfo blind_person;          // blind 输出；始终等于 last_real

    int last_real_center_x;           // 最后真实框中心 x
    int target_x;                     // 丢失瞬间冻结的水平构图目标位置
    unsigned int align_mode;          // 丢失瞬间冻结的构图模式
    int slide_direction;              // -1: error<0; +1: error>0；0 表示无需滑动

    float travelled_px;               // 本轮已累计的图像等效滑动距离(px)
    float max_travel_px;              // 本轮最大允许距离 = last_real.w * ratio
    bool actual_motion_started;       // 仅实际速度积分产生位移后才使 Tracker 失效旧坐标系
} PTZ_BLIND_CTRL_S;

static PTZ_BLIND_CTRL_S gBlindCtrl = {0};

// ============================================================================
// [NEW][PTZ_BLIND_DISTANCE_BY_SPEED] 距离阈值
//
// 1.0f：一次 blind episode 最多继续滑动 1 个最后真实 BODY box 宽度。
// 如果实机仍偏激进，可改成 0.75f / 0.5f。
// ============================================================================
#define PTZ_BLIND_MAX_TRAVEL_BOX_W_RATIO  1.0f

// 时间只做故障保护：例如速度读取/换算异常一直为 0 时，不能无限 blind。
// 正常情况下应由 travelled_px >= max_travel_px 提前结束。
#define PTZ_BLIND_FAILSAFE_MS             2000UL

// 第一版 blind 阶段禁止 tilt。垂直控制使用 person->y + 动态窗口，不能简单套水平策略。
#define PTZ_BLIND_ENABLE_TILT              0

// ============================================================================
// [NEW][PTZ_BLIND_DISTANCE_BY_SPEED] PTZ speed code -> 图像等效 px/s 标定
//
// 你的 vL5[4]/vR6[4] 是 VISCA 速度码，不是 px/s，不能直接和 BODY box width(px) 比。
// 这里统一做量纲转换：
//
//   speed_code -> angular_speed(deg/s) -> image_speed(px/s)
//
// 当前源码没有机芯“速度码 -> 实际角速度”的官方曲线，因此下面 1.0f 只是默认初值，
// 必须根据你的机芯文档或实机标定修改。所有误差集中在这一个参数/函数里，不污染
// blind 状态机。若后续可以直接取得真实 pan 角速度，只需要替换
// ptz_pan_speed_code_to_px_per_sec()，其它代码无需改动。
// ============================================================================
#ifndef PTZ_PAN_DEG_PER_SEC_PER_SPEED_CODE
#define PTZ_PAN_DEG_PER_SEC_PER_SPEED_CODE  1.0f
#endif

#define PTZ_MIN_HORIZONTAL_FOV_DEG          1.0f

// 当前水平 PTZ 实际下发状态；由 ptz_follow_horizontal() 每次发命令时更新。
static unsigned char gPanSpeedCode = 0;
static int gPanFollowDirection = 0;        // -1 / +1，与 error 符号一致；stop=0
static float gPanSpeedPxPerSec = 0.0f;     // 当前估计图像等效水平速度

// 自动变倍原先是 aiface_thread() 的局部状态；为了让所有 PTZ 行为收口到统一接口，移到 PTZ 层。
static unsigned long gLastZoomTime = 0;
static unsigned long gWidthStartTime = 0;
static int gIsWidthConditionMet = 0;

// [MOD] 统一 PTZ 跟随入口。核心调度层只需要传入真实主目标或 NULL。
int ptz_follow_update(const PersonInfo *real_person);

// 供下一帧 tracker_run 使用的只读状态。死区内没有任何实际 pan 位移时保持 IDLE，
// 保留 Tracker 原有的短时空间快速匹配；真正移动过后才进入坐标系失效恢复。
static int ptz_tracker_blind_phase(void)
{
    if (!gBlindCtrl.actual_motion_started)
        return TRACK_PTZ_BLIND_IDLE;
    if (gBlindCtrl.phase == PTZ_BLIND_SLIDING)
        return TRACK_PTZ_BLIND_SLIDING;
    if (gBlindCtrl.phase == PTZ_BLIND_STOPPED)
        return TRACK_PTZ_BLIND_STOPPED;
    return TRACK_PTZ_BLIND_IDLE;
}

// ============================================================================
// [MOD][PTZ_BLIND_DISTANCE_BY_SPEED] END
// ============================================================================

// ================
int open_light(int v1);

int read_debug_json(){
  #if !FX_DEBUG    
    return 0; // fxw 
  #endif

    if(debugJson) return 1;
    debugJson = 1;

    int fd = open("/oem/aitrack_debug.json", O_RDWR);
    if(fd < 0){ 
        printf("[%s %d] read json error \n", __func__, __LINE__);
        close(fd);

        // 新建一个
        FILE *file = fopen("/oem/aitrack_debug.json", "w");
        if(file){
            fprintf(file, "{\n"
                     "    \"positionLog\": 1\n"
                     "}");

            fflush(file); int fd = fileno(file); fsync(fd);
            fclose(file); 
        }

        return 1;
    }

    char buf[4096] = {0};
    int ret = read(fd, buf, sizeof(buf));
    close(fd);


    cJSON *root = cJSON_Parse(buf);
    positionLog = cJSON_GetObjectItem(root, "positionLog")->valueint;
    // KP_LINEAR = cJSON_GetObjectItem(root, "KP_LINEAR")->valuedouble;
    // KP_SQUARE = cJSON_GetObjectItem(root, "KP_SQUARE")->valuedouble;
    // SMOOTH_ALPHA = cJSON_GetObjectItem(root, "SMOOTH_ALPHA")->valuedouble;
    cJSON_Delete(root);


    printf("== [read_debug_json]: \
    positionLog=%d \
    ==\n", positionLog);

    return 0;
}

// fx260826

int prepare_stop_face(){

    // [MOD][PTZ_BLIND_FOLLOW]
    // 注意：正常“主目标匹配丢失”路径已经不再调用本函数，而改走 ptz_follow_update(NULL)。
    // 本函数仅保留给取帧失败等非正常视觉输入场景/兼容旧调用；其前30ms vS8 停止逻辑不参与 blind follow。

    unsigned long now_ms = get_millis();

    //第一次丢失，打上起始时间戳  
    if(lose_start_ms == 0) {
        lose_start_ms = now_ms;

        // open_light(1);
        // gCb.send_visca_cmd(vS8);
        // if(!vPStatus) vPStatus = 1;
    }
     
    g_lost_duration_ms = now_ms - lose_start_ms;
    // 前30ms内发送跟踪启动指令，为了防止漏发，多发几次
    if(g_lost_duration_ms < 30UL){
        open_light(1);
        gCb.send_visca_cmd(vS8);
        if(!vPStatus) vPStatus = 1;
    }
  
  
    // 判断是否超时 5秒
    if( g_lost_duration_ms >= TOTAL_LOSE_TIMEOUT_MS ) stop_face();


    return 0;
}

// fx260826
/*
    使用场景：
        1. 超时
        2. 休眠 (网页开关、遥控器控制)
*/
int stop_face(){

    // ========================================================================
    // [NEW][PTZ_BLIND_FOLLOW] 彻底停止/休眠时必须清空 blind 控制状态，
    // 防止下一次启动错误复用上一段真实框和衰减进度。
    // ========================================================================
    memset(&gBlindCtrl, 0, sizeof(gBlindCtrl));
    gIsWidthConditionMet = 0;
    gWidthStartTime = 0;

    // [NEW][PTZ_BLIND_DISTANCE_BY_SPEED] 彻底停止时清空 pan 速度积分状态。
    gPanSpeedCode = 0;
    gPanFollowDirection = 0;
    gPanSpeedPxPerSec = 0.0f;

    if(!vPStatus) {
        lose_start_ms = 0;
        g_lost_duration_ms = 0;
        return 1; // || !vTStatus 只要调用 ptz 那 vPStatus 和 vTStatus 就不会是 0
    }

    // printf("=============== [stop_face] =============== \n");

    if(gCfg.u8Mode == 2) gCb.send_visca_cmd(vPreset2); // fx260328_1 回到主场

    gCb.send_visca_cmd(vS8);
    

    lose_start_ms = 0;
    g_lost_duration_ms = 0;

    faceMatchOK = 0;
    vPStatus = 0;
    vTStatus = 0;

    open_light(1);

    debugJson = 0;read_debug_json();

    return 0;
}

int open_light(int v1){
  #if !FX_DEBUG    
    return 0; // fxw 
  #endif

    if(v1 == 1){ // 开红灯
        // printf("========== on  red ============== \n");
        if(lightStatus != 12){
            gCb.send_visca_cmd(vROff13);
            gCb.send_visca_cmd(vGOff15);
            gCb.send_visca_cmd(vROn12);
            lightStatus = 12;
        }
    }else if(v1 == 2){ // 开绿灯
        if(lightStatus != 14){ 
            gCb.send_visca_cmd(vROff13);
            gCb.send_visca_cmd(vGOff15);
            gCb.send_visca_cmd(vGOn14);
            lightStatus = 14;
        }
    }

    return 0;
}

int ptz(int v1){

    if(moveC > 2) moveC = 0; // 多发几次左右移动
    if(sotpC > 2) sotpC = 0;

    // 距离中心越远越快, 先不全部改， 只改 水平
    // if(personCenter) // 中心?
    // if(person->x) // 左上角 ?

    if(v1 == 7){
        if(vPStatus != v1 || sotpC < 2) {
            gCb.send_visca_cmd(vSp7);
            vPStatus = v1;
            sotpC++;
        }
    }else if(v1 == 5){
        if (vPStatus != v1 || moveC <= 2){
            gCb.send_visca_cmd(vL5);
            vPStatus = v1;
            moveC++;
        }
    }else if(v1 == 6){
         if (vPStatus != v1 || moveC <= 2){
            gCb.send_visca_cmd(vR6);
            vPStatus = v1;
            moveC++;
        }
    }else if(v1 == 3){
        if(vTStatus != v1) {
            gCb.send_visca_cmd(vSt3);
            vTStatus = v1;
        } 
    }else if(v1 == 1){
        if(vTStatus != v1 || moveC <= 2) {
            gCb.send_visca_cmd(vU1);
            vTStatus = v1;
            moveC++;
        } 
    }else if(v1 == 2){
        if(vTStatus != v1 || moveC <= 2) {
            gCb.send_visca_cmd(vD2);
            vTStatus = v1;
            moveC++;
        } 
    }else if(v1 == 20){
        if(vZoomStatus != v1) {
            gCb.send_visca_cmd(vZoom21);
            vZoomStatus = v1;
        } 
    }else if(v1 == 21){
        if(vZoomStatus != v1) {
            gCb.send_visca_cmd(vZoom21);
            vZoomStatus = v1;
        } 
    }else if(v1 == 22){
        if(vZoomStatus != v1) {
            // gCb.send_visca_cmd(vZoom22);
            vZoomStatus = v1;
        } 
    }else if(v1 == 23){
        if(vZoomStatus != v1) {
            // gCb.send_visca_cmd(vZoom23);
            vZoomStatus = v1;
        } 
    }else if(v1 == 24){
        if(vZoomStatus != v1) {
            // gCb.send_visca_cmd(vZoom24);
            vZoomStatus = v1;
        } 
    }
    

    return 0;
}

int set_zoom_level(int v1){  // v1 倍数

    if(vZoomStatus == v1) return 1;
    vZoomStatus = v1;

    switch (v1)
    {
        case 1:
            vZoom20[4] = 0x00;vZoom20[5] = 0x00;
            vZoom20[6] = 0x00;vZoom20[7] = 0x00;
        break;
        case 2:
            vZoom20[4] = 0x01;vZoom20[5] = 0x01;
            vZoom20[6] = 0x05;vZoom20[7] = 0x0a;
        break;
        case 3:
            vZoom20[4] = 0x01;vZoom20[5] = 0x0c;
            vZoom20[6] = 0x01;vZoom20[7] = 0x07;
        break;
        case 4:
            vZoom20[4] = 0x02;vZoom20[5] = 0x02;
            vZoom20[6] = 0x0c;vZoom20[7] = 0x0f;
        break;
        case 5:
            vZoom20[4] = 0x02;vZoom20[5] = 0x08;
            vZoom20[6] = 0x03;vZoom20[7] = 0x03;
        break;
        case 6:
            vZoom20[4] = 0x02;vZoom20[5] = 0x0c;
            vZoom20[6] = 0x03;vZoom20[7] = 0x09;
        break;
        case 7:
            vZoom20[4] = 0x02;vZoom20[5] = 0x0f;
            vZoom20[6] = 0x05;vZoom20[7] = 0x0e;
        break;
        case 8:
            vZoom20[4] = 0x03;vZoom20[5] = 0x02;
            vZoom20[6] = 0x01;vZoom20[7] = 0x0a;
        break;
        case 9:
            vZoom20[4] = 0x03;vZoom20[5] = 0x04;
            vZoom20[6] = 0x06;vZoom20[7] = 0x06;
        break;
        case 10:
            vZoom20[4] = 0x03;vZoom20[5] = 0x06;
            vZoom20[6] = 0x06;vZoom20[7] = 0x0f;
        break;
        case 11:
            vZoom20[4] = 0x03;vZoom20[5] = 0x08;
            vZoom20[6] = 0x01;vZoom20[7] = 0x0d;
        break;
        case 12:
            vZoom20[4] = 0x03;vZoom20[5] = 0x09;
            vZoom20[6] = 0x09;vZoom20[7] = 0x0d;
        break;
        case 13:
            vZoom20[4] = 0x03;vZoom20[5] = 0x0a;
            vZoom20[6] = 0x0e;vZoom20[7] = 0x06;
        break;
        case 14:
            vZoom20[4] = 0x03;vZoom20[5] = 0x0c;
            vZoom20[6] = 0x00;vZoom20[7] = 0x0c;
        break;
        case 15:
            vZoom20[4] = 0x03;vZoom20[5] = 0x0c;
            vZoom20[6] = 0x0f;vZoom20[7] = 0x09;
        break;
        case 16:
            vZoom20[4] = 0x03;vZoom20[5] = 0x0d;
            vZoom20[6] = 0x0c;vZoom20[7] = 0x05;
        break;
        case 17:
            vZoom20[4] = 0x03;vZoom20[5] = 0x0e;
            vZoom20[6] = 0x07;vZoom20[7] = 0x0a;
        break;
        case 18:
            vZoom20[4] = 0x03;vZoom20[5] = 0x0f;
            vZoom20[6] = 0x00;vZoom20[7] = 0x0d;
        break;
        case 19:
            vZoom20[4] = 0x03;vZoom20[5] = 0x0f;
            vZoom20[6] = 0x08;vZoom20[7] = 0x0a;
        break;
        case 20:
            vZoom20[4] = 0x04;vZoom20[5] = 0x00;
            vZoom20[6] = 0x00;vZoom20[7] = 0x00;
        break;
    
        default:
        break;
    }

    return 0;
}


// ============================================================================
// [NEW][PTZ_BLIND_FOLLOW] 统一 PTZ 跟随实现
// ============================================================================

static int ptz_horizontal_dead_zone(const PersonInfo *person)
{
    if (person == NULL) return 12;

    int deadZone = person->w / 4;
    if (deadZone < 12) deadZone = 12;
    if (deadZone > 42) deadZone = 42;
    return deadZone;
}


// ============================================================================
// [NEW][PTZ_BLIND_DISTANCE_BY_SPEED]
// 将当前真正下发的 VISCA 水平速度码换算成图像等效 px/s。
//
// 近似关系：
//   当前水平FOV ~= 1x水平FOV / zoom
//   px_per_deg  ~= image_width / 当前水平FOV
//   deg_per_sec ~= speed_code * 标定系数
//
// 注意：真正精度取决于 PTZ_PAN_DEG_PER_SEC_PER_SPEED_CODE 的实机标定。
// ============================================================================
static float ptz_pan_speed_code_to_px_per_sec(unsigned char speed_code)
{
    if (speed_code == 0) return 0.0f;

    float zoom = gZoomValue;
    if (zoom < 1.0f) zoom = 1.0f;

    float horizontal_fov_deg = gFieldAngle / zoom;
    if (horizontal_fov_deg < PTZ_MIN_HORIZONTAL_FOV_DEG)
        horizontal_fov_deg = PTZ_MIN_HORIZONTAL_FOV_DEG;

    float px_per_deg = (float)AIFR_VIDEO_WIDTH / horizontal_fov_deg;
    float angular_speed_deg_s =
        (float)speed_code * PTZ_PAN_DEG_PER_SEC_PER_SPEED_CODE;

    return angular_speed_deg_s * px_per_deg;
}

// [NEW] 记录“这一帧之后 PTZ 将以什么水平速度/方向运动”。
static void ptz_record_pan_motion(unsigned char speed_code, int follow_direction)
{
    gPanSpeedCode = speed_code;
    gPanFollowDirection = follow_direction;
    gPanSpeedPxPerSec = ptz_pan_speed_code_to_px_per_sec(speed_code);
}

// [NEW] 水平停止时同步清掉速度估计，避免下一帧 blind 继续积分旧速度。
static void ptz_record_pan_stop(void)
{
    gPanSpeedCode = 0;
    gPanFollowDirection = 0;
    gPanSpeedPxPerSec = 0.0f;
}

/*
 * [NEW] 只在真实主目标存在时刷新 TARGET_X。
 * blind 阶段使用“丢失瞬间冻结”的 gBlindCtrl.target_x，不能让虚拟框改变构图目标。
 *
 * 保留原始语义：
 *   u32Align == 1: 居左，TARGET_X 随真实 person->w 动态变化
 *   u32Align == 2: 居中，并按原代码将 u32Align 清 0
 *   u32Align == 3: 居右，TARGET_X 随真实 person->w 动态变化
 */
static void ptz_refresh_target_x_from_real(const PersonInfo *person)
{
    if (person == NULL || !gCfg.u32Align) return;

    H_OFFSET = (int)(person->w * 0.25f);

    if (gCfg.u32Align == 1) {
        TARGET_X = 30 + person->w - H_OFFSET;
    }
    else if (gCfg.u32Align == 2) {
        TARGET_X = AIFR_VIDEO_WIDTH / 2;
        gCfg.u32Align = 0;
    }
    else if (gCfg.u32Align == 3) {
        TARGET_X = AIFR_VIDEO_WIDTH - person->w - 30 + H_OFFSET;
    }
}

/*
 * [MOVED][PTZ_BLIND_FOLLOW]
 * 原 aiface_thread() 中的“自动变倍”逻辑收口到 PTZ 接口内部。
 * 注意：该函数只允许真实目标调用；blind 虚拟框绝不参与自动变倍。
 */
static void ptz_follow_auto_zoom_real(const PersonInfo *person)
{
    if (person == NULL) return;

    const short scaleLevel1 = (short)(52 * wScale);
    const short scaleLevel2 = (short)(140 * wScale);

    if (gCfg.u8Mode == 2) return;

    if (gCfg.u8Zoom) {
        unsigned long now = get_millis();
        if (now - gLastZoomTime >= 2200) {
            if (person->w <= scaleLevel1) {
                if (!gIsWidthConditionMet) {
                    gWidthStartTime = now;
                    gIsWidthConditionMet = 1;
                }
                else if (now - gWidthStartTime >= 2200) {
                    scaleLevelRecord += 2;
                    if (scaleLevelRecord >= 20) scaleLevelRecord = 20;
                    set_zoom_level(scaleLevelRecord);
                    gCb.send_visca_cmd(vZoom20);
                    gIsWidthConditionMet = 0;
                }
            }
            else if (person->w >= scaleLevel2) {
                if (!gIsWidthConditionMet) {
                    gWidthStartTime = now;
                    gIsWidthConditionMet = 1;
                }
                else if (now - gWidthStartTime >= 2200) {
                    scaleLevelRecord -= 2;
                    if (scaleLevelRecord <= 1) scaleLevelRecord = 1;
                    set_zoom_level(scaleLevelRecord);
                    gCb.send_visca_cmd(vZoom20);
                    gIsWidthConditionMet = 0;
                }
            }
            else {
                if (gIsWidthConditionMet) gIsWidthConditionMet = 0;
            }

            gLastZoomTime = now;
        }
    }
    else {
        ptz(21);
        if (scaleLevelRecord != 1) scaleLevelRecord = 1;
    }
}

/*
 * [MOVED + SMALL ADAPTATION][PTZ_BLIND_FOLLOW]
 * 原水平 PTZ 控制逻辑。
 *
 * target_x:
 *   REAL  -> 当前 TARGET_X
 *   BLIND -> 丢失瞬间冻结的 gBlindCtrl.target_x
 *
 * align_mode:
 *   用于保留原代码居左/居右时的额外速度补偿。
 *
 * 除 target_x 来源外，KP、平方项、平滑、动态死区、镜像方向均保持原逻辑。
 */
static void ptz_follow_horizontal(const PersonInfo *person,
                                  int target_x,
                                  unsigned int align_mode)
{
    if (person == NULL) return;

    personCenter = person->x + person->w / 2;

    if(gZoomValue > 4) KP_SQUARE = 0.00015f;
    else if(gZoomValue > 3) KP_SQUARE = 0.0002f;
    else if(gZoomValue > 2) KP_SQUARE = 0.00025f;
    else if(gZoomValue > 1.5) KP_SQUARE = 0.00035f;
    else KP_SQUARE = 0.0005f;

    float factor_linear = 1.0f / (1.0f + (gZoomValue - 1.0f) * 0.6f);
    float factor_square = 1.0f / (1.0f + (gZoomValue - 1.0f) * 0.4f);
    float dynamic_KP_LINEAR = KP_LINEAR * factor_linear;
    float dynamic_KP_SQUARE = KP_SQUARE * factor_square;
    float dynamic_ALPHA = SMOOTH_ALPHA / (1.0f + (gZoomValue - 1.0f) * 0.08f);

    int error = personCenter - target_x;
    int absError = abs(error);

#ifdef HD300
    // ===== 原 HD300 小目标/远距离/减速带计算保留 =====
    // 这些变量在原版本中没有参与下面 targetSpeed 计算，这里不擅自改变原算法。
    float targetSizeFactor = 1.0f;
    int objW = person->w;
    if(objW < 160)
    {
        targetSizeFactor = 0.5f + 0.5f * ((float)objW - 40.0f)/(120.0f - 40.0f);
        if(targetSizeFactor < 0.5f) targetSizeFactor = 0.5f;
    }

    const int ERROR_SATURATE = 80;
    float useError = absError;
    if(useError > ERROR_SATURATE) useError = ERROR_SATURATE;

    const int SLOW_DOWN_RANGE = 120;
    float slowFactor = 1.0f;
    if(absError < SLOW_DOWN_RANGE)
    {
        int dz = objW / 4;
        if (dz < 12) dz = 12;
        if (dz > 42) dz = 42;

        if(absError > dz)
        {
            slowFactor = (float)(absError - dz) / (SLOW_DOWN_RANGE - dz);
            if(slowFactor < 0.15f) slowFactor = 0.15f;
        }
        else
        {
            slowFactor = 0.0f;
        }
    }

    (void)targetSizeFactor;
    (void)useError;
    (void)slowFactor;
#endif

    float targetSpeed = 0.0f;
    targetSpeed += (absError * dynamic_KP_LINEAR);
    targetSpeed += ((float)absError * (float)absError * dynamic_KP_SQUARE);

    if (targetSpeed > 0x4f) targetSpeed = 0x4f;
    if (targetSpeed < 0x02) targetSpeed = 0x02;

    float currentSmoothedSpeed =
        lastSmoothedSpeed * (1.0f - dynamic_ALPHA) +
        targetSpeed * dynamic_ALPHA;
    lastSmoothedSpeed = currentSmoothedSpeed;

    unsigned char finalSpeed = (unsigned char)(currentSmoothedSpeed + 0.5f);

    int deadZone = ptz_horizontal_dead_zone(person);

    if (absError <= deadZone) {
        lastSmoothedSpeed = 0.0f;

        // [NEW][PTZ_BLIND_DISTANCE_BY_SPEED]
        // 真实 PTZ 已进入水平停止状态，速度积分必须同步归零。
        ptz_record_pan_stop();

        ptz(7);
        return;
    }

    vL5[4] = finalSpeed;
    vR6[4] = finalSpeed;

    // [NEW] sentSpeedCode 必须记录“最终真正发送的速度码”，包括原有左右构图额外补偿。
    unsigned char sentSpeedCode = finalSpeed;
    int followDirection = (error < 0) ? -1 : 1;

    if (error < 0) {
        if(align_mode == 1) vL5[4] += 0x10;

        if(gMirrorValue == 1) {
            sentSpeedCode = vR6[4];
            ptz(6);
        }
        else {
            sentSpeedCode = vL5[4];
            ptz(5);
        }
    }
    else {
        if(align_mode == 3) vR6[4] += 0x10;

        if(gMirrorValue == 1) {
            sentSpeedCode = vL5[4];
            ptz(5);
        }
        else {
            sentSpeedCode = vR6[4];
            ptz(6);
        }
    }

    // [NEW][PTZ_BLIND_DISTANCE_BY_SPEED]
    // 记录本次实际下发水平速度，供下一帧 blind 距离积分使用。
    ptz_record_pan_motion(sentSpeedCode, followDirection);
}

/*
 * [MOVED][PTZ_BLIND_FOLLOW]
 * 原垂直 PTZ 控制逻辑完整收口。
 * 第一版仅 REAL 调用；BLIND 默认直接停止 tilt。
 */
static void ptz_follow_vertical_real(const PersonInfo *person)
{
    if (person == NULL) return;

    do {
        if(gCfg.u8Speed == 0) break;

        if(gCfg.u32TopSpace) {
            if(gCfg.u32TopSpace == 1) {
                V_BASE_TOP_SPACE_RATIO = 0.10f;
                V_VERTICAL_OFFSET = -20;
            }
            else if(gCfg.u32TopSpace == 2) {
                V_BASE_TOP_SPACE_RATIO = 0.15f;
                V_VERTICAL_OFFSET = 0;
            }
            else if(gCfg.u32TopSpace == 3) {
                V_BASE_TOP_SPACE_RATIO = 0.20f;
                V_VERTICAL_OFFSET = 20;
            }
            gCfg.u32TopSpace = 0;
        }

        float currentZoom = (float)gZoomValue;

        float dynamicTopSpaceRatio =
            V_BASE_TOP_SPACE_RATIO -
            (currentZoom - 1.0f) * V_TOP_SPACE_REDUCTION_FACTOR;
        if (dynamicTopSpaceRatio < 0.05f) dynamicTopSpaceRatio = 0.05f;

        int targetTopY = (int)(AIFR_VIDEO_HEIGHT * dynamicTopSpaceRatio);
        targetTopY += V_VERTICAL_OFFSET;

        int dynamicAlign =
            (int)(V_BASE_ALIGN_HEIGHT *
                  (1.0f + (currentZoom - 1.0f) * 0.15f));

        int windowMinY = targetTopY;
        int windowMaxY = targetTopY + dynamicAlign;
        int windowCenterY = (windowMinY + windowMaxY) / 2;

        float dynKP_Linear = V_KP_LINEAR / powf(currentZoom, 0.8f);
        float dynKP_Square = V_KP_SQUARE / powf(currentZoom, 1.4f);
        float dynAlpha =
            V_STRONG_SMOOTH_ALPHA /
            (1.0f + (currentZoom - 1.0f) * 0.05f);

        bool shouldMove = false;
        int targetForErrorCalculation = 0;

        if (person->y < windowMinY) {
            shouldMove = true;
            targetForErrorCalculation = windowCenterY;
            isTargetLocked = true;
            lockedTargetY = windowCenterY;
        }
        else if (person->y > windowMaxY) {
            shouldMove = true;
            targetForErrorCalculation = windowCenterY;
            isTargetLocked = true;
            lockedTargetY = windowCenterY;
        }
        else {
            if (isTargetLocked) {
                int distanceToTarget = abs(person->y - lockedTargetY);
                if (distanceToTarget < 5) {
                    isTargetLocked = false;
                    shouldMove = false;
                }
                else {
                    shouldMove = true;
                    targetForErrorCalculation = lockedTargetY;
                }
            }
            else {
                shouldMove = false;
            }
        }

        float vTargetSpeed = 0.0f;
        int vError = 0;

        if (shouldMove) {
            vError = person->y - targetForErrorCalculation;
            int absVError = abs(vError);

            vTargetSpeed =
                (absVError * dynKP_Linear) +
                ((float)absVError * (float)absVError * dynKP_Square);

            float maxVSpeed = 0x25 / powf(currentZoom, 0.5f);
            if (maxVSpeed < 0x03) maxVSpeed = 0x03;
            if (vTargetSpeed > maxVSpeed) vTargetSpeed = maxVSpeed;
            if (vTargetSpeed < 0x02) vTargetSpeed = 0x02;
        }
        else {
            vTargetSpeed = 0.0f;
            vError = 0;
            lastVSmoothedSpeed = 0.0f;
            ptz(3);
            break;
        }

        float currentVSmoothedSpeed =
            lastVSmoothedSpeed * (1.0f - dynAlpha) +
            vTargetSpeed * dynAlpha;
        lastVSmoothedSpeed = currentVSmoothedSpeed;

        unsigned char finalVSpeed =
            (unsigned char)(currentVSmoothedSpeed + 0.5f);
        vU1[5] = finalVSpeed;
        vD2[5] = finalVSpeed;

        if (vError < 0) ptz(1);
        else ptz(2);

    } while(0);
}

/*
 * [MOD][PTZ_BLIND_DISTANCE_BY_SPEED]
 * 保存最后一次真实主目标。
 *
 * 注意：blind 期间绝不更新这里；因此从头到尾固定返回的都是最后真实框。
 */
static void ptz_blind_record_real(const PersonInfo *person, unsigned long now_ms)
{
    if (person == NULL) return;

    gBlindCtrl.valid_real = true;
    gBlindCtrl.active = false;
    gBlindCtrl.phase = PTZ_BLIND_IDLE;

    gBlindCtrl.start_ms = 0;
    gBlindCtrl.last_integrate_ms = now_ms;

    gBlindCtrl.last_real = *person;
    gBlindCtrl.blind_person = *person;

    gBlindCtrl.last_real_center_x = person->x + person->w / 2;
    gBlindCtrl.target_x = TARGET_X;
    gBlindCtrl.align_mode = gCfg.u32Align;

    gBlindCtrl.slide_direction = 0;
    gBlindCtrl.travelled_px = 0.0f;
    gBlindCtrl.actual_motion_started = false;
    gBlindCtrl.max_travel_px =
        (float)person->w * PTZ_BLIND_MAX_TRAVEL_BOX_W_RATIO;

    if (gBlindCtrl.max_travel_px < 1.0f)
        gBlindCtrl.max_travel_px = 1.0f;
}

/* [MOD] 结束当前 blind episode，但保留最后真实框，等待真实目标重新出现。 */
static void ptz_blind_clear_episode(void)
{
    gBlindCtrl.active = false;
    gBlindCtrl.phase = PTZ_BLIND_IDLE;
    gBlindCtrl.start_ms = 0;
    gBlindCtrl.last_integrate_ms = 0;
    gBlindCtrl.slide_direction = 0;
    gBlindCtrl.travelled_px = 0.0f;
    gBlindCtrl.actual_motion_started = false;
}

/*
 * [NEW][PTZ_BLIND_DISTANCE_BY_SPEED]
 * 开始一次 blind slide。
 *
 * 这里不生成任何新位置：blind_person 永远就是 last_real。
 */
static bool ptz_blind_start(unsigned long now_ms)
{
    if (!gBlindCtrl.valid_real) return false;

    const int error =
        gBlindCtrl.last_real_center_x - gBlindCtrl.target_x;
    const int deadZone = ptz_horizontal_dead_zone(&gBlindCtrl.last_real);

    // 最后真实目标已经在水平死区内：没有继续 blind 滑动的意义。
    if (abs(error) <= deadZone) {
        gBlindCtrl.active = false;
        gBlindCtrl.phase = PTZ_BLIND_STOPPED;
        gBlindCtrl.slide_direction = 0;
        return false;
    }

    gBlindCtrl.active = true;
    gBlindCtrl.phase = PTZ_BLIND_SLIDING;
    gBlindCtrl.start_ms = now_ms;
    gBlindCtrl.last_integrate_ms = now_ms;

    gBlindCtrl.blind_person = gBlindCtrl.last_real;
    gBlindCtrl.slide_direction = (error < 0) ? -1 : 1;

    gBlindCtrl.travelled_px = 0.0f;
    gBlindCtrl.actual_motion_started = false;
    gBlindCtrl.max_travel_px =
        (float)gBlindCtrl.last_real.w * PTZ_BLIND_MAX_TRAVEL_BOX_W_RATIO;

    if (gBlindCtrl.max_travel_px < 1.0f)
        gBlindCtrl.max_travel_px = 1.0f;

    if(positionLog) {
        printf(
            "[PTZ_BLIND_DISTANCE] action=start real_box=(%d,%d,%d,%d) "
            "real_cx=%d target_x=%d direction=%d max_travel_px=%.2f "
            "speed_code=%u speed_px_s=%.2f\n",
            gBlindCtrl.last_real.x,
            gBlindCtrl.last_real.y,
            gBlindCtrl.last_real.w,
            gBlindCtrl.last_real.h,
            gBlindCtrl.last_real_center_x,
            gBlindCtrl.target_x,
            gBlindCtrl.slide_direction,
            gBlindCtrl.max_travel_px,
            (unsigned int)gPanSpeedCode,
            gPanSpeedPxPerSec);
    }

    return true;
}

/*
 * [NEW][PTZ_BLIND_DISTANCE_BY_SPEED]
 * blind 的核心：固定返回最后真实 box，只计算“还能不能继续返回”。
 *
 * 返回 true：
 *     *out_person == last_real，继续用这个固定 box 驱动 PTZ。
 *
 * 返回 false：
 *     已达到一个 BODY 宽度、方向异常、failsafe 超时或无可靠 last_real；
 *     上层停止 PTZ，不再提供 blind box。
 */
static bool ptz_blind_get_fixed_person(unsigned long now_ms, PersonInfo *out_person)
{
    if (out_person == NULL || !gBlindCtrl.valid_real)
        return false;

    // STOPPED 是本次“真实目标丢失”episode 的终态，不能仅因 active=false
    // 就再次 ptz_blind_start()。只有新的真实 BODY 经 ptz_blind_record_real()
    // 将 phase 恢复为 IDLE 后，下一次丢失才允许开始新的 blind slide。
    // 否则 distance_limit 后的每个 LOST frame 都会重置 travelled_px 并重复滑动。
    if (gBlindCtrl.phase == PTZ_BLIND_STOPPED)
        return false;

    if (!gBlindCtrl.active) {
        if (!ptz_blind_start(now_ms))
            return false;
    }

    unsigned long dt_ms = now_ms - gBlindCtrl.last_integrate_ms;
    gBlindCtrl.last_integrate_ms = now_ms;

    // 调度异常保护：避免偶发长停顿一次性积分出巨大距离。
    if (dt_ms > 200UL)
        dt_ms = 200UL;

    const float dt_sec = (float)dt_ms / 1000.0f;

    // ------------------------------------------------------------------------
    // [NEW] 只累计“仍沿丢失瞬间跟随方向”的水平运动。
    // 如果控制方向发生反转，说明原 blind continuation 语义已经失效，直接结束。
    // ------------------------------------------------------------------------
    if (gPanFollowDirection != 0 &&
        gPanFollowDirection != gBlindCtrl.slide_direction) {

        gBlindCtrl.active = false;
        gBlindCtrl.phase = PTZ_BLIND_STOPPED;

        if(positionLog) {
            printf(
                "[PTZ_BLIND_DISTANCE] action=stop reason=direction_changed "
                "blind_dir=%d current_dir=%d travelled_px=%.2f max_px=%.2f\n",
                gBlindCtrl.slide_direction,
                gPanFollowDirection,
                gBlindCtrl.travelled_px,
                gBlindCtrl.max_travel_px);
        }

        return false;
    }

    // speed_px_s 表示上一控制周期真正下发的 pan 速度对应的图像等效速度。
    if (gPanFollowDirection == gBlindCtrl.slide_direction &&
        gPanSpeedPxPerSec > 0.0f && dt_sec > 0.0f) {

        const float delta_px = fabsf(gPanSpeedPxPerSec) * dt_sec;
        gBlindCtrl.travelled_px += delta_px;
        if (delta_px > 0.0f)
            gBlindCtrl.actual_motion_started = true;
    }

    // ------------------------------------------------------------------------
    // 主停止条件：实际/估计滑动距离达到最后真实 BODY box 宽度。
    // ------------------------------------------------------------------------
    if (gBlindCtrl.travelled_px >= gBlindCtrl.max_travel_px) {
        gBlindCtrl.active = false;
        gBlindCtrl.phase = PTZ_BLIND_STOPPED;

        if(positionLog) {
            printf(
                "[PTZ_BLIND_DISTANCE] action=stop reason=distance_limit "
                "travelled_px=%.2f max_px=%.2f box_w=%d age_ms=%lu\n",
                gBlindCtrl.travelled_px,
                gBlindCtrl.max_travel_px,
                gBlindCtrl.last_real.w,
                now_ms - gBlindCtrl.start_ms);
        }

        return false;
    }

    // ------------------------------------------------------------------------
    // 仅做故障保护，不作为正常滑动控制条件。
    // ------------------------------------------------------------------------
    if (now_ms - gBlindCtrl.start_ms >= PTZ_BLIND_FAILSAFE_MS) {
        gBlindCtrl.active = false;
        gBlindCtrl.phase = PTZ_BLIND_STOPPED;

        if(positionLog) {
            printf(
                "[PTZ_BLIND_DISTANCE] action=stop reason=failsafe_timeout "
                "travelled_px=%.2f max_px=%.2f age_ms=%lu speed_px_s=%.2f\n",
                gBlindCtrl.travelled_px,
                gBlindCtrl.max_travel_px,
                now_ms - gBlindCtrl.start_ms,
                gPanSpeedPxPerSec);
        }

        return false;
    }

    // ------------------------------------------------------------------------
    // 最关键 invariant：blind box 从头到尾完全等于最后真实匹配 box。
    // x/y/w/h 全部不改。
    // ------------------------------------------------------------------------
    gBlindCtrl.blind_person = gBlindCtrl.last_real;
    *out_person = gBlindCtrl.last_real;

    if(positionLog) {
        printf(
            "[PTZ_BLIND_DISTANCE] action=output age_ms=%lu dt_ms=%lu "
            "box=(%d,%d,%d,%d) speed_code=%u speed_px_s=%.2f "
            "travelled_px=%.2f max_px=%.2f\n",
            now_ms - gBlindCtrl.start_ms,
            dt_ms,
            out_person->x,
            out_person->y,
            out_person->w,
            out_person->h,
            (unsigned int)gPanSpeedCode,
            gPanSpeedPxPerSec,
            gBlindCtrl.travelled_px,
            gBlindCtrl.max_travel_px);
    }

    return true;
}

/* [MOD] blind 完成或无可靠历史时，明确停止水平/垂直运动。 */
static void ptz_blind_stop_motion(void)
{
    lastSmoothedSpeed = 0.0f;
    lastVSmoothedSpeed = 0.0f;
    isTargetLocked = false;

    // [NEW] 同步停止速度积分状态。
    ptz_record_pan_stop();

    ptz(7); // pan stop
    ptz(3); // tilt stop
}

/*
 * ==========================================================================
 * [MOD][PUBLIC PTZ ENTRY] 唯一 PTZ 跟随入口
 * ==========================================================================
 *
 * 核心调度层仍然只需要：
 *
 *     ptz_follow_update(real_person);  // 有真实主目标
 *     ptz_follow_update(NULL);         // 本帧真实主目标丢失
 *
 * REAL：
 *   - 正常 auto zoom / pan / tilt；
 *   - 保存最后真实 box；
 *   - 保存当前真实 pan speed，供随后 blind 积分。
 *
 * LOST：
 *   - 不改变 box；
 *   - 每帧固定使用 last_real box；
 *   - 通过当前 PTZ pan speed × dt 累计等效滑动距离；
 *   - 距离达到 1×last_real.w 后停止返回 blind box并停止 pan；
 *   - blind 禁止 auto zoom，第一版停止 tilt；
 *   - PTZ_BLIND_FAILSAFE_MS 仅做异常兜底。
 */
int ptz_follow_update(const PersonInfo *real_person)
{
    unsigned long now_ms = get_millis();

    // ------------------------------------------------------------------------
    // REAL: 当前帧有真实主目标
    // ------------------------------------------------------------------------
    if (real_person != NULL) {
        unsigned long previous_lost_ms = g_lost_duration_ms;
        float previous_blind_travel = gBlindCtrl.travelled_px;

        ptz_blind_clear_episode();
        lose_start_ms = 0;
        g_lost_duration_ms = 0;

        // 真实目标恢复后，恢复真实 PTZ 逻辑。
        ptz_refresh_target_x_from_real(real_person);

#if FX_DEBUG
        open_light(2);
#endif

        // 原自动变倍只允许真实目标使用。
        ptz_follow_auto_zoom_real(real_person);

        // 水平按真实 TARGET_X 控制；函数内部会记录最终实际下发 speed code。
        ptz_follow_horizontal(real_person, TARGET_X, gCfg.u32Align);

        // 垂直仍完全使用原真实控制规则。
        ptz_follow_vertical_real(real_person);

        // [MOD] 必须在真实水平 PTZ 控制执行后记录，保证当前 pan speed 已更新。
        ptz_blind_record_real(real_person, now_ms);

        // 主场模式原有“记录主场”行为也收口到统一接口。
        if(gCfg.u8Mode == 2 && onefaceMatchOK == 1){
            gCb.send_visca_cmd(vPreset1);
            onefaceMatchOK = 0;
        }

        if(positionLog && previous_lost_ms > 0) {
            int real_cx = real_person->x + real_person->w / 2;
            printf(
                "[PTZ_BLIND_DISTANCE] action=reacquired lost_ms=%lu "
                "blind_travel_px=%.2f real_cx=%d target_x=%d\n",
                previous_lost_ms,
                previous_blind_travel,
                real_cx,
                TARGET_X);
        }

        return PTZ_FOLLOW_REAL;
    }

    // ------------------------------------------------------------------------
    // LOST: 当前帧没有真实主目标
    // ------------------------------------------------------------------------
    if (lose_start_ms == 0)
        lose_start_ms = now_ms;

    g_lost_duration_ms = now_ms - lose_start_ms;

    // 保留原 8 秒总丢失超时语义。
    if (g_lost_duration_ms >= TOTAL_LOSE_TIMEOUT_MS) {
        ptz_blind_stop_motion();
        stop_face();
        return PTZ_FOLLOW_STOPPED;
    }

    // blind 阶段绝不能继续累计真实尺寸条件，避免恢复后立即误触发 zoom。
    gIsWidthConditionMet = 0;
    gWidthStartTime = 0;

    PersonInfo blind_person;

    if (ptz_blind_get_fixed_person(now_ms, &blind_person)) {
        // --------------------------------------------------------------------
        // [MOD] BLIND：从头到尾都是最后真实框，没有任何位置衰减/外推。
        // --------------------------------------------------------------------

        // 禁止自动变倍。

        // 水平继续使用丢失瞬间冻结的 TARGET_X。
        // 此调用会得到本控制周期新的 finalSpeed，并更新 gPanSpeedPxPerSec，
        // 下一帧用它积分本周期实际继续滑动的距离。
        ptz_follow_horizontal(&blind_person,
                              gBlindCtrl.target_x,
                              gBlindCtrl.align_mode);

#if PTZ_BLIND_ENABLE_TILT
        ptz_follow_vertical_real(&blind_person);
#else
        // 第一版：最后真实 y 不被当作当前真实垂直位置，直接停止 tilt。
        lastVSmoothedSpeed = 0.0f;
        isTargetLocked = false;
        ptz(3);
#endif

        return PTZ_FOLLOW_BLIND;
    }

    // 已达到距离阈值 / failsafe / 无 last_real / 方向异常。
    // 从此不再使用 blind box 驱动 PTZ。
    ptz_blind_stop_motion();
    return PTZ_FOLLOW_STOPPED;
}

// ============================================================================
// [MOD][PTZ_BLIND_DISTANCE_BY_SPEED] 统一 PTZ 跟随实现 END
// ============================================================================



#pragma endregion

#pragma endregion fxw


void * aiface_thread()
{	
    // 1.0.2 : 模式不同时间丢失帧数时间不同
    // 1.0.3 : 水平停止多发几次，引入 sotpC
    // 1.0.4 : KP_SQUARE 由 0.0005f -> 0.0002f
    printf("======== [aitrack 1-ptz] v1.6 t4  ======== \n");

    // unsigned char *pBGRData = NULL;
    int s32MilliSec = 200;
    PersonInfo* person;
	const char* model_dir = "/oem/model";

    //fx260626 start
  #if FX_CHIP == 2 // dv500
    const char* so_path = "/oem/lib/opencv/libaiDetect.so";
  #else // ss927
    const char* so_path = "/oem/opencv/libaiDetect.so";
  #endif
    printf("[libaitrack] so_path=%s \n", so_path);

    void* handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) RETURN_NULL("dlopen %s failed", so_path);

    const char* err = dlerror();

    // TrackHandle* track_create();
    typedef TrackHandle* (*fn_track_create_t)(void);
    static fn_track_create_t track_create = NULL; 
    track_create = (fn_track_create_t)dlsym(handle, "track_create");
    err = dlerror();
    if(err){
        printf("dlsym track_create err: %s\n", err);
        dlclose(handle);
        handle = NULL;
        return NULL;
    }
    
    // void track_init(TrackHandle* handle);
    typedef void (*fn_track_init_t)(TrackHandle*);
    static fn_track_init_t track_init = NULL;
    track_init = (fn_track_init_t)dlsym(handle, "track_init");
    err = dlerror();
    if(err){
        printf("dlsym track_init err: %s\n", err);
        dlclose(handle);
        handle = NULL;
        return NULL;
    }

    // int track_run(TrackHandle* handle, TrackInput* input, OutResult* out_result);
    typedef int (*fn_track_run_t)(TrackHandle*, TrackInput*, OutResult*);
    static fn_track_run_t track_run = NULL;
    track_run = (fn_track_run_t)dlsym(handle, "track_run");
    err = dlerror();
    if(err){
        printf("dlsym track_run err: %s\n", err);
        dlclose(handle);
        handle = NULL;
        return NULL;
    }
    //fx260626 end

    TrackHandle* track = track_create();
    if (!track) { printf("=== [aiface_thread puas-track] Failed to create track handle\n"); return NULL; }
    track_init(track);
    memset(&gCfg, 0, sizeof(GSF_AI_CFG_S));
    gsf_aifr_get_param(0, &gCfg);
    
    OutResult ret;
    MainTarget mTarget = {0};
    TrackInput trackInput = {
        .pBGRData = NULL,
        .width = AIFR_VIDEO_WIDTH,
        .height = AIFR_VIDEO_HEIGHT,
        .mainTarget = &mTarget,
        // .flipValue = gFlipValue
        .zoomValue = gZoomValue,
        .ptz_blind_phase = TRACK_PTZ_BLIND_IDLE
    };
    int tracker_blind_phase = TRACK_PTZ_BLIND_IDLE;


    // [MOD][PTZ_BLIND_FOLLOW] 自动变倍状态已收口到 ptz_follow_update() 内部。

    usleep(1500000);

    #if FX_DEBUG
        open_light(1);
    #endif

    // 两个派系： 1. 毛主管： 主场模式    2. 余总： 自动变倍模式

    gCb.send_visca_cmd(vPreset0); // 清除预置位

  #if FX_CHIP == 2    
    // gCb.pfn_start_rect(); //fx060627
  #endif
    // fxw debug
    // unsigned long start_ms, end_ms, end_ms2;
    // unsigned long stat_start = get_millis();
    // int loop_cnt = 0;

	while (taskRunFlag != 0){

        // [MOD][PTZ_BLIND_FOLLOW] 每帧先清空，禁止沿用上一帧真实 person 指针。
        person = NULL;

        // usleep(500000); continue;
        // start_ms = get_millis(); // fxw debug


        #pragma region 获取帧 以及失败处理🚩

        // printf("===> gCfg.u8Enable=%u gCb.read_mode()=%d \n", gCfg.u8Enable, gCb.read_mode() );


        if(gCb.read_mode() == 0 && gCfg.u8Enable){ // 遥控器/发指令 关闭
            // 网页同步关闭
            gCfg.u8Enable = 0;
            gsf_aifr_set_param_enable(0);
        }
        else if(gCb.read_mode() == 1 && !gCfg.u8Enable){
            // 网页同步开启
            gCfg.u8Enable = 1;
            gsf_aifr_set_param_enable(1);
            
        }

		if(gCb.read_mode==NULL || gCb.read_mode()==0 || !gCfg.u8Enable){  // 发指令/遥控 ... 休眠

            if(gCfg.u8Mode == 2){
                if(gCb.read_mode==NULL || gCb.read_mode()==0) pressRemoteControl = 1; // fx260328_1 按下遥控器
            }

            stop_face();
            usleep(500000);
            continue; 
        }
		
        // 20260313-1
		// trackInput.pBGRData = aiav_get_bgr_frame_addr(s32MilliSec);
        trackInput.pBGRData = get_safe_bgr_copy(s32MilliSec);
        if(trackInput.pBGRData == NULL){ prepare_stop_face(); continue; }

        // trackInput.flipValue = gFlipValue;
        trackInput.zoomValue = gZoomValue;
        trackInput.ptz_blind_phase = tracker_blind_phase;
        // track_run(track, pBGRData, &ret, &mainTarget);
        track_run(track, &trackInput, &ret);
        if(trackInput.mainTarget->x !=0 || trackInput.mainTarget->x2 != 0){
            trackInput.mainTarget->x = 0;
            trackInput.mainTarget->y = 0;
            trackInput.mainTarget->x2 = 0;
            trackInput.mainTarget->y2 = 0;
        }

        // fxw debug
        // end_ms = get_millis();
        // unsigned long single_cost = end_ms - start_ms; // track_run 运行的时间

        // 20260313-1
        // aiav_bgr_release_frame();
        free(trackInput.pBGRData);

        printf("[aiface_thread] ===== ret.count=%d \n", ret.count);
        if(ret.count <= 0){
            // =================================================================
            // [MOD][PTZ_BLIND_FOLLOW]
            // 没有任何检测框也属于“本帧无真实主目标”，必须让统一 PTZ 接口
            // 继续“固定最后真实框 + 速度积分限距”，而不是 prepare_stop_face() 立即发送 vS8 停止。
            // =================================================================
            ptz_follow_update(NULL);
            tracker_blind_phase = ptz_tracker_blind_phase();
            continue;
        }

        #pragma endregion 

        #pragma region 比对人脸ID 🚩

        if(gCfg.u32Px || gCfg.u32Py){ // 如果网页传入了 人脸 数据
            
            if(gFlipValue == 1 && gMirrorValue != 1){ // 上下左右都颠倒
                gCfg.u32Px = AIFR_VIDEO_WIDTH - gCfg.u32Px;
            }
            else if(gMirrorValue == 1 && gFlipValue != 1){ // 左右颠倒
                gCfg.u32Px = AIFR_VIDEO_WIDTH - gCfg.u32Px;
            }


            for (int j = 0; j < ret.count; ++j) {

                if( (gCfg.u32Px > ret.infos[j].x && gCfg.u32Px < ret.infos[j].x + ret.infos[j].w) && (gCfg.u32Py > ret.infos[j].y && gCfg.u32Py < ret.infos[j].y + ret.infos[j].h) ){ // 这样就表示 x,y 坐标 在人脸区域
                    // person = ret.infos[j];
                    gCfg.u32Px = 0; gCfg.u32Py = 0;

                    // fx2026-1-29-1
                    trackInput.mainTarget->x = ret.infos[j].x;
                    trackInput.mainTarget->y = ret.infos[j].y;
                    trackInput.mainTarget->x2 =  ret.infos[j].x + ret.infos[j].w;
                    trackInput.mainTarget->y2 = ret.infos[j].y + ret.infos[j].h;
                    faceMatchOK = 1;
                    if(gCfg.u8Mode == 2) onefaceMatchOK = 1; // fx260328_1

                    // fx260826
                    lose_start_ms = 0;
                    g_lost_duration_ms = 0;

                    break;
                }
            }
            gCfg.u32Px = 0; gCfg.u32Py = 0;

            continue; // fx2026-1-29-1
        }  

        // 丢帧达到一定数量 自动选择置信度最高的为主目标
        if(gCfg.u8Mode == 2){
            // 主场模式必须要按遥控器才行奥 注意 && pressRemoteControl == 1
            if(!faceMatchOK && g_lost_duration_ms >= AUTO_PICK_TIMEOUT_MS && pressRemoteControl == 1){ // fx260328_1 如果没有匹配成功，选第 0 个作为主目标
                trackInput.mainTarget->x = ret.infos[0].x;
                trackInput.mainTarget->y = ret.infos[0].y;
                trackInput.mainTarget->x2 = ret.infos[0].x + ret.infos[0].w;
                trackInput.mainTarget->y2 = ret.infos[0].y + ret.infos[0].h;
                faceMatchOK = 1;
                onefaceMatchOK = 1; // fx260328_1
                pressRemoteControl = 0; // fx260328_1

                // fx260826
                lose_start_ms = 0;
                g_lost_duration_ms = 0;
                continue;
            }
        }
        else{
            if(!faceMatchOK && g_lost_duration_ms >= AUTO_PICK_TIMEOUT_MS) {
                trackInput.mainTarget->x = ret.infos[0].x;
                trackInput.mainTarget->y = ret.infos[0].y;
                trackInput.mainTarget->x2 = ret.infos[0].x + ret.infos[0].w;
                trackInput.mainTarget->y2 = ret.infos[0].y + ret.infos[0].h;
                faceMatchOK = 1;

                // fx260826
                lose_start_ms = 0; 
                g_lost_duration_ms = 0;
                continue;
            }
        }
        
        
        // if(!faceMatchOK) continue;

        for (int j = 0; j < ret.count; j++) { // 获取主目标(id:1) 的 坐标信息, 如果没有 faceMatchOK=0

            // if(positionLog) printf("[index:%d] id:%d \n", j, ret.infos[j].id);

            if(ret.infos[j].id == 1){ // 主目标的 id 一直是 1
                if(!faceMatchOK) faceMatchOK = 1;

                // [MOD][PTZ_BLIND_FOLLOW] 这里只负责拿到“真实主目标”指针。
                // 丢失计时清零、主场预置位、PTZ 控制、last_real 保存都由统一接口负责。
                person = &ret.infos[j];

                break;
            }else { // 设置了主目标就应该有 id 0, 如果没有就重新设置
                if(faceMatchOK) faceMatchOK = 0;
            }
        }
                
        if(!faceMatchOK){
            // =================================================================
            // [MOD][PTZ_BLIND_FOLLOW]
            // 当前检测可能还有其他人，但没有真实主目标 id=1：交给统一接口进入 blind。
            // =================================================================
            ptz_follow_update(NULL);
            tracker_blind_phase = ptz_tracker_blind_phase();
            continue;
        }
        
        #pragma endregion

        //fx060627
      #if FX_CHIP == 2  
        // gRect.s32X = person->x*drawScale;
        // gRect.s32Y = person->y*drawScale;
        // gRect.u32Width = person->w*drawScale;
        // gRect.u32Height = person->h*drawScale;
        // gCb.pfn_set_rect(1, &gRect);
      #endif


        // =====================================================================
        // [MOD][PTZ_BLIND_FOLLOW] PTZ 所有控制已收口到一个接口。
        // 核心调度函数不再展开自动变倍 / 水平 / 垂直控制细节。
        // 当前 person 来自 ret.infos[id==1]，因此这里明确是真实主目标。
        // =====================================================================
        ptz_follow_update(person);
        tracker_blind_phase = ptz_tracker_blind_phase();


        // fxw debug
        // end_ms2 = get_millis();
        // unsigned long single_cost2 = end_ms2 - end_ms; // track_run 运行之后运行的时间
        // printf("track_run运行时间: %lu ms, 到结束时间： %lu deadZone=%d \n", single_cost, single_cost2, deadZone);
        // loop_cnt++;
        // if(end_ms2 - stat_start >= 1000){
        //     printf("最近1s总循环次数：%d\n", loop_cnt);
        //     stat_start = end_ms2;
        //     loop_cnt = 0;
        // }


        // usleep(400000);
	}

  #if FX_CHIP == 2       
    // gCb.pfn_stop_rect(); //fx060627
  #endif

	pthread_join(gAifrPid, NULL);
}

int gsf_aifr_init(void)
{

	int s32Ret = 0;
	
	memset(&gAifrMan, 0, sizeof(GSF_AIFR_MAN));
	
	//CbFuncs Cb;
	//Cb.read_mode = get_humen_tracking_mode;
   	//Cb.send_visca_cmd = gsf_uart_send_visca;
   	
   	//gsf_set_ai_function(&Cb);
   	
   	aiav_video_start(AIFR_VIDEO_WIDTH, AIFR_VIDEO_HEIGHT, AIFR_SRC_RATE, AIFR_DST_RATE);
   	
	taskRunFlag = 1;
	pthread_create(&gAifrPid, NULL, aiface_thread, (void *)NULL);
	
        
	gAifrMan.u8Init = 1;
	printf("gsf AIFR man init\n");
	
	return 0;
}

int gsf_aifr_uninit(void)
{

    // [NEW][PTZ_BLIND_FOLLOW] 模块退出时清理 blind 状态。
    memset(&gBlindCtrl, 0, sizeof(gBlindCtrl));

	// sxs_bodydetect_destroy(&ctx);
	aiav_video_stop();
	if(taskRunFlag)
	{
		taskRunFlag = 0;
		pthread_join(gAifrPid, NULL);
	}
	
	gAifrMan.u8Init = 0;

	return 0;
}

int gsf_aifr_refresh(int nCh, void *pCfg)
{
}

// #endif

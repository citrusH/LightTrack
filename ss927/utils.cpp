#include "utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#if defined(_MSC_VER)
#include <windows.h>
#else
#include <sys/stat.h>
#endif
#include "acl/acl.h"
#include <numeric>
#include <sstream>
#include <iomanip>


using namespace cv;
using namespace std;

Result Utils::ReadBinFile(const std::string& fileName, void*& inputBuff, uint32_t& fileSize)
{
	if (CheckPathIsFile(fileName) == FAILED) {
		ERROR_LOG("%s is not a file", fileName.c_str());
		return FAILED;
	}

	std::ifstream binFile(fileName, std::ifstream::binary);
	if (binFile.is_open() == false) {
		ERROR_LOG("open file %s failed", fileName.c_str());
		return FAILED;
	}

	binFile.seekg(0, binFile.end);
	uint32_t binFileBufferLen = binFile.tellg();
	if (binFileBufferLen == 0) {
		ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
		binFile.close();
		return FAILED;
	}
	binFile.seekg(0, binFile.beg);

	inputBuff = std::malloc(binFileBufferLen);
	if (inputBuff == nullptr) {
		ERROR_LOG("malloc host bin buffer failed, size=%u", binFileBufferLen);
		binFile.close();
		return FAILED;
	}
	binFile.read(static_cast<char*>(inputBuff), binFileBufferLen);
	binFile.close();
	fileSize = binFileBufferLen;
	return SUCCESS;
}

Result Utils::MemcpyHostToDevice(const void* host_data,
                                 void* device_buffer,
                                 size_t size)
{
    if (host_data == nullptr || device_buffer == nullptr || size == 0) {
        ERROR_LOG("host-to-device memcpy received invalid buffer, size=%zu", size);
        return FAILED;
    }

    aclError ret = aclrtMemcpy(device_buffer, size, host_data, size,
                               ACL_MEMCPY_HOST_TO_DEVICE);

    if (ret != ACL_SUCCESS) {
        ERROR_LOG("memcpy failed. size=%zu, error=%d",
                  size, static_cast<int32_t>(ret));
        return FAILED;
    }

    return SUCCESS;
}

Result Utils::CheckPathIsFile(const std::string& fileName)
{
	#if defined(_MSC_VER)
		DWORD bRet = GetFileAttributes((LPCSTR)fileName.c_str());
		if (bRet == FILE_ATTRIBUTE_DIRECTORY) {
			ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
			return FAILED;
		}
	#else
		struct stat sBuf;
		int fileStatus = stat(fileName.data(), &sBuf);
		if (fileStatus == -1) {
			ERROR_LOG("failed to get file");
			return FAILED;
		}
		if (S_ISREG(sBuf.st_mode) == 0) {
			ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
			return FAILED;
		}
	#endif
		return SUCCESS;
}


// 余弦距离计算
float Utils::cosine_distance(const std::vector<float>& emb1, const std::vector<float>& emb2) {
    if (emb1.size() != emb2.size()) {
        throw std::invalid_argument("Embedding dimensions must match");
    }

    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < emb1.size(); ++i) {
        dot_product += emb1[i] * emb2[i];
        norm1 += emb1[i] * emb1[i];
        norm2 += emb2[i] * emb2[i];
    }

    norm1 = std::sqrt(norm1);
    norm2 = std::sqrt(norm2);

    if (norm1 < 1e-10f || norm2 < 1e-10f) {
        return 1.0f;  // 返回最大距离
    }

    float cosine_sim = dot_product / (norm1 * norm2);
    cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));

    float cosine_dist = 1.0f - cosine_sim;
    // std::cout << "emb sim : " << cosine_sim << std::endl;
    return cosine_dist;
}

// 单个IOU计算
/*
float Utils::iou_single(const cv::Mat& bbox1, const cv::Mat& bbox2) {
    const float* data1 = bbox1.ptr<float>(0);
    float x1 = data1[0];
    float y1 = data1[1];
    float w1 = data1[2];
    float h1 = data1[3];

    // 提取 bbox2 的参数
    const float* data2 = bbox2.ptr<float>(0);
    float x2 = data2[0];
    float y2 = data2[1];
    float w2 = data2[2];
    float h2 = data2[3];

    float xx1 = std::max(x1, x2);
    float yy1 = std::max(y1, y2);
    float xx2 = std::min(x1 + w1, x2 + w2);
    float yy2 = std::min(y1 + h1, y2 + h2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area1 = w1 * h1;
    float area2 = w2 * h2;

    float iou = inter / (area1 + area2 - inter + 1e-6f);
    return iou;
}
*/

// 根据 bbox 推断身体可见度
EmbQualityMetrics Utils::estimate_emb_quality(
    const cv::Mat& bbox,          // [x,y,w,h]
    const cv::Size& frame_size,
    float det_confidence)
{
    EmbQualityMetrics m;

    float w = bbox.at<float>(0, 2);
    float h = bbox.at<float>(0, 3);
    float area = w * h;
    float frame_area = (float)(frame_size.width * frame_size.height);

    // ── 1. 尺寸得分：太小的框特征不可靠 ──────────────
    float area_ratio = area / frame_area;
    m.size_score = std::min(1.0f, area_ratio / 0.02f); // 占画面2%以上满分

    // ── 2. 宽高比得分：人体正常比例 w:h ≈ 1:2~1:3 ──
    float aspect = (h > 0) ? (w / h) : 0;
    // 全身约 0.3~0.5，半身约 0.5~0.8，超出范围扣分
    if (aspect >= 0.25f && aspect <= 0.55f) {
        // 全身或接近全身
        m.aspect_score = 1.0f;
        m.visibility_ratio = 1.0f;
    } else if (aspect > 0.55f && aspect <= 0.85f) {
        // 半身
        m.aspect_score = 0.6f;
        m.visibility_ratio = 0.5f;
    } else if (aspect > 0.85f) {
        // 只有头/胸，特征很不稳定
        m.aspect_score = 0.2f;
        m.visibility_ratio = 0.25f;
    } else {
        // aspect < 0.25，极度细长，异常
        m.aspect_score = 0.3f;
        m.visibility_ratio = 0.3f;
    }

    // ── 3. 综合得分 ───────────────────────────────
    m.final_score = m.size_score * 0.3f
                  + m.aspect_score * 0.5f
                  + det_confidence * 0.2f;

    return m;
}

float Utils::iou_single(const cv::Mat& bbox1, const cv::Mat& bbox2) {
    // 提取 bbox1 的参数 (x1, y1, x2, y2)
    const float* data1 = bbox1.ptr<float>(0);
    float x1_1 = data1[0];
    float y1_1 = data1[1];
    float x2_1 = data1[2]; // 直接使用右下角x坐标
    float y2_1 = data1[3]; // 直接使用右下角y坐标

    // 提取 bbox2 的参数 (x1, y1, x2, y2)
    const float* data2 = bbox2.ptr<float>(0);
    float x1_2 = data2[0];
    float y1_2 = data2[1];
    float x2_2 = data2[2]; // 直接使用右下角x坐标
    float y2_2 = data2[3]; // 直接使用右下角y坐标

    float xx1 = std::max(x1_1, x1_2);
    float yy1 = std::max(y1_1, y1_2);
    float xx2 = std::min(x2_1, x2_2); // 直接使用x2
    float yy2 = std::min(y2_1, y2_2); // 直接使用y2

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    float area2 = (x2_2 - x1_2) * (y2_2 - y1_2);

    float iou = inter / (area1 + area2 - inter + 1e-6f);
    return iou;
}



// 批量IOU计算
cv::Mat Utils::iou_batch(const cv::Mat& bboxes1, const cv::Mat& bboxes2) {
    //CV_Assert(bboxes1.type() == CV_32FC1 && bboxes2.type() == CV_32FC1);
    //CV_Assert(bboxes1.cols == 4 && bboxes2.cols == 4);

    int N = bboxes1.rows;
    int M = bboxes2.rows;

    cv::Mat iou_matrix(N, M, CV_32FC1);

    // 预计算面积以提高性能
    std::vector<float> areas1(N);
    std::vector<float> areas2(M);

    for (int i = 0; i < N; ++i) {
        const float* b1 = bboxes1.ptr<float>(i);
        areas1[i] = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    }

    for (int j = 0; j < M; ++j) {
        const float* b2 = bboxes2.ptr<float>(j);
        areas2[j] = (b2[2] - b2[0]) * (b2[3] - b2[1]);
    }

    for (int i = 0; i < N; ++i) {
        const float* b1 = bboxes1.ptr<float>(i);
        float x1_i = b1[0];
        float y1_i = b1[1];
        float x2_i = b1[2];
        float y2_i = b1[3];

        float* iou_row = iou_matrix.ptr<float>(i);

        for (int j = 0; j < M; ++j) {
            const float* b2 = bboxes2.ptr<float>(j);
            float x1_j = b2[0];
            float y1_j = b2[1];
            float x2_j = b2[2];
            float y2_j = b2[3];

            float xx1 = std::max(x1_i, x1_j);
            float yy1 = std::max(y1_i, y1_j);
            float xx2 = std::min(x2_i, x2_j);
            float yy2 = std::min(y2_i, y2_j);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;

            float union_area = areas1[i] + areas2[j] - inter;
            iou_row[j] = (union_area > 0) ? inter / union_area : 0.0f;
        }
    }

    return iou_matrix;
}

cv::Rect_<float> Utils::convert_z_to_bbox(const cv::Mat& z) {
    // 检查输入维度，确保是4维观测向量
    //CV_Assert(z.type() == CV_32FC1 && z.total() == 4);

    const float* data = z.ptr<float>(0);
    float cx = data[0];    // 中心点x坐标
    float cy = data[1];    // 中心点y坐标
    float w = data[2];     // 宽度（直接使用w而不是a*h）
    float h = data[3];     // 高度

    cv::Rect_<float> bbox;
    bbox.x = cx - w / 2.0f;      // 左上角x坐标
    bbox.y = cy - h / 2.0f;      // 左上角y坐标
    bbox.width = bbox.x + w;              // 宽度
    bbox.height = bbox.y + h;             // 高度

    return bbox;
}

cv::Mat Utils::convert_bbox_to_z(const cv::Mat& bbox) {
    // bbox格式: [x, y, x, y] 
    //std::cout << "convert bbox to z : " << bbox << std::endl;
    float x = bbox.at<float>(0, 0);
    float y = bbox.at<float>(0, 1);
    float w = bbox.at<float>(0, 2) - bbox.at<float>(0, 0);
    float h = bbox.at<float>(0, 3) - bbox.at<float>(0, 1);

    float cx = x + w / 2.0f;  // 中心点x
    float cy = y + h / 2.0f;  // 中心点y

    // 返回4维观测向量: [cx, cy, w, h]
    cv::Mat z = (cv::Mat_<float>(4, 1) << cx, cy, w, h);
    return z;
}
// 计算速度方向
std::pair<cv::Vec2f, float> Utils::speed_direction(const cv::Mat& bbox1, const cv::Mat& bbox2) {
    // 提取 bbox1 的参数 (x1, y1, x2, y2)
    const float* data1 = bbox1.ptr<float>(0);
    float x1_1 = data1[0];
    float y1_1 = data1[1];
    float x2_1 = data1[2]; // 这是右下角x坐标，不是宽度
    float y2_1 = data1[3]; // 这是右下角y坐标，不是高度

    // 提取 bbox2 的参数 (x1, y1, x2, y2)
    const float* data2 = bbox2.ptr<float>(0);
    float x1_2 = data2[0];
    float y1_2 = data2[1];
    float x2_2 = data2[2];
    float y2_2 = data2[3];

    // 计算正确的中心点坐标
    float cx1 = (x1_1 + x2_1) / 2.0f;
    float cy1 = (y1_1 + y2_1) / 2.0f;
    float cx2 = (x1_2 + x2_2) / 2.0f;
    float cy2 = (y1_2 + y2_2) / 2.0f;

    cv::Vec2f speed(cx2 - cx1, cy2 - cy1);
    float level = 1.0f - iou_single(bbox1, bbox2);

    float norm = std::sqrt(speed[0] * speed[0] + speed[1] * speed[1]) + 1e-6f;
    speed[0] /= norm;
    speed[1] /= norm;

    return { speed, level };
}



// // 将边界框转换为状态向量
// cv::Mat Utils::convert_bbox_to_z(const cv::Mat& bbox) {
//     // 输入: bbox = [x1, y1, x2, y2]
//     // 输出: z = [cx, cy, a, h]
//     const float* data1 = bbox.ptr<float>(0);
//     float x1 = data1[0];
//     float y1 = data1[1];
//     float w1 = data1[2] - data1[0];
//     float h1 = data1[3] - data1[1];

//     float cx = x1 + w1 / 2.0f;
//     float cy = y1 + h1 / 2.0f;
//     float w = w1;
//     float h = h1;
//     float a = w1 / h1;

//     cv::Mat z(4, 1, CV_32FC1);
//     float* data = z.ptr<float>(0);
//     data[0] = cx;
//     data[1] = cy;
//     data[2] = a;
//     data[3] = h;

//     return z;
// }

// // 从状态向量恢复边界框
// cv::Rect_<float> Utils::convert_z_to_bbox(const cv::Mat& z) {
//     //CV_Assert(z.type() == CV_32FC1 && z.total() == 4);

//     const float* data = z.ptr<float>(0);
//     float cx = data[0];
//     float cy = data[1];
//     float a = data[2];
//     float h = data[3];
//     float w = a * h;

//     cv::Rect_<float> bbox;
//     bbox.x = cx - w / 2.0f;
//     bbox.y = cy - h / 2.0f;
//     bbox.width = w / 2.0f + cx;
//     bbox.height = h / 2.0f + cy;

//     return bbox;
// }

// 计算速度方向


// std::pair<cv::Vec2f, float> Utils::speed_direction(const cv::Mat& bbox1, const cv::Mat& bbox2) {
//     // 提取 bbox1 的参数 (x1, y1, x2, y2)
//     const float* data1 = bbox1.ptr<float>(0);
//     float x1_1 = data1[0];
//     float y1_1 = data1[1];
//     float x2_1 = data1[2]; // 这是右下角x坐标，不是宽度
//     float y2_1 = data1[3]; // 这是右下角y坐标，不是高度

//     // 提取 bbox2 的参数 (x1, y1, x2, y2)
//     const float* data2 = bbox2.ptr<float>(0);
//     float x1_2 = data2[0];
//     float y1_2 = data2[1];
//     float x2_2 = data2[2];
//     float y2_2 = data2[3];

//     // 计算正确的中心点坐标
//     float cx1 = (x1_1 + x2_1) / 2.0f;
//     float cy1 = (y1_1 + y2_1) / 2.0f;
//     float cx2 = (x1_2 + x2_2) / 2.0f;
//     float cy2 = (y1_2 + y2_2) / 2.0f;

//     cv::Vec2f speed(cx2 - cx1, cy2 - cy1);
//     float level = 1.0f - iou_single(bbox1, bbox2);

//     float norm = std::sqrt(speed[0] * speed[0] + speed[1] * speed[1]) + 1e-6f;
//     speed[0] /= norm;
//     speed[1] /= norm;

//     return { speed, level };
// }


// 获取前k帧的观测
cv::Mat Utils::k_previous_obs(const std::map<int, cv::Mat>& observations, int cur_age, int k) {
    // Return a specific observation from observations_
    if (observations.empty()) return cv::Mat::ones(1, 4, CV_32F);

    for (int i = 0; i < k; ++i) {
        int dt = k - i;  // This is \Delta_t for derivative
        auto it = observations.find(cur_age - dt);
        if (it != observations.end()) {
            // std::cout << "we got it, last  " << dt << std::endl;
            return it->second;
        }
    }
    auto max_iter = observations.rbegin();  // Find the largest key in map
    return max_iter->second;
}

// 整数版本的IOU计算
float Utils::GetIOU(const cv::Rect box1, const cv::Rect box2) {
    int area1 = box1.area();
    int area2 = box2.area();
    int inter_area = (box1 & box2).area();
    float iou = static_cast<float>(inter_area) / (area1 + area2 - inter_area + 1e-6f);
    return iou;
}

// 加载模型
uint8_t* Utils::loadModel(const char* path_file, size_t& size_buffer) {
    FILE* fp = fopen(path_file, "rb");
    if (!fp) {
        printf("Don't open %s\n", path_file);
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    size_buffer = ftell(fp);
    rewind(fp);

    uint8_t* buffer = (uint8_t*)malloc(size_buffer);
    if (!buffer) {
        fclose(fp);
        return nullptr;
    }

    fread(buffer, sizeof(uint8_t), size_buffer, fp);
    fclose(fp);

    return buffer;
}

// 将图像扩展为正方形
void Utils::expand2square(const cv::Mat& img, cv::Scalar background_color, cv::Mat& result, int& x_offset, int& y_offset) {
    int width = img.cols;
    int height = img.rows;

    // Calculate the new size and create a new image
    int size = std::max(width, height);
    result = cv::Mat(size, size, img.type(), background_color);

    // Calculate the image paste position
    x_offset = (size - width) / 2;
    y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));
}

//------------------------- 高斯平滑边界框类 -------------------------

std::vector<double> GaussianSmoothBox::generateGaussianKernel(int size, double sigma) {
    if (sigma <= 0) {
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8;
    }

    std::vector<double> kernel(size);
    double sum = 0.0;
    int center = size / 2;
    double twoSigmaSquared = 2.0 * sigma * sigma;
    double sqrtTwoPiSigma = sqrt(2.0 * CV_PI) * sigma;

    for (int i = 0; i < size; ++i) {
        int x = i - center;
        kernel[i] = exp(-(x * x) / twoSigmaSquared) / sqrtTwoPiSigma;
        sum += kernel[i];
    }

    // 归一化
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }

    return kernel;
}

cv::Rect GaussianSmoothBox::smooth(const cv::Rect& currBox) {
    historyBoxes.push_back(currBox);

    // 保持窗口大小
    if (historyBoxes.size() > windowSize) {
        historyBoxes.erase(historyBoxes.begin());
    }

    // 历史数据不足时直接返回当前框
    if (historyBoxes.size() < 3) {  // 至少需要3个点才有平滑意义
        return currBox;
    }

    int currentSize = historyBoxes.size();
    double avgX = 0.0, avgY = 0.0, avgWidth = 0.0, avgHeight = 0.0;

    // 使用对应数量历史框的高斯核权重
    int kernelOffset = gaussianKernel.size() - currentSize;

    for (int i = 0; i < currentSize; ++i) {
        double weight = gaussianKernel[kernelOffset + i];
        const cv::Rect& box = historyBoxes[i];
        avgX += box.x * weight;
        avgY += box.y * weight;
        avgWidth += box.width * weight;
        avgHeight += box.height * weight;
    }

    // 确保宽度和高度为正数
    int width = std::max(1, cvRound(avgWidth));
    int height = std::max(1, cvRound(avgHeight));

    cv::Rect smoothedBox(
        cvRound(avgX),
        cvRound(avgY),
        width,
        height
    );

    return smoothedBox;
}

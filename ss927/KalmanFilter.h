#pragma once

#ifndef KALMANFILTERNEW_H
#define KALMANFILTERNEW_H

#include <opencv2/opencv.hpp>
#include <vector>

struct KalmanAttributes {
    bool IsInitialized = false;
    cv::Mat x;
    cv::Mat P;
    cv::Mat Q;
    cv::Mat B;
    cv::Mat F;
    cv::Mat H;
    cv::Mat R;
    float _alpha_sq;
    cv::Mat M;
    cv::Mat z;
    cv::Mat K;
    cv::Mat y;
    cv::Mat S;
    cv::Mat SI;
    cv::Mat x_prior;
    cv::Mat P_prior;
    cv::Mat x_post;
    cv::Mat P_post;
    std::vector<cv::Mat> history_obs;
};

class KalmanFilterNew {
public:
    /* Constructor */
    KalmanFilterNew();
    KalmanFilterNew(int dim_x_, int dim_z_);

    /* Public methods */
    // 预测/更新采用“计算成功后再提交”的事务语义。返回 false 表示输入状态、
    // 矩阵或计算结果出现 NaN/Inf/维度异常；失败时 x/P 保持调用前的值。
    bool predict();
    bool update(const cv::Mat& z_);
    /* Public variables */
    int dim_x;  // State dimension
    int dim_z;  // Measurement dimension

    cv::Mat x;      // State vector [dim_x, 1]
    cv::Mat P;      // State covariance matrix [dim_x, dim_x]
    cv::Mat Q;      // Process noise covariance [dim_x, dim_x]
    cv::Mat B;      // Control matrix [dim_x, dim_u] (not used)
    cv::Mat F;      // State transition matrix [dim_x, dim_x]
    cv::Mat H;      // Measurement matrix [dim_z, dim_x]
    cv::Mat R;      // Measurement noise covariance [dim_z, dim_z]
    cv::Mat M;      // Process-measurement cross correlation [dim_x, dim_z]
    cv::Mat I;      // Identity matrix [dim_x, dim_x]
    cv::Mat z;      // Measurement vector [dim_z, 1]

    // Computed during innovation step
    cv::Mat K;      // Kalman gain [dim_x, dim_z]
    cv::Mat y;      // Residual [dim_z, 1]
    cv::Mat S;      // System uncertainty [dim_z, dim_z]
    cv::Mat SI;     // Inverse of S [dim_z, dim_z]

    // Saved prior and posterior states
    cv::Mat x_prior;
    cv::Mat P_prior;
    cv::Mat x_post;
    cv::Mat P_post;


    int MAX_HISTORY_SIZE = 20;
    // Additional variables
    float _alpha_sq = 1.0;  // Fading memory control
    bool observed = true;   // Whether the target is being observed

    // History of observations
    std::vector<cv::Mat> history_obs;

private:
    /* Private variables */
    KalmanAttributes attr_saved;  // For saving state when freezing
    std::vector<cv::Mat> new_history;  // Temporary storage for history
};

#endif // KALMANFILTERNEW_H

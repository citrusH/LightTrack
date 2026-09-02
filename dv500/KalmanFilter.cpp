#include "KalmanFilter.h"
#include "utils.h"

#include <cfloat>
#include <cmath>
#include <iostream>

namespace {
bool finite_mat(const cv::Mat& m, int rows = -1, int cols = -1) {
    if (m.empty() || m.type() != CV_32F) return false;
    if (rows >= 0 && m.rows != rows) return false;
    if (cols >= 0 && m.cols != cols) return false;
    return cv::checkRange(m, true, nullptr, -FLT_MAX, FLT_MAX);
}
} // namespace

KalmanFilterNew::KalmanFilterNew() {};

KalmanFilterNew::KalmanFilterNew(int dim_x_, int dim_z_) {
    dim_x = dim_x_;
    dim_z = dim_z_;

    x = cv::Mat::zeros(dim_x_, 1, CV_32F);

    P = cv::Mat::eye(dim_x_, dim_x_, CV_32F);
    Q = cv::Mat::eye(dim_x_, dim_x_, CV_32F);
    B = cv::Mat::eye(dim_x_, dim_x_, CV_32F);
    F = cv::Mat::eye(dim_x_, dim_x_, CV_32F);

    H = cv::Mat::zeros(dim_z_, dim_x_, CV_32F);
    R = cv::Mat::eye(dim_z_, dim_z_, CV_32F);

    M = cv::Mat::zeros(dim_x_, dim_z_, CV_32F);
    I = cv::Mat::eye(dim_x_, dim_x_, CV_32F);
    z = cv::Mat::zeros(dim_z_, 1, CV_32F);

    K = cv::Mat::zeros(dim_x_, dim_z_, CV_32F);
    y = cv::Mat::zeros(dim_z_, 1, CV_32F);
    S = cv::Mat::zeros(dim_z_, dim_z_, CV_32F);

    x_prior = x.clone();
    P_prior = P.clone();
    x_post = x.clone();
    P_post = P.clone();

    _alpha_sq = 1.0f;
    observed = true;
};

bool KalmanFilterNew::predict() {
    if (!finite_mat(x, dim_x, 1)
        || !finite_mat(P, dim_x, dim_x)
        || !finite_mat(F, dim_x, dim_x)
        || !finite_mat(Q, dim_x, dim_x)
        || !std::isfinite(_alpha_sq) || _alpha_sq <= 0.f) {
        return false;
    }

    try {
        cv::Mat next_x = F * x;
        cv::Mat F_P = F * P;
        cv::Mat F_P_Ft;
        cv::gemm(F_P, F, 1.0, cv::Mat(), 0.0, F_P_Ft, cv::GEMM_2_T);
        cv::Mat next_P = _alpha_sq * F_P_Ft + Q;

        if (!finite_mat(next_x, dim_x, 1)
            || !finite_mat(next_P, dim_x, dim_x)) {
            return false;
        }

        x_prior = x.clone();
        P_prior = P.clone();
        x = next_x;
        P = next_P;
        return true;
    } catch (const cv::Exception&) {
        return false;
    }
}

bool KalmanFilterNew::update(const cv::Mat& z_) {
    // 无检测：跳过量测更新，仅依赖 predict() 维持轨迹。
    if (z_.empty()) {
        if (!finite_mat(x, dim_x, 1) || !finite_mat(P, dim_x, dim_x))
            return false;
        x_post = x.clone();
        P_post = P.clone();
        return true;
    }

    if (!finite_mat(x, dim_x, 1)
        || !finite_mat(P, dim_x, dim_x)
        || !finite_mat(H, dim_z, dim_x)
        || !finite_mat(R, dim_z, dim_z)
        || !finite_mat(I, dim_x, dim_x)
        || !finite_mat(z_, dim_z, 1)) {
        return false;
    }

    try {
        cv::Mat next_y = z_ - H * x;

        cv::Mat PHT;
        cv::gemm(P, H, 1.0, cv::Mat(), 0.0, PHT, cv::GEMM_2_T);
        cv::Mat next_S = H * PHT + R;
        if (!finite_mat(next_y, dim_z, 1)
            || !finite_mat(PHT, dim_x, dim_z)
            || !finite_mat(next_S, dim_z, dim_z)) {
            return false;
        }

        // 不显式求逆求增益：创新协方差优先 Cholesky，退化时回退 SVD。
        cv::Mat solved;
        bool solved_ok = cv::solve(next_S, PHT.t(), solved, cv::DECOMP_CHOLESKY);
        if (!solved_ok)
            solved_ok = cv::solve(next_S, PHT.t(), solved, cv::DECOMP_SVD);
        if (!solved_ok || !finite_mat(solved, dim_z, dim_x))
            return false;

        cv::Mat next_K = solved.t();
        cv::Mat next_x = x + next_K * next_y;

        cv::Mat I_KH = I - next_K * H;
        cv::Mat P_tmp;
        cv::gemm(I_KH * P, I_KH, 1.0, cv::Mat(), 0.0, P_tmp, cv::GEMM_2_T);
        cv::Mat KRKt;
        cv::gemm(next_K * R, next_K, 1.0, cv::Mat(), 0.0, KRKt, cv::GEMM_2_T);
        cv::Mat next_P = P_tmp + KRKt;  // Joseph form

        if (!finite_mat(next_K, dim_x, dim_z)
            || !finite_mat(next_x, dim_x, 1)
            || !finite_mat(next_P, dim_x, dim_x)) {
            return false;
        }

        cv::Mat next_SI;
        if (!cv::invert(next_S, next_SI, cv::DECOMP_SVD)
            || !finite_mat(next_SI, dim_z, dim_z)) {
            next_SI = cv::Mat();  // 仅诊断字段；不影响状态更新。
        }

        // 所有关键结果通过有限性检查后再一次性提交。
        y = next_y;
        S = next_S;
        SI = next_SI;
        K = next_K;
        x = next_x;
        P = next_P;
        z = z_.clone();
        x_post = x.clone();
        P_post = P.clone();
        return true;
    } catch (const cv::Exception&) {
        return false;
    }
}

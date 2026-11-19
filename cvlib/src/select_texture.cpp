/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace
{
void post_process_mask(cv::Mat& mask, int kernel_size)
{
    if (kernel_size < 3)
    {
        return;
    }
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

int make_odd_kernel_size(double sigma)
{
    int ks = static_cast<int>(std::ceil(6.0 * sigma));
    if (ks < 3)
    {
        ks = 3;
    }
    if (ks > 59)
    {
        ks = 59;
    }
    if (ks % 2 == 0)
    {
        ++ks;
    }
    return ks;
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    cv::Mat image_gray;
    if (image.channels() > 1)
    {
        cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        image_gray = image;
    }

    const int N_thetas = 8;
    const int N_sigmas = 2;
    const int N_lambdas = 2;

    const double sigmas[N_sigmas] = {2.0, 3.0};
    const double lm_ratios[N_lambdas] = {1.0, 2.0};
    const double gm = 0.5;

    const int N_filters = N_thetas * N_sigmas * N_lambdas;

    std::vector<cv::Mat> kernels_even(N_filters);
    std::vector<cv::Mat> kernels_odd(N_filters);

    int f = 0;
    for (int ti = 0; ti < N_thetas; ++ti)
    {
        const double th = ti * CV_PI / N_thetas;
        for (int si = 0; si < N_sigmas; ++si)
        {
            const double sig = sigmas[si];
            const int ks = make_odd_kernel_size(sig);
            for (int li = 0; li < N_lambdas; ++li)
            {
                const double lm = sig * lm_ratios[li];
                kernels_even[f] = cv::getGaborKernel(cv::Size(ks, ks), sig, th, lm, gm, CV_PI / 2);
                kernels_odd[f] = cv::getGaborKernel(cv::Size(ks, ks), sig, th, lm, gm, 0.0);
                ++f;
            }
        }
    }

    const int half_w = roi.width / 2;
    const int half_h = roi.height / 2;
    const int max_x = std::max(0, image.cols - roi.width);
    const int max_y = std::max(0, image.rows - roi.height);

    std::vector<int> x0_tab(image.cols);
    std::vector<int> y0_tab(image.rows);
    for (int x = 0; x < image.cols; ++x)
    {
        x0_tab[x] = std::max(0, std::min(max_x, x - half_w));
    }
    for (int y = 0; y < image.rows; ++y)
    {
        y0_tab[y] = std::max(0, std::min(max_y, y - half_h));
    }

    const double area = static_cast<double>(roi.width) * static_cast<double>(roi.height);
    cv::Mat dist_sq = cv::Mat::zeros(image.size(), CV_64F);

    cv::Mat resp_e;
    cv::Mat resp_o;
    cv::Mat mag;
    cv::Mat mag_sq;
    cv::Mat integral_mag;
    cv::Mat integral_sq;

    for (f = 0; f < N_filters; ++f)
    {
        cv::filter2D(image_gray, resp_e, CV_32F, kernels_even[f]);
        cv::filter2D(image_gray, resp_o, CV_32F, kernels_odd[f]);
        cv::magnitude(resp_e, resp_o, mag);
        cv::multiply(mag, mag, mag_sq);

        const double sum_mag = cv::sum(mag(roi))[0];
        const double sum_mag_sq = cv::sum(mag_sq(roi))[0];
        const double ref_mean = sum_mag / area;
        double ref_var = sum_mag_sq / area - ref_mean * ref_mean;
        if (ref_var < 0.0)
        {
            ref_var = 0.0;
        }
        const double ref_std = std::sqrt(ref_var);

        cv::integral(mag, integral_mag, CV_64F);
        cv::integral(mag_sq, integral_sq, CV_64F);

        for (int y = 0; y < image.rows; ++y)
        {
            const int y0 = y0_tab[y];
            const int y1 = y0 + roi.height;
            const double* p1u = integral_mag.ptr<double>(y1);
            const double* p1d = integral_mag.ptr<double>(y0);
            const double* p2u = integral_sq.ptr<double>(y1);
            const double* p2d = integral_sq.ptr<double>(y0);
            double* drow = dist_sq.ptr<double>(y);

            for (int x = 0; x < image.cols; ++x)
            {
                const int x0 = x0_tab[x];
                const int x1 = x0 + roi.width;

                const double s = p1u[x1] - p1d[x1] - p1u[x0] + p1d[x0];
                const double s2 = p2u[x1] - p2d[x1] - p2u[x0] + p2d[x0];

                const double mean = s / area;
                double var = s2 / area - mean * mean;
                if (var < 0.0)
                {
                    var = 0.0;
                }
                const double stddev = std::sqrt(var);

                const double dm = mean - ref_mean;
                const double ds = stddev - ref_std;
                drow[x] += dm * dm + ds * ds;
            }
        }
    }

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);
    const double eps_sq = eps * eps;

    for (int y = 0; y < image.rows; ++y)
    {
        const double* drow = dist_sq.ptr<double>(y);
        uchar* rrow = res.ptr<uchar>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            if (drow[x] <= eps_sq)
            {
                rrow[x] = 255;
            }
        }
    }

    const int morph_size = std::max(3, std::min(roi.width, roi.height) / 15);
    post_process_mask(res, morph_size | 1);

    return res;
}
} // namespace cvlib
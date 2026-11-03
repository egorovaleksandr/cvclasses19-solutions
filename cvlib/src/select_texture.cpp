/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <cmath>

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

    double norm_l2() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += v * v;
        }
        return std::sqrt(res);
    }

    double norm_l2_to(const descriptor& right) const
    {
        double res = 0.0;
        for (size_t i = 0; i < size(); ++i)
        {
            const double d = (*this)[i] - right[i];
            res += d * d;
        }
        return std::sqrt(res);
    }
};

void window_stats(const cv::Mat& integral, const cv::Mat& integral_sq, int x0, int y0, int x1, int y1,
                  double& mean, double& stddev)
{
    const double s = integral.at<double>(y1, x1) - integral.at<double>(y0, x1) - integral.at<double>(y1, x0) +
                     integral.at<double>(y0, x0);
    const double s2 = integral_sq.at<double>(y1, x1) - integral_sq.at<double>(y0, x1) -
                      integral_sq.at<double>(y1, x0) + integral_sq.at<double>(y0, x0);
    const double n = static_cast<double>((x1 - x0) * (y1 - y0));

    mean = s / n;

    double var = s2 / n - mean * mean;
    if (var < 0.0)
    {
        var = 0.0;
    }
    stddev = std::sqrt(var);
}

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

    const int N_thetas = 6;
    const int N_sigmas = 3;
    const double sigmas[] = {3.0, 6.0, 10.0};
    const double lm_ratio = 0.8;
    const double gm = 0.5;

    const int N_filters = N_thetas * N_sigmas;
    std::vector<cv::Mat> integrals(N_filters);
    std::vector<cv::Mat> integrals_sq(N_filters);
    descriptor reference(2 * N_filters);

    const int rx0 = roi.x;
    const int ry0 = roi.y;
    const int rx1 = roi.x + roi.width;
    const int ry1 = roi.y + roi.height;

    int k = 0;
    for (int ti = 0; ti < N_thetas; ++ti)
    {
        const double th = ti * CV_PI / N_thetas;
        for (int si = 0; si < N_sigmas; ++si)
        {
            const double sig = sigmas[si];
            const double lm = sig * lm_ratio;

            int ks = static_cast<int>(std::ceil(6.0 * sig));
            if (ks < 3)
            {
                ks = 3;
            }
            if (ks % 2 == 0)
            {
                ++ks;
            }

            const cv::Mat kern_e = cv::getGaborKernel(cv::Size(ks, ks), sig, th, lm, gm, CV_PI / 2);
            const cv::Mat kern_o = cv::getGaborKernel(cv::Size(ks, ks), sig, th, lm, gm, 0.0);

            cv::Mat resp_e;
            cv::Mat resp_o;
            cv::Mat mag;
            cv::filter2D(image_gray, resp_e, CV_32F, kern_e);
            cv::filter2D(image_gray, resp_o, CV_32F, kern_o);
            cv::magnitude(resp_e, resp_o, mag);

            cv::Mat mag_sq;
            cv::multiply(mag, mag, mag_sq);

            cv::integral(mag, integrals[k], CV_64F);
            cv::integral(mag_sq, integrals_sq[k], CV_64F);

            double mean, stddev;
            window_stats(integrals[k], integrals_sq[k], rx0, ry0, rx1, ry1, mean, stddev);
            reference[2 * k] = mean;
            reference[2 * k + 1] = stddev;

            ++k;
        }
    }

    const int half_w = roi.width / 2;
    const int half_h = roi.height / 2;
    const int max_x = std::max(0, image.cols - roi.width);
    const int max_y = std::max(0, image.rows - roi.height);

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);
    descriptor test(2 * N_filters);

    for (int y = 0; y < image.rows; ++y)
    {
        const int y0 = std::max(0, std::min(max_y, y - half_h));
        const int y1 = y0 + roi.height;

        for (int x = 0; x < image.cols; ++x)
        {
            const int x0 = std::max(0, std::min(max_x, x - half_w));
            const int x1 = x0 + roi.width;

            for (int i = 0; i < N_filters; ++i)
            {
                double mean, stddev;
                window_stats(integrals[i], integrals_sq[i], x0, y0, x1, y1, mean, stddev);
                test[2 * i] = mean;
                test[2 * i + 1] = stddev;
            }

            if (test.norm_l2_to(reference) <= eps)
            {
                res.at<uchar>(y, x) = 255;
            }
        }
    }

    const int morph_size = std::max(3, std::min(roi.width, roi.height) / 15);
    post_process_mask(res, morph_size | 1);

    return res;
}
} // namespace cvlib
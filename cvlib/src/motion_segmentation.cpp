/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
constexpr int k_warmup_frames = 30;
constexpr float k_default_gamma = 0.005f;
} // namespace

namespace cvlib
{
motion_segmentation::motion_segmentation() = default;

void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate)
{
    cv::Mat image;
    if (_image.channels() > 1)
    {
        cv::cvtColor(_image, image, cv::COLOR_BGR2GRAY);
    }
    else
    {
        image = _image.getMat();
    }

    cv::Mat fg_zero = cv::Mat::zeros(image.size(), CV_8U);

    if (min_.empty() || min_.size() != image.size())
    {
        image.convertTo(min_, CV_32F);
        image.convertTo(max_, CV_32F);
        image.copyTo(bg_model_);
        fg_zero.copyTo(_fgmask);
        warmup_frames_ = 0;
        return;
    }

    cv::Mat image_f;
    image.convertTo(image_f, CV_32F);

    if (warmup_frames_ < k_warmup_frames)
    {
        cv::Mat tmp_min;
        cv::Mat tmp_max;
        cv::min(min_, image_f, tmp_min);
        cv::max(max_, image_f, tmp_max);
        tmp_min.copyTo(min_);
        tmp_max.copyTo(max_);

        cv::Mat mid = (min_ + max_) * 0.5;
        mid.convertTo(bg_model_, CV_8U);

        fg_zero.copyTo(_fgmask);
        ++warmup_frames_;
        return;
    }

    const float th = static_cast<float>(var_threshold_);

    cv::Mat diff_lo = min_ - image_f - th;
    cv::Mat diff_hi = image_f - max_ - th;
    cv::Mat fg_bool = (diff_lo > 0) | (diff_hi > 0);
    fg_bool.copyTo(_fgmask);

    cv::Mat bg_mask = (fg_bool == 0);

    const float gamma = (learningRate >= 0) ? static_cast<float>(learningRate) : k_default_gamma;

    if (gamma > 0)
    {
        cv::Mat next_min;
        cv::Mat next_max;
        cv::min(min_, image_f, next_min);
        cv::max(max_, image_f, next_max);

        cv::Mat min_drift = min_ + gamma * (image_f - min_);
        cv::Mat max_drift = max_ + gamma * (image_f - max_);

        cv::Mat not_new_low = (image_f >= min_);
        cv::Mat not_new_high = (image_f <= max_);

        min_drift.copyTo(next_min, not_new_low);
        max_drift.copyTo(next_max, not_new_high);

        next_min.copyTo(min_, bg_mask);
        next_max.copyTo(max_, bg_mask);
    }

    cv::Mat mid = (min_ + max_) * 0.5;
    mid.convertTo(bg_model_, CV_8U);
}
} // namespace cvlib
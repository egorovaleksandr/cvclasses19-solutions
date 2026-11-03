/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <random>
#include <vector>

namespace
{
constexpr int k_circle_radius = 3;
constexpr int k_circle_size = 16;
constexpr int k_desc_bits = 256;
constexpr int k_desc_bytes = k_desc_bits / 8;
constexpr int k_patch_half = 15;
constexpr unsigned k_pattern_seed = 42;

const int k_dx[k_circle_size] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
const int k_dy[k_circle_size] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};

inline bool has_run_of_9(uint16_t mask)
{
    const uint32_t m = static_cast<uint32_t>(mask) | (static_cast<uint32_t>(mask) << 16);
    const uint32_t r = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4) & (m >> 5) & (m >> 6) & (m >> 7) & (m >> 8);
    return (r & 0xFFFFu) != 0;
}

const std::vector<cv::Point2i>& brief_pattern()
{
    static const std::vector<cv::Point2i> pattern = [] {
        std::mt19937 rng(k_pattern_seed);
        std::uniform_int_distribution<int> dist(-k_patch_half, k_patch_half);
        std::vector<cv::Point2i> pts;
        pts.reserve(2 * k_desc_bits);
        for (int i = 0; i < 2 * k_desc_bits; ++i)
        {
            pts.emplace_back(dist(rng), dist(rng));
        }
        return pts;
    }();
    return pattern;
}

cv::Mat to_gray(const cv::Mat& image)
{
    if (image.channels() > 1)
    {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    return image;
}
} // namespace

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray _image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    keypoints.clear();

    const cv::Mat image = to_gray(_image.getMat());
    if (image.rows <= 2 * k_circle_radius || image.cols <= 2 * k_circle_radius)
    {
        return;
    }

    cv::Mat mask_mat;
    if (!mask.empty())
    {
        mask_mat = mask.getMat();
    }

    const int t = threshold_;

    for (int y = k_circle_radius; y < image.rows - k_circle_radius; ++y)
    {
        const uchar* const up = image.ptr<uchar>(y - 3);
        const uchar* const row = image.ptr<uchar>(y);
        const uchar* const dn = image.ptr<uchar>(y + 3);
        const uchar* const mrow = mask_mat.empty() ? nullptr : mask_mat.ptr<uchar>(y);

        for (int x = k_circle_radius; x < image.cols - k_circle_radius; ++x)
        {
            if (mrow && mrow[x] == 0)
            {
                continue;
            }

            const int p = row[x];
            const int hi = p + t;
            const int lo = p - t;

            const int nb = (up[x] > hi) + (row[x + 3] > hi) + (dn[x] > hi) + (row[x - 3] > hi);
            const int nd = (up[x] < lo) + (row[x + 3] < lo) + (dn[x] < lo) + (row[x - 3] < lo);
            if (nb < 3 && nd < 3)
            {
                continue;
            }

            uint16_t bright_mask = 0;
            uint16_t dark_mask = 0;
            for (int i = 0; i < k_circle_size; ++i)
            {
                const int v = image.at<uchar>(y + k_dy[i], x + k_dx[i]);
                bright_mask |= static_cast<uint16_t>((v > hi) << i);
                dark_mask |= static_cast<uint16_t>((v < lo) << i);
            }

            if (has_run_of_9(bright_mask) || has_run_of_9(dark_mask))
            {
                keypoints.emplace_back(static_cast<float>(x), static_cast<float>(y), 7.0f);
            }
        }
    }
}

void corner_detector_fast::compute(cv::InputArray _image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    if (keypoints.empty())
    {
        descriptors.release();
        return;
    }

    const cv::Mat image = to_gray(_image.getMat());
    const auto& pattern = brief_pattern();

    descriptors.create(static_cast<int>(keypoints.size()), k_desc_bytes, CV_8U);
    cv::Mat desc = descriptors.getMat();
    desc.setTo(0);

    for (size_t k = 0; k < keypoints.size(); ++k)
    {
        const int cx = cvRound(keypoints[k].pt.x);
        const int cy = cvRound(keypoints[k].pt.y);
        uchar* row = desc.ptr<uchar>(static_cast<int>(k));

        for (int i = 0; i < k_desc_bits; ++i)
        {
            const int x1 = std::max(0, std::min(image.cols - 1, cx + pattern[2 * i].x));
            const int y1 = std::max(0, std::min(image.rows - 1, cy + pattern[2 * i].y));
            const int x2 = std::max(0, std::min(image.cols - 1, cx + pattern[2 * i + 1].x));
            const int y2 = std::max(0, std::min(image.rows - 1, cy + pattern[2 * i + 1].y));

            if (image.at<uchar>(y1, x1) < image.at<uchar>(y2, x2))
            {
                row[i / 8] |= static_cast<uchar>(1u << (i % 8));
            }
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                            bool useProvidedKeypoints)
{
    if (!useProvidedKeypoints)
    {
        detect(image, keypoints, mask);
    }

    compute(image, keypoints, descriptors);
}
} // namespace cvlib
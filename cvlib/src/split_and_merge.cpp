/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace
{
double get_pixel(const cv::Mat& image, int y, int x)
{
    switch (image.depth())
    {
    case CV_8U:
        return image.at<uchar>(y, x);
    case CV_8S:
        return image.at<schar>(y, x);
    case CV_16U:
        return image.at<ushort>(y, x);
    case CV_16S:
        return image.at<short>(y, x);
    case CV_32S:
        return image.at<int>(y, x);
    case CV_32F:
        return image.at<float>(y, x);
    case CV_64F:
        return image.at<double>(y, x);
    }

    return 0.0;
}

void set_pixel(cv::Mat& image, int y, int x, double value)
{
    switch (image.depth())
    {
    case CV_8U:
        image.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
        break;
    case CV_8S:
        image.at<schar>(y, x) = cv::saturate_cast<schar>(value);
        break;
    case CV_16U:
        image.at<ushort>(y, x) = cv::saturate_cast<ushort>(value);
        break;
    case CV_16S:
        image.at<short>(y, x) = cv::saturate_cast<short>(value);
        break;
    case CV_32S:
        image.at<int>(y, x) = cv::saturate_cast<int>(value);
        break;
    case CV_32F:
        image.at<float>(y, x) = static_cast<float>(value);
        break;
    case CV_64F:
        image.at<double>(y, x) = value;
        break;
    }
}

class DSU
{
    public:
    explicit DSU(const cv::Mat& image)
        : parent_(image.rows * image.cols),
          rank_(image.rows * image.cols, 0),
          sum_(image.rows * image.cols, 0.0),
          cnt_(image.rows * image.cols, 1)
    {
        std::iota(parent_.begin(), parent_.end(), 0);

        for (int y = 0; y < image.rows; ++y)
        {
            for (int x = 0; x < image.cols; ++x)
            {
                sum_[y * image.cols + x] = get_pixel(image, y, x);
            }
        }
    }

    int find(int x)
    {
        while (parent_[x] != x)
        {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }

        return x;
    }

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);

        if (a == b)
        {
            return;
        }

        if (rank_[a] < rank_[b])
        {
            std::swap(a, b);
        }

        parent_[b] = a;
        sum_[a] += sum_[b];
        cnt_[a] += cnt_[b];

        if (rank_[a] == rank_[b])
        {
            ++rank_[a];
        }
    }

    double mean(int x)
    {
        const int r = find(x);
        return sum_[r] / cnt_[r];
    }

    private:
    std::vector<int> parent_;
    std::vector<int> rank_;
    std::vector<double> sum_;
    std::vector<int> cnt_;
};

void split_image(cv::Mat image, double stddev)
{
    if (image.empty())
    {
        return;
    }

    cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), stddev);
    split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), stddev);
    split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), stddev);
    split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), stddev);
}

void merge_image(cv::Mat& image, double stddev)
{
    const int width = image.cols;
    const int height = image.rows;

    DSU dsu(image);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const double v = get_pixel(image, y, x);
            const int id = y * width + x;

            if (x + 1 < width && std::abs(v - get_pixel(image, y, x + 1)) < stddev)
            {
                dsu.unite(id, id + 1);
            }

            if (y + 1 < height && std::abs(v - get_pixel(image, y + 1, x)) < stddev)
            {
                dsu.unite(id, id + width);
            }
        }
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            set_pixel(image, y, x, dsu.mean(y * width + x));
        }
    }
}
} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    cv::Mat res = image.clone();

    split_image(res, stddev);

    merge_image(res, stddev);

    return res;
}
} // namespace cvlib
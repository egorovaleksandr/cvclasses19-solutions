/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include "utils.hpp"

namespace
{
int hamming_distance(const uchar* a, const uchar* b, int bytes)
{
    int dist = 0;
    for (int i = 0; i < bytes; ++i)
    {
        uchar x = a[i] ^ b[i];
        while (x)
        {
            dist += x & 1;
            x >>= 1;
        }
    }
    return dist;
}

cv::Mat compute_hamming_histogram(const cv::Mat& descriptors, int bins = 50)
{
    cv::Mat hist = cv::Mat::zeros(bins, 1, CV_32F);

    const int n = descriptors.rows;
    const int bytes = descriptors.cols;
    if (n < 2 || bytes < 1)
    {
        return hist;
    }

    std::vector<float> distances;
    distances.reserve(static_cast<size_t>(n) * (n - 1) / 2);

    for (int i = 0; i < n; ++i)
    {
        const uchar* a = descriptors.ptr<uchar>(i);
        for (int j = i + 1; j < n; ++j)
        {
            const uchar* b = descriptors.ptr<uchar>(j);
            const int h = hamming_distance(a, b, bytes);
            distances.push_back(static_cast<float>(h) / (8.0f * bytes));
        }
    }

    const float range[] = {0.0f, 1.0f};
    const float* hist_range = {range};
    const cv::Mat data(distances);
    cv::calcHist(&data, 1, 0, cv::Mat(), hist, 1, &bins, &hist_range);
    return hist;
}

cv::Mat plot_histograms(const cv::Mat& hist_a, const std::string& label_a,
                        const cv::Mat& hist_b, const std::string& label_b,
                        int w = 640, int h = 400)
{
    cv::Mat plot = cv::Mat::zeros(h, w, CV_8UC3);

    const int baseline_pad = 40;
    const int left_pad = 40;
    const int plot_h = h - baseline_pad;
    const int plot_w = w - left_pad;

    double max_val = 1.0;
    if (!hist_a.empty()) max_val = std::max(max_val, cv::norm(hist_a, cv::NORM_INF));
    if (!hist_b.empty()) max_val = std::max(max_val, cv::norm(hist_b, cv::NORM_INF));

    const int bar_w = plot_w / hist_a.rows;
    for (int i = 0; i < hist_a.rows; ++i)
    {
        const int x0 = left_pad + i * bar_w;
        const int x1 = x0 + bar_w - 1;

        const int ha = static_cast<int>(hist_a.at<float>(i) / max_val * (plot_h - 20));
        const int hb = static_cast<int>(hist_b.at<float>(i) / max_val * (plot_h - 20));

        cv::rectangle(plot, cv::Point(x0, plot_h - ha), cv::Point(x1, plot_h),
                      cv::Scalar(0, 0, 255), cv::FILLED);
        cv::rectangle(plot, cv::Point(x0, plot_h - hb), cv::Point(x1, plot_h),
                      cv::Scalar(255, 0, 0), cv::FILLED);
    }

    cv::line(plot, cv::Point(left_pad, plot_h), cv::Point(w, plot_h), cv::Scalar(128, 128, 128), 1);
    cv::line(plot, cv::Point(left_pad, 0), cv::Point(left_pad, plot_h), cv::Scalar(128, 128, 128), 1);
    cv::putText(plot, "0", cv::Point(left_pad - 10, plot_h + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    cv::putText(plot, "1", cv::Point(w - 20, plot_h + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));

    cv::rectangle(plot, cv::Point(w - 200, 20), cv::Point(w - 180, 30), cv::Scalar(0, 0, 255), cv::FILLED);
    cv::putText(plot, label_a, cv::Point(w - 170, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    cv::rectangle(plot, cv::Point(w - 200, 45), cv::Point(w - 180, 55), cv::Scalar(255, 0, 0), cv::FILLED);
    cv::putText(plot, label_b, cv::Point(w - 170, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

    return plot;
}
} // namespace

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";
    const auto hist_wnd = "histogram";

    const std::string key_a = "ORB";
    const std::string key_b = "FAST_Binary";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::namedWindow(hist_wnd);

    cv::Mat frame;
    auto detector_a = cv::ORB::create();
    auto detector_b = cvlib::corner_detector_fast::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors;

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::imshow(main_wnd, frame);

        detector_b->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));

        utils::put_fps_text(frame, fps);

        const std::string text = "Corners: " + std::to_string(corners.size());
        cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        cv::imshow(demo_wnd, frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            cv::FileStorage file("descriptor.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

            cv::Mat desc_a;
            detector_a->compute(frame, corners, desc_a);
            file << key_a << desc_a;

            cv::Mat desc_b;
            detector_b->compute(frame, corners, desc_b);
            file << key_b << desc_b;

            std::cout << "Dump descriptors complete! \n";

            if (!desc_a.empty() && !desc_b.empty())
            {
                const cv::Mat hist_a = compute_hamming_histogram(desc_a);
                const cv::Mat hist_b = compute_hamming_histogram(desc_b);
                cv::imshow(hist_wnd, plot_histograms(hist_a, key_a, hist_b, key_b));
            }
        }

        std::cout << "Feature points: " << corners.size() << "\r";
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    cv::destroyWindow(hist_wnd);

    return 0;
}
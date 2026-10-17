/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <opencv2/opencv.hpp>

#include <cvlib.hpp>

int demo_split_and_merge(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cv::Mat frame;
    cv::Mat frame_gray;
    cv::Mat frame_blur;

    const auto origin_wnd = "origin";
    const auto demo_wnd = "demo";

    int stddev = 10;
    int blur = 1;
    cv::namedWindow(demo_wnd, 1);
    cv::createTrackbar("stdev", demo_wnd, &stddev, 127);
    cv::createTrackbar("blur", demo_wnd, &blur, 50);

    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

        if (blur > 0)
        {
            const int ksize = 2 * blur + 1;
            cv::GaussianBlur(frame_gray, frame_blur, cv::Size(ksize, ksize), 0);
        }
        else
        {
            frame_blur = frame_gray;
        }

        cv::imshow(origin_wnd, frame);
        cv::imshow(demo_wnd, cvlib::split_and_merge(frame_blur, stddev));
    }

    cv::destroyWindow(origin_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
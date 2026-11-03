/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

namespace
{
struct trackbar_data
{
    cvlib::motion_segmentation* mseg;
};

void on_threshold(int value, void* userdata)
{
    auto* data = static_cast<trackbar_data*>(userdata);
    data->mseg->setVarThreshold(value);
}
} // namespace

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cvlib::motion_segmentation mseg;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    int threshold = 20;
    mseg.setVarThreshold(threshold);

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    trackbar_data tb_data{&mseg};
    cv::createTrackbar("th", demo_wnd, &threshold, 255, on_threshold, &tb_data);

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::imshow(main_wnd, frame);

        mseg.apply(frame, frame_mseg);
        if (!frame_mseg.empty())
        {
            cv::imshow(demo_wnd, frame_mseg);
        }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
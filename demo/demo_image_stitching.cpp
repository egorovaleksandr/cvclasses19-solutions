/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    const auto main_wnd = "orig";
    const auto pano_wnd = "pano";

    cv::namedWindow(main_wnd);
    cv::namedWindow(pano_wnd);

    cv::Mat frame;
    cv::Mat pano;
    cvlib::Stitcher stitcher;

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;
        if (frame.empty()) break;

        // Показываем оригинал, но FPS рисуем только на копии для отображения
        cv::Mat display_frame = frame.clone();
        utils::put_fps_text(display_frame, fps);
        cv::imshow(main_wnd, display_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            pano = stitcher.stitch(frame, pano); // Отправляем чистый frame
            if (!pano.empty())
            {
                cv::imshow(pano_wnd, pano);
            }
            std::cout << "Pano updated! Size: " << pano.size() << "\n";
        }
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(pano_wnd);

    return 0;
}
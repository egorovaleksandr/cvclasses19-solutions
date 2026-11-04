/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_feature_matching(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    auto detector = cvlib::corner_detector_fast::create();
    auto matcher = cvlib::descriptor_matcher(1.2f);

    // Трекбары для настройки
    int ratio_x10 = 12; // Соответствует 1.2f
    int maxDistance = 50; // Порог SSD (Hamming distance для бинарных)
    cv::createTrackbar("ratio x10", demo_wnd, &ratio_x10, 30);
    cv::createTrackbar("maxDist", demo_wnd, &maxDistance, 256);

    /// \brief helper struct for tidy code
    struct img_features
    {
        cv::Mat img;
        std::vector<cv::KeyPoint> corners;
        cv::Mat descriptors;
    };

    img_features ref;
    img_features test;
    std::vector<std::vector<cv::DMatch>> pairs;
    std::vector<std::vector<cv::DMatch>> filtered_pairs;

    cv::Mat main_frame;
    cv::Mat demo_frame;
    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> test.img;
        if (test.img.empty())
            break;

        detector->detect(test.img, test.corners);
        cv::drawKeypoints(test.img, test.corners, main_frame);
        cv::imshow(main_wnd, main_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            ref.img = test.img.clone();
            detector->detectAndCompute(ref.img, cv::Mat(), ref.corners, ref.descriptors);
        }

        if (ref.corners.empty())
        {
            continue;
        }

        detector->compute(test.img, test.corners, test.descriptors);
        
        // Настраиваем ratio из трекбара (значение 12 -> 1.2)
        matcher.set_ratio(ratio_x10 / 10.0f);
        
        // Ищем 2 лучших совпадения для каждой точки (для ratio test)
        matcher.knnMatch(test.descriptors, ref.descriptors, pairs, 2);

        // Фильтруем по maxDistance
        filtered_pairs.clear();
        for (size_t i = 0; i < pairs.size(); ++i)
        {
            if (!pairs[i].empty() && pairs[i][0].distance <= maxDistance)
            {
                filtered_pairs.push_back(pairs[i]);
            }
        }

        cv::drawMatches(test.img, test.corners, ref.img, ref.corners, filtered_pairs, demo_frame);

        utils::put_fps_text(demo_frame, fps);
        cv::imshow(demo_wnd, demo_frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
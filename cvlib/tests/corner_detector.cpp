/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("simple check", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();

    SECTION("empty image")
    {
        cv::Mat image = cv::Mat::zeros(10, 10, CV_8UC1);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("flat white image")
    {
        cv::Mat image(100, 100, CV_8UC1, cv::Scalar(255));
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("flat black image")
    {
        cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC1);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("smooth gradient has no corners")
    {
        cv::Mat image(100, 100, CV_8UC1);
        for (int y = 0; y < 100; ++y)
        {
            for (int x = 0; x < 100; ++x)
            {
                image.at<uchar>(y, x) = static_cast<uchar>(y * 255 / 99);
            }
        }
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("bright square on dark")
    {
        cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::rectangle(image, cv::Rect(40, 40, 20, 20), cv::Scalar(255), -1);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.size() >= 4);
    }

    SECTION("dark square on bright")
    {
        cv::Mat image(100, 100, CV_8UC1, cv::Scalar(255));
        cv::rectangle(image, cv::Rect(40, 40, 20, 20), cv::Scalar(0), -1);
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.size() >= 4);
    }

    SECTION("L-shaped corner")
    {
        cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC1);
        image(cv::Rect(0, 0, 30, 30)) = 255;
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(!out.empty());
    }

    SECTION("mask restricts detection")
    {
        cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::rectangle(image, cv::Rect(20, 20, 60, 60), cv::Scalar(255), -1);

        std::vector<cv::KeyPoint> no_mask_out;
        fast->detect(image, no_mask_out);

        cv::Mat mask = cv::Mat::zeros(100, 100, CV_8UC1);
        mask(cv::Rect(0, 0, 50, 50)) = 255;

        std::vector<cv::KeyPoint> mask_out;
        fast->detect(image, mask_out, mask);

        REQUIRE(!no_mask_out.empty());
        REQUIRE(mask_out.size() < no_mask_out.size());
    }

    SECTION("descriptor shape")
    {
        cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::rectangle(image, cv::Rect(40, 40, 20, 20), cv::Scalar(255), -1);

        std::vector<cv::KeyPoint> kp;
        cv::Mat desc;
        fast->detectAndCompute(image, cv::noArray(), kp, desc);

        REQUIRE(!kp.empty());
        REQUIRE(desc.rows == static_cast<int>(kp.size()));
        REQUIRE(desc.cols == 32);
        REQUIRE(desc.type() == CV_8U);
    }

    SECTION("descriptor is deterministic")
    {
        cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::rectangle(image, cv::Rect(40, 40, 20, 20), cv::Scalar(255), -1);

        std::vector<cv::KeyPoint> kp;
        cv::Mat desc1;
        cv::Mat desc2;
        fast->detectAndCompute(image, cv::noArray(), kp, desc1);
        fast->detectAndCompute(image, cv::noArray(), kp, desc2);

        REQUIRE(cv::norm(desc1, desc2, cv::NORM_INF) == 0);
    }
}
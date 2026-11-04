/* Image stitching algorithm implementation.
 * @file
 * @date 2018-12-10
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cvlib
{
Stitcher::Stitcher() : matcher_(0.85f)
{
    detector_ = corner_detector_fast::create();
}

cv::Mat Stitcher::stitch(const cv::Mat& image, const cv::Mat& pano)
{
    if (pano.empty()) return image.clone();

    std::vector<cv::KeyPoint> kp_image, kp_pano;
    cv::Mat desc_image, desc_pano;

    detector_->detectAndCompute(image, cv::noArray(), kp_image, desc_image);
    detector_->detectAndCompute(pano, cv::noArray(), kp_pano, desc_pano);

    if (desc_image.empty() || desc_pano.empty()) return pano;

    std::vector<std::vector<cv::DMatch>> matches;
    matcher_.knnMatch(desc_image, desc_pano, matches, 2);

    std::vector<cv::Point2f> pts_image, pts_pano;
    for (const auto& match : matches)
    {
        if (match.size() == 1) 
        {
            pts_image.push_back(kp_image[match[0].queryIdx].pt);
            pts_pano.push_back(kp_pano[match[0].trainIdx].pt);
        }
    }

    if (pts_image.size() < 20) return pano; // Увеличили порог для надежности

    cv::Mat inliers;
    cv::Mat H = cv::findHomography(pts_image, pts_pano, cv::RANSAC, 3.0, inliers);
    
    if (H.empty() || cv::countNonZero(inliers) < 15) 
    {
        std::cout << "[Stitch] Not enough inliers or empty H." << std::endl;
        return pano;
    }

    // --- SANITY CHECK (Проверка адекватности матрицы) ---
    std::vector<cv::Point2f> corners = {
        {0.0f, 0.0f}, 
        {static_cast<float>(image.cols), 0.0f}, 
        {static_cast<float>(image.cols), static_cast<float>(image.rows)}, 
        {0.0f, static_cast<float>(image.rows)}
    };
    std::vector<cv::Point2f> transformed_corners;
    cv::perspectiveTransform(corners, transformed_corners, H);

    // Считаем площадь искаженного кадра
    double new_area = cv::contourArea(transformed_corners);
    double orig_area = image.cols * image.rows;

    // Если площадь изменилась более чем в 2 раза (увеличилась или уменьшилась) - это ложная матрица
    if (new_area < orig_area * 0.5 || new_area > orig_area * 2.0) 
    {
        std::cout << "[Stitch] Bad homography rejected (area check)." << std::endl;
        return pano;
    }
    // --------------------------------------------------

    // Вычисляем границы
    float min_x = 0, min_y = 0, max_x = static_cast<float>(pano.cols), max_y = static_cast<float>(pano.rows);
    for (const auto& pt : transformed_corners) {
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
    }

    cv::Mat T = (cv::Mat_<double>(3,3) << 1, 0, -min_x, 0, 1, -min_y, 0, 0, 1);
    cv::Mat shifted_H = T * H;

    int width = static_cast<int>(std::round(max_x - min_x));
    int height = static_cast<int>(std::round(max_y - min_y));
    
    // Защита от создания гигантского холста (больше 5000 пикселей)
    if (width > 5000 || height > 5000) 
    {
        std::cout << "[Stitch] Canvas too large." << std::endl;
        return pano;
    }

    cv::Size canvas_size(width, height);

    cv::Mat canvas = cv::Mat::zeros(canvas_size, CV_8UC3);
    cv::Mat pano_mask = cv::Mat::zeros(canvas_size, CV_8UC1);
    cv::Mat warp_mask = cv::Mat::zeros(canvas_size, CV_8UC1);

    int pano_x = static_cast<int>(std::round(-min_x));
    int pano_y = static_cast<int>(std::round(-min_y));

    cv::Rect pano_roi(pano_x, pano_y, pano.cols, pano.rows);
    pano_roi &= cv::Rect(0, 0, canvas.cols, canvas.rows);
    
    if (pano_roi.area() > 0)
    {
        pano(cv::Rect(pano_roi.x - pano_x, pano_roi.y - pano_y, pano_roi.width, pano_roi.height))
            .copyTo(canvas(pano_roi));
        pano_mask(pano_roi).setTo(255);
    }

    cv::Mat warped_img;
    cv::warpPerspective(image, warped_img, shifted_H, canvas_size);
    cv::warpPerspective(cv::Mat(image.size(), CV_8UC1, cv::Scalar(255)), warp_mask, shifted_H, canvas_size);

    // Distance Transform Blending
    cv::Mat dist_pano, dist_warp;
    cv::distanceTransform(pano_mask, dist_pano, cv::DIST_L2, 3);
    cv::distanceTransform(warp_mask, dist_warp, cv::DIST_L2, 3);

    cv::Mat weight_pano, weight_warp, weight_sum;
    cv::add(dist_pano, dist_warp, weight_sum);
    cv::max(weight_sum, 1e-5f, weight_sum);

    cv::divide(dist_pano, weight_sum, weight_pano);
    cv::divide(dist_warp, weight_sum, weight_warp);

    cv::Mat canvas_f, warped_f;
    canvas.convertTo(canvas_f, CV_32FC3);
    warped_img.convertTo(warped_f, CV_32FC3);

    cv::Mat weight_pano_3, weight_warp_3;
    cv::cvtColor(weight_pano, weight_pano_3, cv::COLOR_GRAY2BGR);
    cv::cvtColor(weight_warp, weight_warp_3, cv::COLOR_GRAY2BGR);

    cv::Mat result_f = canvas_f.mul(weight_pano_3) + warped_f.mul(weight_warp_3);
    cv::Mat result;
    result_f.convertTo(result, CV_8UC3);

    // Обрезка
    cv::Mat gray;
    cv::cvtColor(result, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point> non_zero;
    cv::findNonZero(gray > 0, non_zero);

    if (non_zero.empty()) return pano;

    return result(cv::boundingRect(non_zero)).clone();
}
} // namespace cvlib
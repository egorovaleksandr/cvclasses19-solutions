/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <vector>

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
} // namespace

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.clear();
    matches.resize(q_desc.rows);

    const int bytes = q_desc.cols;
    const int actual_k = std::min(k, t_desc.rows);

    for (int i = 0; i < q_desc.rows; ++i)
    {
        const uchar* q = q_desc.ptr<uchar>(i);
        std::vector<cv::DMatch> current_matches;
        current_matches.reserve(t_desc.rows);

        for (int j = 0; j < t_desc.rows; ++j)
        {
            const uchar* t = t_desc.ptr<uchar>(j);
            int dist = hamming_distance(q, t, bytes);
            current_matches.emplace_back(i, j, static_cast<float>(dist));
        }

        if (current_matches.empty()) continue;

        std::partial_sort(current_matches.begin(), current_matches.begin() + actual_k, current_matches.end());

        if (actual_k >= 2 && current_matches.size() >= 2)
        {
            // Ratio of SSD (или Hamming distance) check
            if (current_matches[0].distance < ratio_ * current_matches[1].distance)
            {
                matches[i].push_back(current_matches[0]);
            }
        }
        else
        {
            for (int m = 0; m < actual_k; ++m)
            {
                matches[i].push_back(current_matches[m]);
            }
        }
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.clear();
    matches.resize(q_desc.rows);

    const int bytes = q_desc.cols;

    for (int i = 0; i < q_desc.rows; ++i)
    {
        const uchar* q = q_desc.ptr<uchar>(i);
        for (int j = 0; j < t_desc.rows; ++j)
        {
            const uchar* t = t_desc.ptr<uchar>(j);
            int dist = hamming_distance(q, t, bytes);
            if (dist <= maxDistance)
            {
                matches[i].emplace_back(i, j, static_cast<float>(dist));
            }
        }
    }
}
} // namespace cvlib
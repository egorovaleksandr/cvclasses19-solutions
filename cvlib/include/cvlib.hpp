/* Computer Vision Functions.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <opencv2/opencv.hpp>

namespace cvlib
{
/// \brief Split and merge algorithm for image segmentation
cv::Mat split_and_merge(const cv::Mat& image, double stddev);

/// \brief Segment texuture on passed image according to sample in ROI
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps);

/// \brief Motion Segmentation algorithm
class motion_segmentation : public cv::BackgroundSubtractor
{
    public:
    motion_segmentation();
    void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = -1) override;
    void getBackgroundImage(cv::OutputArray backgroundImage) const override
    {
        backgroundImage.assign(bg_model_);
    }
    private:
    cv::Mat bg_model_;
};

/// \brief FAST corner detection algorithm
class corner_detector_fast : public cv::Feature2D
{
    public:
    static cv::Ptr<corner_detector_fast> create();
    virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;
    virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;
    virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                  bool useProvidedKeypoints = false) override;
    virtual cv::String getDefaultName() const override
    {
        return "FAST_Binary";
    }

    void setThreshold(int threshold) { threshold_ = threshold; }
    int getThreshold() const { return threshold_; }

    private:
    int threshold_ = 20;
};

/// \brief Descriptor matched based on ratio of SSD
class descriptor_matcher : public cv::DescriptorMatcher
{
    public:
    descriptor_matcher(float ratio = 1.5) : ratio_(ratio) {}
    void set_ratio(float r) { ratio_ = r; }

    protected:
    virtual void knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k,
                              cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;
    virtual void radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                 cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;
    virtual bool isMaskSupported() const override { return false; }
    virtual cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override
    {
        cv::Ptr<cv::DescriptorMatcher> copy = new descriptor_matcher(*this);
        if (emptyTrainData) copy->clear();
        return copy;
    }
    private:
    float ratio_;
};

/// \brief Stitcher for merging images into big one
class Stitcher
{
    public:
    Stitcher();
    cv::Mat stitch(const cv::Mat& image, const cv::Mat& pano);

    private:
    cv::Ptr<corner_detector_fast> detector_;
    descriptor_matcher matcher_;
};
} // namespace cvlib

#endif // __CVLIB_HPP__
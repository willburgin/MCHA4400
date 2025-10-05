#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "image_features.h"

PointFeature::PointFeature()
    : score(0)
    , x(0)
    , y(0)
{}

PointFeature::PointFeature(const double & score_, const double & x_, const double & y_)
    : score(score_)
    , x(x_)
    , y(y_)
{}

bool PointFeature::operator<(const PointFeature & other) const
{
    return (score > other.score);
}

// std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures)
// {
//     std::vector<PointFeature> features;
    
//     // Convert to grayscale
//     cv::Mat gray;
//     cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
//     // FAST corner detector with non-max suppression disabled
//     cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(
//         35,     // threshold - lower value for more features
//         false,  // nonmaxSuppression disabled
//         cv::FastFeatureDetector::TYPE_9_16
//     );
    
//     // Detect using OpenCV's KeyPoint
//     std::vector<cv::KeyPoint> keypoints;
//     detector->detect(gray, keypoints);
    
//     // Convert cv::KeyPoint to PointFeature
//     for (const auto& kp : keypoints) {
//         features.emplace_back(kp.response, kp.pt.x, kp.pt.y);
//     }
    
//     // Sort by score (corner strength) - strongest first
//     std::sort(features.begin(), features.end());  // Uses your operator
    
//     // Cap to maxNumFeatures
//     if (features.size() > static_cast<size_t>(maxNumFeatures)) {
//         features.resize(maxNumFeatures);
//     }
    
//     return features;
// }

std::vector<PointFeature> detectFeatures(const cv::Mat & img, const int & maxNumFeatures)
{
    std::vector<PointFeature> features;
    
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // Compute minimum eigenvalues (Shi-Tomasi corner response)
    cv::Mat dst;
    cv::cornerMinEigenVal(gray, dst, 4, 3); // blockSize=5, apertureSize=3
    
    float thresh = 0.0008f; // Threshold for corner detection
    
    // Extract corners above threshold
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            float eigenVal = dst.at<float>(y, x);
            if (eigenVal > thresh) {
                features.emplace_back(eigenVal, static_cast<double>(x), static_cast<double>(y));
            }
        }
    }
    
    // Sort by score (eigenvalue) - strongest first
    std::sort(features.begin(), features.end()); // Uses your operator
    
    // Cap to maxNumFeatures
    if (features.size() > static_cast<size_t>(maxNumFeatures)) {
        features.resize(maxNumFeatures);
    }
    
    return features;
}

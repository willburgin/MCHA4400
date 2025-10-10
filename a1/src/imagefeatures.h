#ifndef IMAGEFEATURES_H
#define IMAGEFEATURES_H 

#include <opencv2/core.hpp>
#include <vector>

struct ArucoDetectionResult {
    cv::Mat image;
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
};

struct ShiTomasiDetectionResult {
    cv::Mat image;
    std::vector<cv::Point2f> points;
    std::vector<float> scores;
};

cv::Mat detectAndDrawHarris(const cv::Mat & img, int maxNumFeatures);
ShiTomasiDetectionResult detectAndDrawShiAndTomasi(const cv::Mat & img, int maxNumFeatures);
cv::Mat detectAndDrawFAST(const cv::Mat & img, int maxNumFeatures);
ArucoDetectionResult detectAndDrawArUco(const cv::Mat & img, int maxNumFeatures);

#endif

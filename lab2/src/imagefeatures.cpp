#include <string>
#include <print>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"

cv::Mat detectAndDrawHarris(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Harris detector expects grayscale input.

    cv::Mat dst;                             // Store scores computed by harris detector.
    cv::cornerHarris(gray, dst, 2, 3, 0.05); // input:output:neighborhoodsize:aperture:harrisparameter

    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal);
    float thresh = 0.01f * static_cast<float>(maxVal); // Keeping scores that are at least 1% as strong as the strongest corner.

    std::vector<std::pair<cv::Point, float>> corner_points; // Initialise vector for storing.
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            float score = dst.at<float>(i, j);
            if (score > thresh)
            {
                corner_points.emplace_back(cv::Point(j, i), score);
            }
        }
    }
    // Draw all detected corners above threshold in green
    for (const auto &[pt, score] : corner_points)
    {
        cv::circle(imgout, pt, 3, cv::Scalar(0, 255, 0), 1); // Green in BGR
    }

    // Sort by texture score (descending).
    std::sort(corner_points.begin(), corner_points.end(), [](const auto &a, const auto &b)
              {
                  return a.second > b.second; // Second element in corner_points
              });

    // Print statement setups:
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corner_points.size());
    std::println("{:<5} {:<10} {:<10} {:<10}", "Index", "X", "Y", "Score");

    // Limit to maxNumFeatures.
    if (corner_points.size() > static_cast<float>(maxNumFeatures))
    {
        corner_points.resize(maxNumFeatures);
    }

    // Print each feature
    for (size_t i = 0; i < corner_points.size(); ++i)
    {
        const auto &[pt, score] = corner_points[i];
        std::println("{:<5} {:<10} {:<10} {:.8f}", i + 1, pt.x, pt.y, score);
    }
    // TODO: Draw on output image
    for (size_t i = 0; i < corner_points.size(); ++i)
    {
        const auto &[pt, score] = corner_points[i];

        // Draw a red circle at the feature
        cv::circle(imgout, pt, 4, cv::Scalar(0, 0, 255), 1);

        // Label the feature with its index number
        std::string label = std::to_string(i + 1);
        cv::putText(imgout, label,
                    pt + cv::Point(5, -5), // offset text position
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.9,                   // font scale
                    cv::Scalar(255, 0, 0), // green text
                    2);                    // thickness
    }

    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Shi and Tomasi detector expects grayscale input.

    cv::Mat dst;                             // Store eigenvalues computed by detector.
    cv::cornerMinEigenVal(gray, dst, 2, 3); 

    double minVal, maxVal;
    cv::minMaxLoc(dst, &minVal, &maxVal);
    float thresh = 0.1f * static_cast<float>(maxVal); 

    std::vector<std::pair<cv::Point, float>> corner_points; // Initialise vector for storing.
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            float eig = dst.at<float>(i, j);
            if (eig > thresh)
            {
                corner_points.emplace_back(cv::Point(j, i), eig);
            }
        }
    }
    // Draw all detected corners above threshold in green
    for (const auto &[pt, eig] : corner_points)
    {
        cv::circle(imgout, pt, 3, cv::Scalar(0, 255, 0), 1); // Green in BGR
    }

    // Sort by eigen values(descending).
    std::sort(corner_points.begin(), corner_points.end(), [](const auto &a, const auto &b)
              {
                  return a.second > b.second; // Second element in corner_points
              });

    // Print statement setups:
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}", corner_points.size());
    std::println("{:<5} {:<10} {:<10} {:<10}", "Index", "X", "Y", "Eigenvalue");

    // Limit to maxNumFeatures.
    if (corner_points.size() > static_cast<float>(maxNumFeatures))
    {
        corner_points.resize(maxNumFeatures);
    }

    // Print each feature
    for (size_t i = 0; i < corner_points.size(); ++i)
    {
        const auto &[pt, eig] = corner_points[i];
        std::println("{:<5} {:<10} {:<10} {:.8f}", i + 1, pt.x, pt.y, eig);
    }
    // TODO: Draw on output image
    for (size_t i = 0; i < corner_points.size(); ++i)
    {
        const auto &[pt, eig] = corner_points[i];

        // Draw a red circle at the feature
        cv::circle(imgout, pt, 4, cv::Scalar(0, 0, 255), 1);

        // Label the feature with its index number
        std::string label = std::to_string(i + 1);
        cv::putText(imgout, label,
                    pt + cv::Point(5, -5), // offset text position
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.9,                   // font scale
                    cv::Scalar(255, 0, 0), // green text
                    2);                    // thickness
    }

    return imgout;
}

cv::Mat detectAndDrawFAST(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // FAST detector expects grayscale input.

    std::vector<cv::KeyPoint> keypoints;          // Store keypoints computed by FAST detector.
    cv::FAST(gray, keypoints, 50, true);         // input:output:threshold:nonMaxSupression

    // Draw all detected corners above threshold in green
    for (const auto& kp : keypoints)
    {
        cv::circle(imgout, kp.pt, 3, cv::Scalar(0, 255, 0), 1); // Green in BGR
    }

    // Sort by texture score (descending).
    std::sort(keypoints.begin(), keypoints.end(), [](const auto &a, const auto &b)
              {
                  return a.response > b.response;
              });

    // Print statement setups:
    std::println("Image width: {}", img.cols);
    std::println("Image height: {}", img.rows);
    std::println("Features requested: {}", maxNumFeatures);
    std::println("Features detected: {}",   keypoints.size());
    std::println("{:<5} {:<10} {:<10} {:<10}", "Index", "X", "Y", "Score");

    // Limit to maxNumFeatures.
    if (keypoints.size() > static_cast<float>(maxNumFeatures))
    {
        keypoints.resize(maxNumFeatures);
    }

    // Print each feature
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        const auto& kp = keypoints[i];
        std::println("{:<5} {:<10} {:<10} {:.8f}", i + 1, kp.pt.x, kp.pt.y, kp.response);
    }
    // TODO: Draw on output image
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        const auto& kp = keypoints[i];

        // Draw a red circle at the feature
        cv::circle(imgout, kp.pt, 4, cv::Scalar(0, 0, 255), 1);

        // Label the feature with its index number
        std::string label = std::to_string(i + 1);
        cv::putText(imgout, label,
                    kp.pt + cv::Point2f(5, -5), // offset text position
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.9,                   // font scale
                    cv::Scalar(255, 0, 0), // green text
                    2);                    // thickness
    }


    return imgout;
}

cv::Mat detectAndDrawArUco(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();

    // TODO

    return imgout;
}

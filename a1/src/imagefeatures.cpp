#include <string>
#include <print>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include "imagefeatures.h"

cv::Mat detectAndDrawHarris(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Harris detector expects grayscale input.
    cv::Mat dst; // Store scores computed by harris detector.

    cv::cornerHarris(gray, dst, 3, 3, 0.04); // input:output:neighborhoodsize:aperture:harrisparameter

    float thresh = 0.0004f; // Example absolute threshold
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
    
    // Sort by texture score (descending).
    std::sort(corner_points.begin(), corner_points.end(), [](const auto &a, const auto &b)
    {
        return a.second > b.second; // Second element in corner_points
    });
    
    // APPLY NON-MAXIMUM SUPPRESSION to corner_points
    int suppressionRadius = 10; // Minimum distance between corners
    std::vector<std::pair<cv::Point, float>> suppressed_points;
    
    for (const auto &[candidate_pt, candidate_score] : corner_points) 
    {
        bool suppress = false;
        for (const auto &[accepted_pt, accepted_score] : suppressed_points) 
        {
            double distance = cv::norm(candidate_pt - accepted_pt);
            if (distance < suppressionRadius) 
            {
                suppress = true;
                break;
            }
        }
        
        if (!suppress) 
        {
            suppressed_points.emplace_back(candidate_pt, candidate_score);
        }
    }
    
    corner_points = suppressed_points; // Replace with suppressed results
    
    // Draw all detected corners (now suppressed) in green
    for (const auto &[pt, score] : corner_points)
    {
        cv::circle(imgout, pt, 3, cv::Scalar(0, 255, 0), 1); // Green in BGR
    }
    
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
    
    // Draw on output image
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
                   0.9, // font scale
                   cv::Scalar(255, 0, 0), // green text
                   2); // thickness
    }
    
    return imgout;
}

cv::Mat detectAndDrawShiAndTomasi(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Shi and Tomasi detector expects grayscale input.
    cv::Mat dst; // Store eigenvalues computed by detector.
    cv::cornerMinEigenVal(gray, dst, 5, 3); // img:output:neighbourhoodsize:aperture

    float thresh = 0.009f; // Example absolute threshold

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
    
    // Sort by eigen values(descending).
    std::sort(corner_points.begin(), corner_points.end(), [](const auto &a, const auto &b)
    {
        return a.second > b.second; // Second element in corner_points
    });
    
    // APPLY NON-MAXIMUM SUPPRESSION to corner_points
    int suppressionRadius = 10; // Minimum distance between corners
    std::vector<std::pair<cv::Point, float>> suppressed_points;
    
    for (const auto &[candidate_pt, candidate_eig] : corner_points)
    {
        bool suppress = false;
        for (const auto &[accepted_pt, accepted_eig] : suppressed_points)
        {
            double distance = cv::norm(candidate_pt - accepted_pt);
            if (distance < suppressionRadius)
            {
                suppress = true;
                break;
            }
        }
        if (!suppress)
        {
            suppressed_points.emplace_back(candidate_pt, candidate_eig);
        }
    }
    
    corner_points = suppressed_points; // Replace with suppressed results
    // END SUPPRESSION
    
    // Draw all detected corners (now suppressed) in green
    for (const auto &[pt, eig] : corner_points)
    {
        cv::circle(imgout, pt, 3, cv::Scalar(0, 255, 0), 1); // Green in BGR
    }
    
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
    
    // Draw on output image
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
                   0.9, // font scale
                   cv::Scalar(255, 0, 0), // green text
                   2); // thickness
    }
    
    return imgout;
}

cv::Mat detectAndDrawFAST(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // FAST detector expects grayscale input.
    std::vector<cv::KeyPoint> keypoints; // Store keypoints computed by FAST detector.
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(65, true, cv::FastFeatureDetector::TYPE_9_16);
    detector->detect(gray, keypoints);    

    int suppressionRadius = 20;
    std::vector<cv::KeyPoint> suppressed_keypoints;
    for (const auto& candidate : keypoints) {
        bool suppress = false;
        for (const auto& accepted : suppressed_keypoints) {
            double distance = cv::norm(candidate.pt - accepted.pt);
            if (distance < suppressionRadius) {
                suppress = true;
                break;
            }
        }
        if (!suppress) {
            suppressed_keypoints.push_back(candidate);
        }
    }
    keypoints = suppressed_keypoints;

    // Sort by response score (descending).
    std::sort(keypoints.begin(), keypoints.end(), [](const auto &a, const auto &b)
    {
        return a.response > b.response;
    });
    
    // Draw all detected corners (now suppressed) in green
    for (const auto &kp : keypoints)
    {
        cv::circle(imgout, kp.pt, 3, cv::Scalar(0, 255, 0), 1); // Green in BGR
    }
    
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

ArucoDetectionResult detectAndDrawArUco(const cv::Mat &img, int maxNumFeatures)
{
    cv::Mat imgout = img.clone();
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Aruco detector expects grayscale input.

    // Build the ArUco detector function from online example.
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates; 
    std::vector<int> markerIds;

    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    detector.detectMarkers(gray, markerCorners, markerIds, rejectedCandidates);

    // Combine markerIds and markerCorners together and sort by ID
    std::vector<std::pair<int, std::vector<cv::Point2f>>> sortedMarkers;
    for (size_t i = 0; i < markerIds.size(); ++i) 
    {
        sortedMarkers.emplace_back(markerIds[i], markerCorners[i]);
    }
    std::sort(sortedMarkers.begin(), sortedMarkers.end(), [](const auto& a, const auto& b) 
    { 
        return a.first < b.first; 
    });

    // Then iterate through sorted markers
    for (const auto& [id, corners] : sortedMarkers)
    {
        // Draw green circles at all 4 corners
        for (const auto& pt : corners) {
            cv::circle(imgout, pt, 3, cv::Scalar(0, 255, 0), 1);
        }

        std::string label = std::to_string(id);
        cv::putText(imgout, label,
                    corners[0] + cv::Point2f(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.9,
                    cv::Scalar(255, 0, 0),
                    2);
    }

    // Return the result with image, IDs, and corners
    return {imgout, markerIds, markerCorners};
}

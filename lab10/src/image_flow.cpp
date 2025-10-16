#include <filesystem>
#include <string>
#include <print>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include "BufferedVideo.h"
#include "Camera.h"
#include "image_flow.h"

void runImageFlow(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();

    // TODO: Lab 10
    int divisor         = 2;                    // Image scaling factor
    int imgModulus      = 15;                    // Take frames divisible by this number
    int maxNumFeatures  = 700;                  // Maximum number of features per frame
    int minNumFeatures  = 600;                   // Minimum number of feature per frame

    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_"
                                   + std::to_string(divisor)
                                   + "_"
                                   + std::to_string(imgModulus)
                                   + videoPath.extension().string();
        outputPath = outputDirectory / outputFilename;
    }

    // Load camera calibration
    Camera camera;
    assert(std::filesystem::exists(cameraPath));
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::READ);
    assert(fs.isOpened());
    fs["camera"] >> camera;

    // Display loaded calibration data
    camera.printCalibration();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);

    std::println("Input video: {}", videoPath.string());
    std::println("Total number of frames: {}", nFrames);
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::println("Input video frame rate: {}", fps);
    std::println("Input video dimensions: [{} x {}]",
                cap.get(cv::CAP_PROP_FRAME_WIDTH),
                cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::println("{:>20}: {}", "maxNumFeatures", maxNumFeatures);
    std::println("{:>20}: {}", "minNumFeatures", minNumFeatures);
    std::println("{:>20}: {}", "divisor", divisor);
    std::println("{:>20}: {}", "imgModulus", imgModulus);
    std::println("");

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        double outputFps    = fps/imgModulus;
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Image flow
    // TODO: Lab 10
    cv::Mat imgk, imgkm1;
    std::vector<cv::Point2f> rQOik, rQOikm1;
    int frameIndex = 0;
    
    while (true)
    {
        // Extract images from video and create grayscale copy
        cv::Mat frame = bufferedVideoReader.read();
        if (frame.empty())
            break;
        
        // Only use frames divisible by imgModulus
        if (frameIndex % imgModulus != 0) {
            frameIndex++;
            continue;
        }
        
        std::println("Reading frame {}", frameIndex);
        
        // Resize if needed
        cv::Mat resizedFrame;
        if (divisor > 1) {
            cv::resize(frame, resizedFrame, cv::Size(), 1.0/divisor, 1.0/divisor, cv::INTER_AREA);
        } else {
            resizedFrame = frame;
        }
        
        // Convert to grayscale
        cv::cvtColor(resizedFrame, imgk, cv::COLOR_BGR2GRAY);
        
        // Initialize features on first frame or when count is too low
        if (imgkm1.empty() || rQOikm1.size() < minNumFeatures) {
            std::println("Initialise/reinitialise feature set.");
            
            cv::goodFeaturesToTrack(imgk, rQOik, maxNumFeatures, 0.01, 10);
            cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
            cv::cornerSubPix(imgk, rQOik, cv::Size(5, 5), cv::Size(-1, -1), criteria);
            
            std::println("Found {} features.", rQOik.size());
            
            imgk.copyTo(imgkm1);
            rQOikm1 = rQOik;
            frameIndex++;
            continue;
        }
        
        // For every appropriate frame:
        std::println("Calculating flow between frame {} and {}.", frameIndex, frameIndex - imgModulus);
        
        // Calculate optical flow
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(imgkm1, imgk, rQOikm1, rQOik, status, err);
        
        // Filter by status
        std::vector<cv::Point2f> rQOik_filtered, rQOikm1_filtered;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i]) {
                rQOik_filtered.push_back(rQOik[i]);
                rQOikm1_filtered.push_back(rQOikm1[i]);
            }
        }
        
        std::println("After filtering by status, there are {} associations.", rQOik_filtered.size());
        std::println("No. inliers = {}, No. outliers = {}", rQOik_filtered.size(), 0);
        
        std::vector<uchar> mask;
        int nInliers = 0;
        int nOutliers = 0;
        
        // Vectors to store inliers and outliers separately for display
        std::vector<cv::Point2f> rQOik_inliers, rQOikm1_inliers;
        std::vector<cv::Point2f> rQOik_outliers, rQOikm1_outliers;
        
        if (rQOik_filtered.size() >= 8) {  // Need at least 8 points for fundamental matrix
            // Scale points to original image size for undistortion
            std::vector<cv::Point2f> rQOik_scaled, rQOikm1_scaled;
            for (size_t i = 0; i < rQOik_filtered.size(); i++) {
                rQOik_scaled.push_back(cv::Point2f(rQOik_filtered[i].x * divisor, 
                                                    rQOik_filtered[i].y * divisor));
                rQOikm1_scaled.push_back(cv::Point2f(rQOikm1_filtered[i].x * divisor, 
                                                      rQOikm1_filtered[i].y * divisor));
            }
            
            // Convert to Eigen for undistortion
            Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_eigen(2, rQOik_scaled.size());
            Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_eigen(2, rQOikm1_scaled.size());
            for (size_t i = 0; i < rQOik_scaled.size(); i++) {
                rQOik_eigen(0, i) = rQOik_scaled[i].x;
                rQOik_eigen(1, i) = rQOik_scaled[i].y;
                rQOikm1_eigen(0, i) = rQOikm1_scaled[i].x;
                rQOikm1_eigen(1, i) = rQOikm1_scaled[i].y;
            }
            
            // Undistort points
            Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_undistorted = camera.undistort(rQOik_eigen);
            Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_undistorted = camera.undistort(rQOikm1_eigen);
            
            // Convert back to cv::Point2f
            std::vector<cv::Point2f> rQOik_undist_cv, rQOikm1_undist_cv;
            for (int i = 0; i < rQOik_undistorted.cols(); i++) {
                rQOik_undist_cv.push_back(cv::Point2f(rQOik_undistorted(0, i), 
                                                       rQOik_undistorted(1, i)));
                rQOikm1_undist_cv.push_back(cv::Point2f(rQOikm1_undistorted(0, i), 
                                                         rQOikm1_undistorted(1, i)));
            }
            
            // Find fundamental matrix on undistorted points
            double threshold = 1.0;  // Epipolar error threshold in pixels
            cv::Mat F = cv::findFundamentalMat(rQOikm1_undist_cv, rQOik_undist_cv, 
                                               cv::FM_RANSAC, threshold, 0.99, mask);
            
            // Separate inliers and outliers
            for (size_t i = 0; i < mask.size(); i++) {
                if (mask[i]) {
                    nInliers++;
                    rQOik_inliers.push_back(rQOik_filtered[i]);
                    rQOikm1_inliers.push_back(rQOikm1_filtered[i]);
                } else {
                    nOutliers++;
                    rQOik_outliers.push_back(rQOik_filtered[i]);
                    rQOikm1_outliers.push_back(rQOikm1_filtered[i]);
                }
            }
            
            // Use only inliers for next iteration
            rQOik_filtered = rQOik_inliers;
            rQOikm1_filtered = rQOikm1_inliers;
        } else {
            nInliers = rQOik_filtered.size();
            rQOik_inliers = rQOik_filtered;
            rQOikm1_inliers = rQOikm1_filtered;
        }
        
        std::println("No. inliers = {}, No. outliers = {}", nInliers, nOutliers);
        
        // Display flow field
        cv::Mat displayFrame = resizedFrame.clone();  
        
        // Draw outliers in red
        for (size_t i = 0; i < rQOik_outliers.size(); i++) {
            cv::arrowedLine(displayFrame, rQOikm1_outliers[i], rQOik_outliers[i], 
                          cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.1);
        }
        
        // Draw inliers in green (on top of outliers)
        for (size_t i = 0; i < rQOik_inliers.size(); i++) {
            cv::arrowedLine(displayFrame, rQOikm1_inliers[i], rQOik_inliers[i], 
                          cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.1);
        }
        
        cv::imshow("LK Demo", displayFrame);
        
        if (doExport) {
            bufferedVideoWriter.write(displayFrame);
        }
        
        char key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
        
        // v) Copy current to previous
        imgk.copyTo(imgkm1);
        rQOikm1 = rQOik_filtered;
        
        frameIndex++;
    }

    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
}

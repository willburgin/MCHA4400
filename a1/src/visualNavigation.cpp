#include <filesystem>
#include <string>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "imagefeatures.h"
#include "Camera.h"

void runVisualNavigationFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, int scenario, int interactive, const std::filesystem::path & outputDirectory)
{
    assert(!videoPath.empty());

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_out"
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
    double fps = cap.get(cv::CAP_PROP_FPS);

    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = 2*cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double outputFps    = fps;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual navigation

    // Initialisation
    // int numInitialLandmarks = 5;
    // int stateDim = 12 + 6*numInitialLandmarks;

    // Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(stateDim, stateDim);
    // P0.block<3,3>(0,0) *= 1.0;
    // P0.block<3,3>(3,3) *= 1.0;
    // P0.block<3,3>(6,6) *= 1.0;
    // P0.block<3,3>(9,9) *= 1.0;
    // for (int i = 0; i < numInitialLandmarks; ++i)
    // {
    //     int idx = 12 + 6*i;
    //     P0.block<3,3>(idx, idx) *= 100.0; // landmark position
    //     P0.block<3,3>(idx+3, idx+3) *= 10.0; // landmark orientation
    // }


    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }
        double dt = 1/fps;
        cv::Mat outputFrame;
        if (scenario == 1)
        {
            outputFrame = detectAndDrawArUco(imgin, 20);
            
        }
        else if (scenario == 3)
        {
            outputFrame = detectAndDrawShiAndTomasi(imgin, 20);
        }
        
        // Process frame

        // Update state

        // Update plot

        // Write output frame 
        // TODO: Display image returned from detectAndDraw on screen and wait for 1000/fps milliseconds
        cv::imshow("Video Feature Detection", outputFrame);
        int delay = static_cast<int>(1000.0 / fps);
        if (cv::waitKey(delay) == 27) 
        {
            break;
        }
        if (doExport)
        {
            // cv::Mat imgout; /* plot.getFrame();*/ // TODO: Uncomment this to get the frame image
            cv::Mat imgout;
            bufferedVideoWriter.write(imgout);
        }
    }

    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
}

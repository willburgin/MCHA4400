#include <iostream>
#include <print>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <filesystem>
#include "calibrate.h"
#include "Camera.h"
#include "serialisation.hpp"
#include "BufferedVideo.h"

void calibrateCamera(const std::filesystem::path & configPath)
{
    // TODO
    // - Read XML at configPath
    cv::FileStorage fs(configPath.string(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open config file: " << configPath.string() << std::endl;
        return;
    }
    ChessboardData chessboardData(configPath);

    // Perform camera calibration
    Camera camera;
    camera.calibrate(chessboardData);

    // Write the camera matrix and lens distortion parameters to camera.xml file
    cv::FileStorage outputFs("../data/camera.xml", cv::FileStorage::WRITE);
    outputFs<<"camera"<<camera; 
    outputFs.release();

    chessboardData.drawCorners();
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        cv::namedWindow("Calibration images (press ESC to quit, any other key to continue)", cv::WINDOW_NORMAL);
        cv::resizeWindow("Calibration images (press ESC to quit, any other key to continue)", 1024, 768);
        cv::imshow("Calibration images (press ESC to quit, any other key to continue)", chessboardImage.image);
        char c = static_cast<char>(cv::waitKey(0));
        if (c == 27) // ESC to quit, any other key to continue
            break;
    }
    cv::destroyAllWindows();
}


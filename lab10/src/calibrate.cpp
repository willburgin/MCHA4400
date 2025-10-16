#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Camera.h"
#include "calibrate.h"

void calibrateCamera(const std::filesystem::path & configPath)
{
    // Read chessboard data using configuration file
    ChessboardData chessboardData(configPath);

    // Calibrate camera from chessboard data
    Camera cam;
    cam.calibrate(chessboardData);

    // Write camera calibration to file
    std::filesystem::path cameraPath = configPath.parent_path() / "camera.xml";
    cv::FileStorage fs(cameraPath.string(), cv::FileStorage::WRITE);
    fs << "camera" << cam;
    fs.release();

    // Visualise the camera calibration results
    chessboardData.drawBoxes(cam);
    
    // Create a named window that can be resized
    cv::namedWindow("Calibration images", cv::WINDOW_NORMAL);
    cv::resizeWindow("Calibration images", 1280, 720);  // Adjust these dimensions as needed
    
    for (const auto & chessboardImage : chessboardData.chessboardImages)
    {
        cv::imshow("Calibration images", chessboardImage.image);
        char c = static_cast<char>(cv::waitKey(0));
        if (c == 27 || c == 'q' || c == 'Q') // ESC, q or Q to quit, any other key to continue
            break;
    }
}

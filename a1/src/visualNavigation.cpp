#include <filesystem>
#include <string>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include "BufferedVideo.h"
#include "visualNavigation.h"
#include "imagefeatures.h"
#include "Camera.h"
#include "SystemSLAMPoseLandmarks.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "Plot.h"

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

    Eigen::Matrix3d Rbc;
    Rbc << 0, 0, 1,   // b1 = c3
            1, 0, 0,   // b2 = c1
            0, 1, 0;   // b3 = c2

    camera.Tbc.rotationMatrix = Rbc;
    camera.Tbc.translationVector = Eigen::Vector3d::Zero();

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

    Plot plot(camera);
    
    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    bool videoWriterOpened = false;
    // SLAM System Initialization for Scenario 1
    std::unique_ptr<SystemSLAMPoseLandmarks> system;
    std::unique_ptr<MeasurementSLAMUniqueTagBundle> measurement;
    bool systemInitialized = false;
    
    double time = 0.0;

    // Track frame number for interactive mode
    static int frameCount = 0;


    while (true)
    {
        // Get next input frame
        cv::Mat imgin = bufferedVideoReader.read();
        if (imgin.empty())
        {
            break;
        }
        double dt = 1/fps;
        time += dt;
        
        cv::Mat outputFrame;
        if (scenario == 1)
        {
            auto result = detectAndDrawArUco(imgin, 20);
            outputFrame = result.image;
            
            if (!systemInitialized)
            {
                // Initialize system with NO landmarks initially - let it grow dynamically
                int stateDim = 12; // Only body states initially
                
                Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(stateDim);
                initialMean(8) = -1.8;
                Eigen::MatrixXd initialCov = Eigen::MatrixXd::Identity(stateDim, stateDim);

                // best solution so far - individual state scaling:
                initialCov.diagonal()(0) *= 0.3;  // vx
                initialCov.diagonal()(1) *= 0.3;  // vy  
                initialCov.diagonal()(2) *= 0.3;  // vz
                
                initialCov.diagonal()(3) *= 0.2;  // wx
                initialCov.diagonal()(4) *= 0.2;  // wy
                initialCov.diagonal()(5) *= 0.25;  // wz
                
                initialCov.diagonal()(6) *= 0.05;  // x
                initialCov.diagonal()(7) *= 0.05;  // y
                initialCov.diagonal()(8) *= 0.005;  // z
                
                initialCov.diagonal()(9) *= 0.08;  // roll
                initialCov.diagonal()(10) *= 0.08; // pitch
                initialCov.diagonal()(11) *= 0.08; // yaw

                // initialCov.diagonal().head(6) *= 0.001;
                // initialCov.diagonal().segment(6, 6) *= 0.001;
                
                auto initialDensity = GaussianInfo<double>::fromMoment(initialMean, initialCov);
                system = std::make_unique<SystemSLAMPoseLandmarks>(initialDensity);
                systemInitialized = true;
            }
            
            // SLAM ESTIMATION LOOP - let the system grow dynamically
            
            // 1. Predict state forward in time
            std::cout << "Before predict - system dim: " << system->density.dim() << std::endl;
            system->predict(time);
            std::cout << "After predict - system dim: " << system->density.dim() << std::endl;

            
            // 2. Create measurement with ALL detected ArUco data (even new markers)
            if (!result.markerIds.empty())
            {
                int numMarkers = result.markerIds.size();
                Eigen::Matrix<double, 8, Eigen::Dynamic> Y(8, numMarkers);  // 8 coordinates per marker (4 corners × 2 coordinates)
                
                for (int i = 0; i < numMarkers; ++i)
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        Y(2*j, i) = result.markerCorners[i][j].x;     // x coordinate of corner j
                        Y(2*j + 1, i) = result.markerCorners[i][j].y; // y coordinate of corner j
                    }
                }
                std::cout << "Y: " << Y << std::endl;
                measurement = std::make_unique<MeasurementSLAMUniqueTagBundle>(time, Y, camera);
                // Set the detected marker IDs
                measurement->setFrameMarkerIDs(result.markerIds);                
                // 3. Process measurement - this handles data association and landmark initialization
                std::cout << "Before measurement update - system dim: " << system->density.dim() << std::endl;
                measurement->process(*system);
                std::cout << "After measurement update - system dim: " << system->density.dim() << std::endl;

            }
            
            // 4. Update visualization
            system->view() = outputFrame.clone();
            if (measurement) {
                plot.setData(*system, *measurement);
                plot.render();
            }
        }
        else if (scenario == 3)
        {
            outputFrame = detectAndDrawShiAndTomasi(imgin, 20);
        }
        
        frameCount++;
        // Handle interactive mode
        bool isLastFrame = (frameCount == nFrames);
        if (interactive == 2 || (interactive == 1 && isLastFrame))
        {
            // Start handling plot GUI events (blocking)
            plot.start();
        }

        if (doExport && systemInitialized && measurement)
        {
            cv::Mat imgout = plot.getFrame();
            
            // Open video writer on first frame (like Lab 3)
            if (!videoWriterOpened)
            {
                int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                videoOut.open(outputPath.string(), codec, fps, imgout.size());
                
                if (!videoOut.isOpened()) {
                    std::cerr << "Failed to open video writer!" << std::endl;
                    doExport = false;
                } else {
                    bufferedVideoWriter.start(videoOut);
                    videoWriterOpened = true;
                    std::cout << "Video writer opened: " << imgout.size() << " @ " << fps << " fps" << std::endl;
                }
            }
            
            if (videoWriterOpened) {
                bufferedVideoWriter.write(imgout);
            }
        }
    }

    // After the main while loop, ensure cleanup happens
    if (doExport)
    {
        std::cout << "Finalizing video export..." << std::endl;
        bufferedVideoWriter.stop();
        videoOut.release();  // Explicitly release the video writer
        std::cout << "Video export complete: " << outputPath << std::endl;
    }
    bufferedVideoReader.stop();
}

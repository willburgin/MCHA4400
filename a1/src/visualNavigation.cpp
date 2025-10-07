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
#include "SystemSLAMPointLandmarks.h"
#include "MeasurementSLAMUniqueTagBundle.h"
#include "MeasurementSLAMDuckBundle.h"
#include "Plot.h"
#include "DuckDetectorONNX.h"

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
    // align frame of body to camera frame
    Eigen::Matrix3d Rbc;
    Rbc << 0, 0, 1,     // b1 = c3
            1, 0, 0,   // b2 = c1
            0, 1, 0;   // b3 = c2

    camera.Tbc.rotationMatrix = Rbc;
    camera.Tbc.translationVector = Eigen::Vector3d::Zero();

    // display loaded calibration data
    camera.printCalibration();

    // open input video
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
    // SLAM system initialization (base class pointer works here)
    std::unique_ptr<SystemSLAM> system;
    
    // Scenario-specific measurement pointers
    std::unique_ptr<MeasurementSLAMUniqueTagBundle> tagMeasurement;
    std::unique_ptr<MeasurementDuckBundle> duckMeasurement;
    
    bool systemInitialized = false;
    
    // Duck detector initialization for scenario 2
    std::unique_ptr<DuckDetectorONNX> duckDetector;
    std::filesystem::path outputImageDirectory;
    if (scenario == 2)
    {
        std::filesystem::path modelPath = "../data/duck_with_postprocessing.onnx";
        duckDetector = std::make_unique<DuckDetectorONNX>(modelPath.string());
        std::cout << "Duck detector initialized with model: " << modelPath << std::endl;
        
        // Create images subdirectory for scenario 2
        if (doExport)
        {
            outputImageDirectory = outputDirectory / "images";
            if (!std::filesystem::exists(outputImageDirectory))
            {
                std::cout << "Creating directory: " << outputImageDirectory << std::endl;
                std::filesystem::create_directories(outputImageDirectory);
            }
        }
    }
    
    double time = 0.0;

    // track frame number for interactive mode
    static int frameCount = 0;


    while (true)
    {
        // get next input frame
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
                // initialize system with NO landmarks initially - let it grow dynamically
                int stateDim = 12; // only body states initially
                
                Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(stateDim);
                initialMean(8) = -1.8;
                Eigen::MatrixXd initialCov = Eigen::MatrixXd::Identity(stateDim, stateDim);

                // best solution so far - individual state scaling
                initialCov.diagonal()(0) *= 0.3;  // vx
                initialCov.diagonal()(1) *= 0.3;  // vy  
                initialCov.diagonal()(2) *= 0.3;  // vz
                
                initialCov.diagonal()(3) *= 0.2;  // wx
                initialCov.diagonal()(4) *= 0.2;  // wy
                initialCov.diagonal()(5) *= 0.3;  // wz
                
                initialCov.diagonal()(6) *= 0.01;  // x
                initialCov.diagonal()(7) *= 0.01;  // y
                initialCov.diagonal()(8) *= 0.01;  // z
                
                initialCov.diagonal()(9) *= 0.01;  // roll
                initialCov.diagonal()(10) *= 0.01; // pitch
                initialCov.diagonal()(11) *= 0.01; // yaw
                
                auto initialDensity = GaussianInfo<double>::fromMoment(initialMean, initialCov);
                system = std::make_unique<SystemSLAMPoseLandmarks>(initialDensity);
                systemInitialized = true;
            }
            
            // SLAM estimation loop - let the system grow dynamically
            
            // predict state forward in time
            // create measurement with ALL detected ArUco data (even new markers)
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
                tagMeasurement = std::make_unique<MeasurementSLAMUniqueTagBundle>(time, Y, camera);
                tagMeasurement->setFrameMarkerIDs(result.markerIds);
                tagMeasurement->process(*system);

            }
            
            // update visualization
            system->view() = outputFrame.clone();
            if (tagMeasurement) {
                plot.setData(*system, *tagMeasurement);
                plot.render();
            }
        }
        else if (scenario == 2)
        {
            // Run duck detector
            std::string baseName = videoPath.stem().string();
            std::string frameName = baseName + "_" + std::to_string(frameCount);
            std::cout << "Processing " << frameName << "..." << std::flush;
            
            auto detections = duckDetector->detect(imgin);
            outputFrame = detections.image;  // Annotated image from DuckDetectionResult struct
            std::cout << " done" << std::endl;
            
            // Initialize system on first frame
            if (!systemInitialized)
            {
                int stateDim = 12; // Only body states initially
                Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(stateDim);
                initialMean(8) = -1.0;

                Eigen::MatrixXd initialCov = Eigen::MatrixXd::Identity(stateDim, stateDim);
                // Scale initial uncertainties
                initialCov.diagonal()(0) *= 0.01;  // vx
                initialCov.diagonal()(1) *= 0.01;  // vy
                initialCov.diagonal()(2) *= 0.01;  // vz
                initialCov.diagonal()(3) *= 0.09;  // wx
                initialCov.diagonal()(4) *= 0.09;  // wy
                initialCov.diagonal()(5) *= 0.09;  // wz
                initialCov.diagonal()(6) *= 0.005; // x
                initialCov.diagonal()(7) *= 0.005; // y
                initialCov.diagonal()(8) *= 0.005; // z
                initialCov.diagonal()(9) *= 0.001; // roll
                initialCov.diagonal()(10) *= 0.001; // pitch
                initialCov.diagonal()(11) *= 0.001; // yaw
                
                auto initialDensity = GaussianInfo<double>::fromMoment(initialMean, initialCov);
                system = std::make_unique<SystemSLAMPointLandmarks>(initialDensity);
                systemInitialized = true;
            }
            
            // Create measurement from duck detections
            if (!detections.centroids.empty())
            {
                int numDucks = detections.centroids.size();
                Eigen::Matrix<double, 3, Eigen::Dynamic> Y(3, numDucks);  // 3 rows: x, y, area
                
                for (int i = 0; i < numDucks; ++i)
                {
                    Y(0, i) = detections.centroids[i].x;  // Centroid x
                    Y(1, i) = detections.centroids[i].y;  // Centroid y
                    Y(2, i) = detections.areas[i];         // Area in pixels²
                }
                
                duckMeasurement = std::make_unique<MeasurementDuckBundle>(time, Y, camera);
                duckMeasurement->process(*system);
            }
            
            // Update visualization
            system->view() = outputFrame.clone();
            if (duckMeasurement) {
                plot.setData(*system, *duckMeasurement);
                plot.render();
            }
            
            // Save images if exporting
            if (doExport && (frameCount % 15 == 0))
            {
                std::string frameFilename = frameName + ".png";
                std::string frameFilenameEval = frameName + "_eval.png";
                std::filesystem::path framePath = outputImageDirectory / frameFilename;
                cv::imwrite(framePath.string(), imgin);
                std::filesystem::path frameAnnotatedPath = outputImageDirectory / frameFilenameEval;
                cv::imwrite(frameAnnotatedPath.string(), outputFrame);
                std::cout << " Saved: " << frameFilename << " and " << frameFilenameEval << std::endl;
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

        if (doExport)
        {
            cv::Mat imgout;
            
            // Get the appropriate output frame based on scenario
            if (scenario == 1 && systemInitialized && tagMeasurement)
            {
                imgout = plot.getFrame();
            }
            else if (scenario == 2)
            {
                imgout = outputFrame;
            }
            else if (scenario == 3)
            {
                imgout = outputFrame;
            }
            
            // Write frame if we have output
            if (!imgout.empty())
            {
                // Open video writer on first frame
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

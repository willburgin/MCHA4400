#include <filesystem>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
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

// Structure to hold cached duck detections for a single frame
struct CachedDuckDetections
{
    std::vector<cv::Point2f> centroids;
    std::vector<int> areas;
};

// Save duck detections to file
void saveDuckDetectionsToFile(const std::filesystem::path& cachePath, 
                               const std::vector<CachedDuckDetections>& allDetections)
{
    std::ofstream file(cachePath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open cache file for writing: " << cachePath << std::endl;
        return;
    }
    
    // Write number of frames
    size_t numFrames = allDetections.size();
    file.write(reinterpret_cast<const char*>(&numFrames), sizeof(numFrames));
    
    // Write each frame's detections
    for (const auto& frameDetections : allDetections)
    {
        // Write number of detections in this frame
        size_t numDetections = frameDetections.centroids.size();
        file.write(reinterpret_cast<const char*>(&numDetections), sizeof(numDetections));
        
        // Write centroids
        for (const auto& centroid : frameDetections.centroids)
        {
            file.write(reinterpret_cast<const char*>(&centroid.x), sizeof(centroid.x));
            file.write(reinterpret_cast<const char*>(&centroid.y), sizeof(centroid.y));
        }
        
        // Write areas
        for (const auto& area : frameDetections.areas)
        {
            file.write(reinterpret_cast<const char*>(&area), sizeof(area));
        }
    }
    
    file.close();
    std::cout << "Saved duck detections cache to: " << cachePath << std::endl;
}

// Load duck detections from file
bool loadDuckDetectionsFromFile(const std::filesystem::path& cachePath,
                                std::vector<CachedDuckDetections>& allDetections)
{
    std::ifstream file(cachePath, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }
    
    // Read number of frames
    size_t numFrames;
    file.read(reinterpret_cast<char*>(&numFrames), sizeof(numFrames));
    allDetections.resize(numFrames);
    
    // Read each frame's detections
    for (auto& frameDetections : allDetections)
    {
        // Read number of detections in this frame
        size_t numDetections;
        file.read(reinterpret_cast<char*>(&numDetections), sizeof(numDetections));
        
        frameDetections.centroids.resize(numDetections);
        frameDetections.areas.resize(numDetections);
        
        // Read centroids
        for (auto& centroid : frameDetections.centroids)
        {
            file.read(reinterpret_cast<char*>(&centroid.x), sizeof(centroid.x));
            file.read(reinterpret_cast<char*>(&centroid.y), sizeof(centroid.y));
        }
        
        // Read areas
        for (auto& area : frameDetections.areas)
        {
            file.read(reinterpret_cast<char*>(&area), sizeof(area));
        }
    }
    
    file.close();
    std::cout << "Loaded duck detections cache from: " << cachePath << std::endl;
    std::cout << "Total frames in cache: " << numFrames << std::endl;
    return true;
}

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
    std::vector<CachedDuckDetections> cachedDetections;
    bool useCachedDetections = false;
    
    if (scenario == 2)
    {
        // Define cache file path
        std::filesystem::path cacheDirectory = "../data";
        std::string cacheFilename = videoPath.stem().string() + "_duck_detections.bin";
        std::filesystem::path cachePath = cacheDirectory / cacheFilename;
        
        // Try to load cached detections
        if (std::filesystem::exists(cachePath))
        {
            std::cout << "Found cached duck detections file." << std::endl;
            if (loadDuckDetectionsFromFile(cachePath, cachedDetections))
            {
                useCachedDetections = true;
                std::cout << "Using cached duck detections (performance optimized)." << std::endl;
            }
            else
            {
                std::cerr << "Failed to load cache file. Will run detector online." << std::endl;
            }
        }
        
        // If no cache exists or loading failed, run detector and create cache
        if (!useCachedDetections)
        {
            std::cout << "No cache found. Running duck detector offline to create cache..." << std::endl;
            
            std::filesystem::path modelPath = "../data/duck_with_postprocessing.onnx";
            duckDetector = std::make_unique<DuckDetectorONNX>(modelPath.string());
            
            // Run detector on all frames and cache results
            cv::VideoCapture tempCap(videoPath.string());
            cachedDetections.reserve(nFrames);
            
            int frameIdx = 0;
            while (true)
            {
                cv::Mat frame;
                tempCap >> frame;
                if (frame.empty()) break;
                
                std::cout << "\rProcessing frame " << (frameIdx + 1) << "/" << nFrames << std::flush;
                
                auto detections = duckDetector->detect(frame);
                CachedDuckDetections frameCache;
                frameCache.centroids = detections.centroids;
                frameCache.areas = detections.areas;
                cachedDetections.push_back(frameCache);
                
                frameIdx++;
            }
            std::cout << std::endl;
            
            tempCap.release();
            
            // Save cache to file
            saveDuckDetectionsToFile(cachePath, cachedDetections);
            useCachedDetections = true;
            
            std::cout << "Duck detection cache created successfully." << std::endl;
        }
        
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
            std::string baseName = videoPath.stem().string();
            std::string frameName = baseName + "_" + std::to_string(frameCount);
            
            // Get detections from cache
            std::vector<cv::Point2f> centroids;
            std::vector<int> areas;
            
            if (useCachedDetections && frameCount < cachedDetections.size())
            {
                centroids = cachedDetections[frameCount].centroids;
                areas = cachedDetections[frameCount].areas;
                std::cout << "Using cached detections for " << frameName 
                          << " (" << centroids.size() << " ducks)" << std::endl;
            }
            
            // Create visualization (optional - you can skip this for pure performance)
            imgin.convertTo(outputFrame, CV_8UC3, 0.5, 0); // Darken the image
            
            // Draw detections on output frame
            for (size_t i = 0; i < centroids.size(); ++i)
            {
                cv::Point center(cvRound(centroids[i].x), cvRound(centroids[i].y));
                
                // Draw black X at centroid
                int len = 3;
                cv::line(outputFrame, center - cv::Point(len, len), center + cv::Point(len, len), cv::Scalar(0, 0, 0), 2);
                cv::line(outputFrame, center - cv::Point(-len, len), center + cv::Point(-len, len), cv::Scalar(0, 0, 0), 2);
                
                // Overlay info
                std::string text = "ID=" + std::to_string(i) +
                                " (" + std::to_string(cvRound(centroids[i].x)) +
                                "," + std::to_string(cvRound(centroids[i].y)) +
                                ") A=" + std::to_string(areas[i]);
                
                cv::putText(outputFrame, text, center + cv::Point(6, -6),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                           cv::LINE_AA);
            }
            
            // Initialize system on first frame
            if (!systemInitialized)
            {
                int stateDim = 12; // Only body states initially
                Eigen::VectorXd initialMean = Eigen::VectorXd::Zero(stateDim);
                initialMean(8) = -1.0;

                Eigen::MatrixXd initialCov = Eigen::MatrixXd::Identity(stateDim, stateDim);
                
                initialCov.diagonal()(0) *= 0.4;  // vx
                initialCov.diagonal()(1) *= 0.1;  // vy
                initialCov.diagonal()(2) *= 0.1;  // vz
                initialCov.diagonal()(3) *= 0.1;  // wx
                initialCov.diagonal()(4) *= 0.1;  // wy
                initialCov.diagonal()(5) *= 0.4;  // wz
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
            
            // Create measurement from cached duck detections
            if (!centroids.empty())
            {
                int numDucks = centroids.size();
                Eigen::Matrix<double, 3, Eigen::Dynamic> Y(3, numDucks);  // 3 rows: x, y, area
                
                for (int i = 0; i < numDucks; ++i)
                {
                    Y(0, i) = centroids[i].x;  // Centroid x
                    Y(1, i) = centroids[i].y;  // Centroid y
                    Y(2, i) = areas[i];         // Area in pixels²
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
            else if (scenario == 2 && systemInitialized && duckMeasurement)
            {
                imgout = plot.getFrame();
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

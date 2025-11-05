#include <print>
#include <numbers>
#include <filesystem>
#include <Eigen/Core>
#include <vector>
#include <deque>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "BufferedVideo.h"
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "rotation.hpp"
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"
#include "SystemVisualNavPoseLandmarks.h"
#include "MeasurementOutdoorFlowBundle.h"
#include "MeasurementIndoorFlowBundle.h"
#include "visualNavigation.h"
#include "MeasurementAltimeter.h"
#include "MeasurementSLAMIdenticalTagBundle.h"
#include "imagefeatures.h"
#include "Plot.h"

// Forward declarations (reuse from visual odometry)
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
static void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotTransVelocity(cv::Mat & img, const Eigen::VectorXd & x, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);
static void plotStateInfo(cv::Mat & img, const Eigen::Vector6d & etak, const std::vector<DJIVideoCaption> & djiVideoCaption, int frameIndex, int scenario);
static void plotTrajectory2D(cv::Mat & img, const Eigen::Vector6d & etak, const std::vector<DJIVideoCaption> & djiVideoCaption, int frameIndex);
static std::deque<Eigen::Vector2d> estimatedTraj;
static std::deque<Eigen::Vector2d> gpsTraj;
static cv::Mat trajImg;
static Eigen::Vector2d gpsOrigin(0, 0);
static bool gpsOriginSet = false;

void runVisualNavigationFromVideo(
    const std::filesystem::path & videoPath, 
    const std::filesystem::path & cameraPath, 
    int scenario, 
    int interactive, 
    const std::filesystem::path & outputDirectory)
{
    int imgModulus = 1;
    // TODO: Tune these parameters
    if (scenario == 4) {
        imgModulus  = 10;        // Take frames divisible by this number
    } else if (scenario == 5) {
        imgModulus  = 1;
    }
    int divisor     = 2;         // Image scaling factor (used for plotting only)

    assert(!videoPath.empty());

    // Subtitle path and GPS data (Scenario 4 only)
    std::filesystem::path subtitlePath;
    std::vector<DJIVideoCaption> djiVideoCaption;

    if (scenario == 4) {
        subtitlePath = videoPath.parent_path() / (videoPath.stem().string() + ".SRT");
        assert(std::filesystem::exists(subtitlePath));
        std::println("Subtitle file: {}", subtitlePath.string());
        
        // Load and parse subtitle file
        djiVideoCaption = getVideoCaptions(subtitlePath);
    }

    // Output video path
    std::filesystem::path outputPath;
    bool doExport = !outputDirectory.empty();
    if (doExport)
    {
        std::string outputFilename = videoPath.stem().string()
                                   + "_"
                                   + std::to_string(divisor)
                                   + "_"
                                   + std::to_string(imgModulus)
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

    // Set camera pose w.r.t. body
    Eigen::Matrix3d Rbc;
    Rbc <<  0, 0, 1,     // b1 = c3
            1, 0, 0,     // b2 = c1
            0, 1, 0;     // b3 = c2
    camera.Tbc.rotationMatrix = Rbc;
    camera.Tbc.translationVector = Eigen::Vector3d::Zero();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);
    double fps = cap.get(cv::CAP_PROP_FPS);

    std::println("Input video: {}", videoPath.string());
    std::println("Total number of frames: {}", nFrames);
    std::println("Input video frame rate: {}", fps);
    std::println("Input video dimensions: [{} x {}]",
                cap.get(cv::CAP_PROP_FRAME_WIDTH),
                cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::println("BufferedVideoReader created");
    BufferedVideoReader bufferedVideoReader(5);
    bufferedVideoReader.start(cap);

    cv::VideoWriter videoOut;
    BufferedVideoWriter bufferedVideoWriter(3);
    if (doExport)
    {
        cv::Size frameSize;
        frameSize.width     = cap.get(cv::CAP_PROP_FRAME_WIDTH)/divisor;
        frameSize.height    = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/divisor;
        double outputFps    = fps/imgModulus;
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }
    
    Eigen::Vector6d initialPose;
    if (scenario == 4) {
        initialPose = getInitialPose(djiVideoCaption[0]);
    } else {
        // Scenario 5: Indoor, start at origin
        initialPose.setZero();
    }

    // Define transformation from z to x
    auto phimap = [](const Eigen::VectorXd& z, Eigen::MatrixXd& J) {
        Eigen::VectorXd x(18);
        x.segment<6>(0) = z.segment<6>(0);   // nu
        x.segment<6>(6) = z.segment<6>(6);   // eta
        x.segment<6>(12) = z.segment<6>(6);  // zeta = eta initially
        
        // Jacobian: dx/dz
        J.resize(18, 12);
        J.setZero();
        J.block<6,6>(0,0).setIdentity();   // dnu/dnu = I
        J.block<6,6>(6,6).setIdentity();   // deta/deta = I
        J.block<6,6>(12,6).setIdentity();  // dzeta/deta = I (since zeta=eta)
        
        return x;
    };

    // Start with 12-dim uncertainty on [nu; eta]
    Eigen::VectorXd z0(12);
    z0.segment<6>(0).setZero();  // nu = 0
    z0.segment<6>(6) = initialPose;  // eta from GPS

    Eigen::MatrixXd Pz(12, 12);
    Pz.setZero();
    if (scenario == 4) {
        Pz.block<3,3>(0,0) = 2.0 * Eigen::Matrix3d::Identity();  // velocity uncertainty
        Pz.block<3,3>(3,3) = 2.0 * Eigen::Matrix3d::Identity();  // angular velocity
        Pz.block<3,3>(6,6) = 0.001 * Eigen::Matrix3d::Identity(); // position
        Pz.block<3,3>(9,9) = 0.001 * Eigen::Matrix3d::Identity(); // orientation
    } else if (scenario == 5) {
        std::println("Pz created");
        Pz.block<3,3>(0,0) = 0.3 * Eigen::Matrix3d::Identity();  // velocity uncertainty
        Pz.block<3,3>(3,3) = 0.3 * Eigen::Matrix3d::Identity();  // angular velocity
        Pz.block<3,3>(6,6) = 0.01 * Eigen::Matrix3d::Identity(); // position
        Pz.block<3,3>(9,9) = 0.01 * Eigen::Matrix3d::Identity(); // orientation
    }

    GaussianInfo<double> densityZ = GaussianInfo<double>::fromSqrtMoment(z0, Pz);

    // Transform to 18-dim with proper correlations
    GaussianInfo<double> initialDensity = densityZ.affineTransform(phimap);
    
    // Create system based on scenario
    std::unique_ptr<SystemVisualNav> systemPtr;
    if (scenario == 4) {
        systemPtr = std::make_unique<SystemVisualNav>(initialDensity);
    } else if (scenario == 5) {
        systemPtr = std::make_unique<SystemVisualNavPoseLandmarks>(initialDensity);
    }
    std::unique_ptr<MeasurementSLAMIdenticalTagBundle> measTags;
    SystemVisualNav & system = *systemPtr;
    
    // Create Plot for scenario 5
    std::unique_ptr<Plot> plot;
    if (scenario == 5) {
        std::println("Plot created");
        plot = std::make_unique<Plot>(camera);
        // plot->start();
    }
    
    // Feature tracking state (similar to visual odometry)
    cv::Mat imgk_raw;
    cv::Mat imgkm1_raw;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1;
    
    // Variables for plotting
    Eigen::Vector6d etakm1;
    if (!djiVideoCaption.empty())
    {
        etakm1 = getInitialPose(djiVideoCaption[0]);
    }
    else
    {
        etakm1.setZero();
    }

    std::println("Starting visual navigation (Scenario {})...", scenario);
    double previousAltitude = -1.0;
    
    // Main Processing Loop 
    int k = 0;
    for (int i = 0;; ++i)
    {
        // Capture frame by frame
        imgk_raw = bufferedVideoReader.read();
        if (imgk_raw.empty())
        {
            break;
        }

        // Process only frames divisible by imgModulus
        if (i % imgModulus == 0)
        {
            double time = i / fps;
            
            if (k > 0)
            {
                std::println("\n=== Frame {} (t = {:.3f}s) ===", i, time);
                cv::Mat imgout;
                // SCENARIO-SPECIFIC MEASUREMENT PROCESSING
                if (scenario == 4) {
                    // Scenario 4: Outdoor Flow Bundle + Altimeter
                    
                    // Check for altimeter update
                    bool haveFlow = true;
                    bool haveAlt = false;
                    double currentAltitude = 0.0;
                    if (i < static_cast<int>(djiVideoCaption.size()) && k > 0)
                    {
                        currentAltitude = djiVideoCaption[i].altitude;
                        if (previousAltitude < 0 || currentAltitude != previousAltitude)
                        {
                            haveAlt = true;
                        }
                    }

                    // Mark frame as a flow frame (drives cloning once in predict(dt>0))
                    system.setFlowEvent(haveFlow);

                    // Altimeter first (if present) – this will call predict(dt>0)
                    if (haveAlt)
                    {
                        std::println("Altimeter update: altitude = {:.3f}m (change: {:.3f}m)", currentAltitude, currentAltitude - previousAltitude);
                        Eigen::VectorXd y_alt(1);
                        y_alt(0) = currentAltitude - 7.0; // apply scale
                        MeasurementAltimeter measAlt(time, y_alt);
                        measAlt.process(system);
                        Eigen::VectorXd x = system.density.mean();
                        std::cout << "altimeter update done" << std::endl;
                        std::cout << "state after measurement: " << x.transpose() << std::endl;
                        previousAltitude = currentAltitude;
                    }
                    
                    // Flow Bundle
                    MeasurementOutdoorFlowBundle measFlow(time, camera, imgk_raw, imgkm1_raw, rQOikm1);
                    measFlow.process(system);
                    std::cout << "flow bundle update done" << std::endl;
                    
                    // Get tracked features for next iteration
                    rQOikm1 = measFlow.trackedPreviousFeatures();
                    const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik = measFlow.trackedCurrentFeatures();
                    
                    // Update state after measurement
                    Eigen::VectorXd x = system.density.mean();
                    Eigen::Vector6d etak = x.segment<6>(6);
                    Eigen::Vector6d zetak = x.segment<6>(12);
                    std::cout << "state after measurement: " << x.transpose() << std::endl;
                    
                    cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);
                    
                    // Get predicted flow field for visualization
                    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = measFlow.predictedFeatures(x, system);
                    
                    // Scale feature coordinates for plotting
                    std::vector<cv::Point2d> rQOikm1_scaled, rQOik_scaled, rQOik_hat_scaled;
                    int np = rQOik.cols();
                    rQOikm1_scaled.resize(np);
                    rQOik_scaled.resize(np);
                    rQOik_hat_scaled.resize(np);
                    for (int j = 0; j < np; ++j)
                    {
                        rQOikm1_scaled[j].x     = rQOikm1(0, j)/divisor;
                        rQOikm1_scaled[j].y     = rQOikm1(1, j)/divisor;

                        rQOik_scaled[j].x       = rQOik(0, j)/divisor;
                        rQOik_scaled[j].y       = rQOik(1, j)/divisor;

                        rQOik_hat_scaled[j].x   = rQOik_hat(0, j)/divisor;
                        rQOik_hat_scaled[j].y   = rQOik_hat(1, j)/divisor;
                    }

                    // Plot flow vectors
                    for (int j = 0; j < rQOik.cols(); ++j)
                    {
                        // Predicted flow (blue)
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_hat_scaled[j], 
                                       cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                        if (measFlow.inlierMask()[j])
                        {
                            // Measured inliers (green)
                            cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], 
                                           cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                        }
                        else
                        {
                            // Measured outliers (red)
                            cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], 
                                           cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                        }
                    }

                    // Plot overlays
                    plotGroundPlane(imgout, etak, camera, divisor);
                    plotHorizon(imgout, etak, camera, divisor);
                    plotCompass(imgout, etak, camera, divisor);
                    plotTransVelocity(imgout, x, etak, etakm1, camera, divisor);
                    plotStateInfo(imgout, etak, djiVideoCaption, i, scenario);
                    plotTrajectory2D(imgout, etak, djiVideoCaption, i);

                    // Display
                    cv::imshow("Visual Navigation - Scenario 4", imgout);
                    char key = cv::waitKey(1);
                    if (key == 'q')
                    {
                        std::println("Key '{}' pressed. Terminating program.", key);
                        break;
                    }

                    // Write output frame
                    if (doExport)
                    {
                        bufferedVideoWriter.write(imgout);
                    }

                    // Update for next iteration
                    rQOikm1.resize(2, rQOik.cols());
                    rQOikm1 = rQOik;
                    etakm1 = etak;
                }
                else if (scenario == 5) {
                    // Scenario 5: Indoor Flow + ArUco SLAM
                    
                    // Mark frame as a flow frame (needed for proper zeta cloning)
                    system.setFlowEvent(true);
                    
                    // 1. Detect ArUco markers first
                    ArucoDetectionResult arucoResult = detectAndDrawArUco(imgk_raw, 100);
                    cv::Mat outputFrame = arucoResult.image.clone();  // Already has ArUco drawn on it
                    
                    // 2. Indoor Flow measurement
                    MeasurementIndoorFlowBundle measFlow(time, camera, imgk_raw, imgkm1_raw, rQOikm1);
                    // measFlow.process(system);
                    std::println("Flow update complete");
                    
                    // Get tracked features for next iteration
                    rQOikm1 = measFlow.trackedPreviousFeatures();
                    const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik = measFlow.trackedCurrentFeatures();
                    
                    // Update state after flow measurement
                    Eigen::VectorXd x = system.density.mean();
                    Eigen::Vector6d etak = x.segment<6>(6);
                    
                    // Get predicted flow field for visualization
                    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = measFlow.predictedFeatures(x, system);
                    
                    // Draw flow vectors on outputFrame (full resolution)
                    for (int j = 0; j < rQOik.cols(); ++j)
                    {
                        cv::Point2f ptPrev(rQOikm1(0, j), rQOikm1(1, j));
                        cv::Point2f ptCurr(rQOik(0, j), rQOik(1, j));
                        cv::Point2f ptPred(rQOik_hat(0, j), rQOik_hat(1, j));
                        
                        // Predicted flow (blue) - thickness scaled for full resolution
                        cv::arrowedLine(outputFrame, ptPrev, ptPred, 
                                       cv::Scalar(255, 0, 0), divisor, cv::LINE_AA);
                
                        if (measFlow.inlierMask()[j])
                        {
                            // Measured inliers (green)
                            cv::arrowedLine(outputFrame, ptPrev, ptCurr, 
                                           cv::Scalar(0, 255, 0), divisor, cv::LINE_AA);
                        }
                        else
                        {
                            // Measured outliers (red)
                            cv::arrowedLine(outputFrame, ptPrev, ptCurr, 
                                           cv::Scalar(0, 0, 255), divisor, cv::LINE_AA);
                        }
                    }
                    
                    // 3. Process ArUco SLAM measurement if markers detected
                    if (!arucoResult.markerCorners.empty())
                    {
                        // Format Y matrix (8 × nDetections)
                        Eigen::Matrix<double, 8, Eigen::Dynamic> Y(8, arucoResult.markerCorners.size());
                        for (size_t j = 0; j < arucoResult.markerCorners.size(); ++j)
                        {
                            for (int c = 0; c < 4; ++c)
                            {
                                Y(2*c, j) = arucoResult.markerCorners[j][c].x;
                                Y(2*c+1, j) = arucoResult.markerCorners[j][c].y;
                            }
                        }
                
                        // Create and process identical tag measurement
                        measTags = std::make_unique<MeasurementSLAMIdenticalTagBundle>(time, Y, camera);
                        measTags->process(system);
                        
                        std::println("ArUco SLAM update complete");
                        std::println("Total landmarks: {}", system.numberLandmarks());
                        
                        // Update Plot (this composites 3D viz with camera view)
                        system.view() = outputFrame.clone();
                        plot->setData(system, *measTags);
                        plot->render();
                    }
                    
                    // Handle interactive mode
                    bool isLastFrame = (i + imgModulus >= nFrames);
                    if (interactive == 2 || (interactive == 1 && isLastFrame))
                    {
                        // Start handling plot GUI events (blocking)
                        plot->start();
                    }
                    
                    // Get the final rendered frame from Plot (has everything)
                    cv::Mat imgout = plot->getFrame();
                    
                    // Export frame if needed
                    if (doExport && !imgout.empty())
                    {
                        bufferedVideoWriter.write(imgout);
                    }
                
                    // Update for next iteration
                    rQOikm1.resize(2, rQOik.cols());
                    rQOikm1 = rQOik;
                    etakm1 = etak;
                }
            }

            // Store current frame for next iteration
            imgk_raw.copyTo(imgkm1_raw);
            k++;
        }
    }

    // Cleanup
    if (doExport)
    {
        bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
    
    std::println("\nVisual navigation complete. Processed {} keyframes.", k);
}

// [All the plotting functions remain EXACTLY the same - copy from original]

// Reuse functions from visual odometry code
Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    double h  = caption0.altitude;    // Altitude (GPS) [m]
    double ga = h - 7;                // Altitude (AGL) [m]

    Eigen::Vector6d eta0;
    eta0(0) = 0;      // North position
    eta0(1) = 0;      // East position
    eta0(2) = -ga;    // Down position (negative altitude)
    eta0(3) = 0;      // Roll
    eta0(4) = 0.04;   // Pitch
    eta0(5) = -1.05;
    
    return eta0;
}

void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    Eigen::Vector3d rBNn = etak.head<3>();
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    Pose<double> Tnb(Rnb, rBNn);
    
    const double gridSize = 2000.0;
    const double gridSpacing = 100.0;
    const int numLines = static_cast<int>(gridSize / gridSpacing) + 1;
    
    auto drawGridLine = [&](const cv::Vec3d& P1, const cv::Vec3d& P2, int numSegments = 100) {
        for (int i = 0; i < numSegments; ++i) {
            double t1 = static_cast<double>(i) / numSegments;
            double t2 = static_cast<double>(i + 1) / numSegments;
            
            cv::Vec3d point1 = P1 + t1 * (P2 - P1);
            cv::Vec3d point2 = P1 + t2 * (P2 - P1);
            
            if (camera.isWorldWithinFOV(point1, Tnb) || camera.isWorldWithinFOV(point2, Tnb)) {
                cv::Vec2d pixel1 = camera.worldToPixel(point1, Tnb);
                cv::Vec2d pixel2 = camera.worldToPixel(point2, Tnb);
                cv::line(img, cv::Point(pixel1[0]/divisor, pixel1[1]/divisor),
                        cv::Point(pixel2[0]/divisor, pixel2[1]/divisor),
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            }
        }
    };
    
    const double groundDown = 0.0;  // Ground plane at Down = 0
    
    // Draw North-South lines in FIXED world coordinates (not camera-relative)
    for (int i = 0; i < numLines; ++i) {
        double east = -gridSize/2 + i * gridSpacing;
        cv::Vec3d P1(gridSize/2, east, groundDown);   // Fixed world coordinates
        cv::Vec3d P2(-gridSize/2, east, groundDown);
        drawGridLine(P1, P2);
    }
    
    // Draw East-West lines in FIXED world coordinates (not camera-relative)
    for (int i = 0; i < numLines; ++i) {
        double north = -gridSize/2 + i * gridSpacing;
        cv::Vec3d P1(north, -gridSize/2, groundDown);  // Fixed world coordinates
        cv::Vec3d P2(north, gridSize/2, groundDown);
        drawGridLine(P1, P2);
    }
}
void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    Eigen::Vector3d rBNn = etak.head<3>();
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    Pose<double> Tnb(Rnb, rBNn);
    
    // Get camera orientation in NED frame
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    Eigen::Matrix3d Rnc = Rnb * Rbc;
    
    // e3 in NED frame (points down)
    Eigen::Vector3d e3_ned(0, 0, 1);
    
    // Transform e3 to camera frame
    // A ray in camera frame that's horizontal in NED satisfies: (Rnc * ray_c)^T * e3_ned = 0
    // Which means: ray_c^T * Rnc^T * e3_ned = 0
    Eigen::Vector3d e3_camera = Rnc.transpose() * e3_ned;
    
    // Get camera intrinsics
    Eigen::Matrix3d K;
    cv::cv2eigen(camera.cameraMatrix, K);
    
    // For a pixel p = [u, v, 1]^T, the ray in camera frame is K^-1 * p
    // Horizon condition: e3_camera^T * K^-1 * p = 0
    // Rearranging: (K^-T * e3_camera)^T * p = 0
    // This gives us the line coefficients [a, b, c] where a*u + b*v + c = 0
    Eigen::Vector3d line_coeffs = K.transpose().inverse() * e3_camera;
    
    double a = line_coeffs(0);
    double b = line_coeffs(1);
    double c = line_coeffs(2);
    
    // Find two points on the horizon line at image boundaries
    cv::Point2d p1, p2;
    
    // Use left and right edges of image
    p1.x = 0;
    p1.y = -c / b;  // Solve for v when u=0
    
    p2.x = img.cols * divisor;
    p2.y = -(a * p2.x + c) / b;  // Solve for v when u=width
    
    // Draw the horizon line
    cv::line(img, cv::Point(p1.x/divisor, p1.y/divisor),
             cv::Point(p2.x/divisor, p2.y/divisor),
             cv::Scalar(0, 0, 255), 2, cv::LINE_AA);  // Red line

}


void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor)
{
    Eigen::Vector3d rBNn = etak.head<3>();
    Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());
    Pose<double> Tnb(Rnb, rBNn);
    
    // Get rotation from NED to camera
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    Eigen::Matrix3d Rnc = Rnb * Rbc;
    Eigen::Matrix3d Rcn = Rnc.transpose();
    
    struct Direction {
        std::string label;
        Eigen::Vector3d unit;
        cv::Scalar color;
    };
    
    std::vector<Direction> directions = {
        {"N",  Eigen::Vector3d(1, 0, 0), cv::Scalar(0, 0, 255)},
        {"E",  Eigen::Vector3d(0, 1, 0), cv::Scalar(0, 255, 0)},
        {"S",  Eigen::Vector3d(-1, 0, 0), cv::Scalar(255, 0, 0)},
        {"W",  Eigen::Vector3d(0, -1, 0), cv::Scalar(255, 255, 0)},
        {"NE", Eigen::Vector3d(1, 1, 0).normalized(), cv::Scalar(0, 128, 255)},
        {"SE", Eigen::Vector3d(-1, 1, 0).normalized(), cv::Scalar(255, 128, 0)},
        {"SW", Eigen::Vector3d(-1, -1, 0).normalized(), cv::Scalar(255, 0, 128)},
        {"NW", Eigen::Vector3d(1, -1, 0).normalized(), cv::Scalar(128, 128, 255)}
    };
    
    for (const auto& dir : directions) {
        Eigen::Vector3d direction_ned(dir.unit(0), dir.unit(1), 0);
        Eigen::Vector3d direction_camera = Rcn * direction_ned;
        
        cv::Vec3d direction_cv(direction_camera(0), direction_camera(1), direction_camera(2));
        
        // Check if direction is within FOV
        if (!camera.isVectorWithinFOV(direction_cv)) continue;
        
        cv::Vec2d pixel = camera.vectorToPixel(direction_cv);
        cv::Point2d pt(pixel[0]/divisor, pixel[1]/divisor);
        
        cv::putText(img, dir.label, pt, cv::FONT_HERSHEY_SIMPLEX,
                   0.8, dir.color, 2, cv::LINE_AA);
    }
}

void plotStateInfo(cv::Mat & img, const Eigen::Vector6d & etak, 
                   const std::vector<DJIVideoCaption> & djiVideoCaption, 
                   int frameIndex, int scenario)
{
    // Text display parameters
    const int textX = 10;
    const int textStartY = 30;
    const int lineSpacing = 25;
    const double fontScale = 0.6;
    const int thickness = 2;
    const cv::Scalar blueColor(255, 0, 0);   // BGR: Blue
    const cv::Scalar greenColor(0, 255, 0);  // BGR: Green
    
    int textY = textStartY;
    
    // Extract estimated position (NED coordinates)
    double estNorth = etak(0);      // North position [m]
    double estEast = etak(1);       // East position [m]
    double estDown = etak(2);       // Down position [m]
    double estAltitude = -estDown;  // Altitude (negative of Down) [m]
    
    // Display estimated North position (blue)
    std::string northText = std::format("Est North: {:.2f} m", estNorth);
    cv::putText(img, northText, cv::Point(textX, textY), 
               cv::FONT_HERSHEY_SIMPLEX, fontScale, blueColor, thickness, cv::LINE_AA);
    
    // Display estimated East position (blue)
    textY += lineSpacing;
    std::string eastText = std::format("Est East: {:.2f} m", estEast);
    cv::putText(img, eastText, cv::Point(textX, textY), 
               cv::FONT_HERSHEY_SIMPLEX, fontScale, blueColor, thickness, cv::LINE_AA);
    
    // Display estimated altitude (blue)
    textY += lineSpacing;
    std::string estAltText = std::format("Est Alt: {:.2f} m", estAltitude);
    cv::putText(img, estAltText, cv::Point(textX, textY), 
               cv::FONT_HERSHEY_SIMPLEX, fontScale, blueColor, thickness, cv::LINE_AA);
    
    // Display measured altitude from GPS (green) if available
    if (frameIndex < static_cast<int>(djiVideoCaption.size()))
    {
        textY += lineSpacing;
        double measuredAltitude = djiVideoCaption[frameIndex].altitude - 7.0;  // Apply same scale as altimeter
        std::string measAltText = std::format("Meas Alt: {:.2f} m", measuredAltitude);
        cv::putText(img, measAltText, cv::Point(textX, textY), 
                   cv::FONT_HERSHEY_SIMPLEX, fontScale, greenColor, thickness, cv::LINE_AA);
    }
}

void plotTransVelocity(cv::Mat & img, const Eigen::VectorXd & x, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor)
{
    // plot secant approximation of translational velocity (Epipole)
    Eigen::Vector3d rBNn_k = etak.head<3>();
    Eigen::Vector3d rBNn_km1 = etakm1.head<3>();
    Eigen::Matrix3d Rnb_k = rpy2rot(etak.tail<3>());
    
    Eigen::Vector3d translation_ned = rBNn_k - rBNn_km1;
    
    if (translation_ned.norm() < 1e-6) return;
    
    Eigen::Matrix3d Rbc = camera.Tbc.rotationMatrix;
    Eigen::Matrix3d Rnc_k = Rnb_k * Rbc;
    Eigen::Matrix3d Rcn_k = Rnc_k.transpose();
    
    Eigen::Vector3d translation_camera = Rcn_k * translation_ned;
    
    cv::Vec3d translation_cv(translation_camera(0), translation_camera(1), translation_camera(2));
    
    // Check if epipole direction is within FOV
    if (!camera.isVectorWithinFOV(translation_cv)) return;
    
    cv::Vec2d pixel = camera.vectorToPixel(translation_cv);
    cv::Point2d epipole(pixel[0]/divisor, pixel[1]/divisor);
    
    cv::circle(img, epipole, 10, cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    cv::circle(img, epipole, 3, cv::Scalar(0, 165, 255), -1, cv::LINE_AA);
    cv::line(img, epipole + cv::Point2d(-15, 0), epipole + cv::Point2d(15, 0),
            cv::Scalar(0, 165, 255), 2, cv::LINE_AA);
    cv::line(img, epipole + cv::Point2d(0, -15), epipole + cv::Point2d(0, 15),
            cv::Scalar(0, 165, 255), 2, cv::LINE_AA);

    // plot translational velocity
    Eigen::Vector3d vBNb = x.segment<3>(0);  // Translational velocity in body frame
    
    // Transform velocity from body frame to NED frame
    Eigen::Vector3d vBNn = Rnb_k * vBNb;
    
    // Transform velocity from NED to camera frame
    Eigen::Vector3d vCNc = Rcn_k * vBNn;
    
    cv::Vec3d velocity_cv(vCNc(0), vCNc(1), vCNc(2));
    
    // Check if velocity direction is within FOV
    if (camera.isVectorWithinFOV(velocity_cv)) {
        cv::Vec2d pixel_vel = camera.vectorToPixel(velocity_cv);
        cv::Point2d velocity_point(pixel_vel[0]/divisor, pixel_vel[1]/divisor);
        
        // Draw velocity vector in GREEN
        cv::circle(img, velocity_point, 10, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::circle(img, velocity_point, 3, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
        cv::line(img, velocity_point + cv::Point2d(-15, 0), velocity_point + cv::Point2d(15, 0),
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::line(img, velocity_point + cv::Point2d(0, -15), velocity_point + cv::Point2d(0, 15),
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        
        // Label
        cv::putText(img, "Velocity (State)", velocity_point + cv::Point2d(15, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
}

static Eigen::Vector2d gpsToNED(double lat, double lon, double alt)
{
    if (!gpsOriginSet) return Eigen::Vector2d(0, 0);
    
    double lat0 = gpsOrigin(0) * M_PI / 180.0;
    double lon0 = gpsOrigin(1) * M_PI / 180.0;
    double lat_rad = lat * M_PI / 180.0;
    double lon_rad = lon * M_PI / 180.0;
    
    double north = (lat_rad - lat0) * 6378137.0;
    double east = (lon_rad - lon0) * 6378137.0 * std::cos(lat0);
    
    return Eigen::Vector2d(north, east);
}
// Add this function with your other plot functions
void plotTrajectory2D(cv::Mat & img, const Eigen::Vector6d & etak, 
                      const std::vector<DJIVideoCaption> & djiVideoCaption, 
                      int frameIndex)
{
    // Initialize trajectory image if needed
    if (trajImg.empty())
    {
        trajImg = cv::Mat(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    }
    
    // Set GPS origin from first frame
    if (!gpsOriginSet && !djiVideoCaption.empty())
    {
        gpsOrigin(0) = djiVideoCaption[0].latitude;
        gpsOrigin(1) = djiVideoCaption[0].longitude;
        gpsOriginSet = true;
    }
    
    // Add current estimated position
    estimatedTraj.push_back(Eigen::Vector2d(etak(0), etak(1)));
    
    // Add current GPS position
    if (frameIndex < static_cast<int>(djiVideoCaption.size()))
    {
        Eigen::Vector2d gpsNED = gpsToNED(djiVideoCaption[frameIndex].latitude,
                                          djiVideoCaption[frameIndex].longitude,
                                          djiVideoCaption[frameIndex].altitude);
        gpsTraj.push_back(gpsNED);
    }
    
    // Clear image
    trajImg.setTo(cv::Scalar(255, 255, 255));
    
    // FIXED scale - centered at origin
    double centerN = 0.0;
    double centerE = 0.0;
    double scale = 3.0;  // meters per pixel (increase to zoom out)
    
    // Convert NED to pixel
    auto nedToPixel = [&](double north, double east) -> cv::Point {
        int x = trajImg.cols/2 + (east - centerE) / scale;
        int y = trajImg.rows/2 - (north - centerN) / scale;
        return cv::Point(x, y);
    };
    
    // Draw grid
    cv::Scalar gridColor(230, 230, 230);
    for (int i = -2000; i <= 2000; i += 100)
    {
        cv::line(trajImg, nedToPixel(i, -2000), nedToPixel(i, 2000), gridColor, 1);
        cv::line(trajImg, nedToPixel(-2000, i), nedToPixel(2000, i), gridColor, 1);
    }
    
    // Draw axes
    cv::line(trajImg, nedToPixel(0, -2000), nedToPixel(0, 2000), cv::Scalar(180, 180, 180), 2);
    cv::line(trajImg, nedToPixel(-2000, 0), nedToPixel(2000, 0), cv::Scalar(180, 180, 180), 2);
    
    // Draw GPS trajectory (green)
    for (size_t i = 1; i < gpsTraj.size(); ++i)
    {
        cv::Point p1 = nedToPixel(gpsTraj[i-1](0), gpsTraj[i-1](1));
        cv::Point p2 = nedToPixel(gpsTraj[i](0), gpsTraj[i](1));
        cv::line(trajImg, p1, p2, cv::Scalar(0, 200, 0), 2, cv::LINE_AA);
    }
    
    // Draw estimated trajectory (blue)
    for (size_t i = 1; i < estimatedTraj.size(); ++i)
    {
        cv::Point p1 = nedToPixel(estimatedTraj[i-1](0), estimatedTraj[i-1](1));
        cv::Point p2 = nedToPixel(estimatedTraj[i](0), estimatedTraj[i](1));
        cv::line(trajImg, p1, p2, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }
    
    // Draw current positions
    if (!gpsTraj.empty())
    {
        cv::Point gp = nedToPixel(gpsTraj.back()(0), gpsTraj.back()(1));
        cv::circle(trajImg, gp, 5, cv::Scalar(0, 200, 0), -1);
    }
    if (!estimatedTraj.empty())
    {
        cv::Point ep = nedToPixel(estimatedTraj.back()(0), estimatedTraj.back()(1));
        cv::circle(trajImg, ep, 5, cv::Scalar(255, 0, 0), -1);
    }
    
    // Draw legend
    cv::putText(trajImg, "GPS", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 200, 0), 2);
    cv::putText(trajImg, "Est", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    
    // Show
    cv::imshow("Trajectory 2D", trajImg);
}

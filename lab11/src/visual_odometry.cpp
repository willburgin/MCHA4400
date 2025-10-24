#include <print>
#include <numbers>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include "to_string.hpp"
#include "BufferedVideo.h"
#include "Pose.hpp"
#include "rotation.hpp"
#include "Camera.h"
#include "DJIVideoCaption.h"
#include "rotation.hpp"
#include "GaussianInfo.hpp"
#include "funcmin.hpp"
#include "SystemVisualNav.h"
#include "MeasurementOutdoorFlowBundle.h"
#include "visual_odometry.h"

// Forward declarations
static Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0);
static void plotGroundPlane(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotHorizon(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotCompass(cv::Mat & img, const Eigen::Vector6d & etak, const Camera & camera, const int & divisor);
static void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor);


void runVisualOdometryFromVideo(const std::filesystem::path & videoPath, const std::filesystem::path & cameraPath, const std::filesystem::path & outputDirectory)
{
    // TODO: Lab 11
    int imgModulus  = 10;                    // Take frames divisible by this number
    int divisor     = 2;                    // Image scaling factor (used for plotting only)

    assert(!videoPath.empty());

    // Subtitle path
    std::filesystem::path subtitlePath = videoPath.parent_path() / (videoPath.stem().string() + ".SRT");
    assert(std::filesystem::exists(subtitlePath));

    // Load and parse subtitle file
    std::vector<DJIVideoCaption> djiVideoCaption = getVideoCaptions(subtitlePath);

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
    // TODO: Lab 11
    Eigen::Matrix3d Rbc;
    Rbc <<  0, 0, 1,     // b1 = c3
            1, 0, 0,   // b2 = c1
            0, 1, 0;   // b3 = c2

    camera.Tbc.rotationMatrix = Rbc;
    camera.Tbc.translationVector = Eigen::Vector3d::Zero();

    // Open input video
    cv::VideoCapture cap(videoPath.string());
    assert(cap.isOpened());
    int nFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    assert(nFrames > 0);


    std::println("Input video: {}", videoPath.string());
    std::println("Subtitle file: {}", subtitlePath.string());
    std::println("Total number of frames: {}", nFrames);
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::println("Input video frame rate: {}", fps);
    std::println("Input video dimensions: [{} x {}]",
                cap.get(cv::CAP_PROP_FRAME_WIDTH),
                cap.get(cv::CAP_PROP_FRAME_HEIGHT));

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
        // int codec = cap.get(cv::CAP_PROP_FOURCC); // use same output video codec as input video
        int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // manually specify output video codec
        videoOut.open(outputPath.string(), codec, outputFps, frameSize);
        bufferedVideoWriter.start(videoOut);
    }

    // Visual odometry
    auto p0 = GaussianInfo<double>::fromSqrtInfo(Eigen::VectorXd::Zero(18), Eigen::MatrixXd::Zero(18, 18)); // dummy initial distribution
    SystemVisualNav system(p0); // dummy system
    Eigen::VectorXd etakm1(6);
    Eigen::VectorXd etak(6);
    etak = getInitialPose(djiVideoCaption[0]);

    cv::Mat imgk_raw;
    cv::Mat imgkm1_raw;
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1;

    for (int i = 0, k = 0;; ++i)
    {
        // Capture frame by frame
        imgk_raw = bufferedVideoReader.read();
        if (imgk_raw.empty())
        {
            break;
        }

        if (i % imgModulus == 0)
        {
            if (k > 0)
            {
                // Create measurement data from image pair and previous tracked features
                MeasurementOutdoorFlowBundle measurement(i/fps, camera, imgk_raw, imgkm1_raw, rQOikm1);

                rQOikm1 = measurement.trackedPreviousFeatures();
                const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOik = measurement.trackedCurrentFeatures();

                // Create cost function with prototype V = costFunc(eta, g, H)
                auto costFunc = [&](const Eigen::VectorXd & etak, Eigen::VectorXd & g, Eigen::MatrixXd & H)
                {
                    return measurement.costOdometry(etak, etakm1, g, H);
                };

                // Minimise cost (maximise log likelihood)
                etak = etakm1; // Start optimisation at initial conditions
                const int verbosity = 3; // 0:none, 1:dots, 2:summary, 3:iter
                int ret = funcmin::NewtonTrust(costFunc, etak, verbosity);
                assert(ret == 0);

                Eigen::Vector3d rBNn = etak.head<3>();
                Eigen::Matrix3d Rnb = rpy2rot(etak.tail<3>());

                std::print("rBNn: \n{}\n", to_string(rBNn));
                std::print("Rnb: \n{}\n", to_string(Rnb));

                Eigen::VectorXd x(18);
                x.setZero();
                x.segment<6>(6) = etak;
                x.segment<6>(12) = etakm1;

                // Prepare output image
                cv::Mat imgout;
                cv::resize(imgk_raw, imgout, cv::Size(), 1.0/divisor, 1.0/divisor);

                // Predicted flow field (used for plotting onto original image)
                Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = measurement.predictedFeatures(x, system);

                // Plotting
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
                    // Predicted flow
                    cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_hat_scaled[j], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

                    if (measurement.inlierMask()[j])
                    {
                        // Measured inliers
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                    }
                    else
                    {
                        // Measured outliers
                        cv::arrowedLine(imgout, rQOikm1_scaled[j], rQOik_scaled[j], cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
                    }
                }

                plotGroundPlane(imgout, etak, camera, divisor);
                plotHorizon(imgout, etak, camera, divisor);
                plotCompass(imgout, etak, camera, divisor);
                plotEpipole(imgout, etak, etakm1, camera, divisor);

                cv::imshow("Visual odometry demo", imgout);
                char key = cv::waitKey(1);
                if (key == 'q')
                {
                    std::println("Key '{}' pressed. Terminating program.", key);
                    break;
                }

                if (doExport)
                {
                    bufferedVideoWriter.write(imgout);
                }

                rQOikm1.resize(2, rQOik.cols());
                rQOikm1 = rQOik;
            }

            imgk_raw.copyTo(imgkm1_raw);
            etakm1 = etak;
            k++;
        }
    } 

    if (doExport)
    {
         bufferedVideoWriter.stop();
    }
    bufferedVideoReader.stop();
}

Eigen::Vector6d getInitialPose(const DJIVideoCaption & caption0)
{
    double h        = caption0.altitude;    // Altitude (GPS) [m]
    double ga       = h - 7;                // Altitude (AGL) [m]

    Eigen::Vector6d eta0;
    // TODO: Lab 11
    eta0(0) = 0;
    eta0(1) = 0;
    eta0(2) = -ga;
    eta0(3) = 0;
    eta0(4) = 0.09;
    eta0(5) = 0;

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
    
    // Debug output
    std::cout << "Horizon line: " << a << "*u + " << b << "*v + " << c << " = 0" << std::endl;
    std::cout << "Horizon at v = " << -c/b << " (center of image)" << std::endl;
    std::cout << "Roll: " << etak(3)*180/M_PI << "°, "
              << "Pitch: " << etak(4)*180/M_PI << "°, "
              << "Yaw: " << etak(5)*180/M_PI << "°" << std::endl;
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

void plotEpipole(cv::Mat & img, const Eigen::Vector6d & etak, const Eigen::Vector6d & etakm1, const Camera & camera, const int & divisor)
{
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
}


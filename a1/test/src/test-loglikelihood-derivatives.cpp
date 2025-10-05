#include <doctest/doctest.h>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "../../src/Camera.h"
#include "../../src/MeasurementSLAMUniqueTagBundle.h"
#include "../../src/SystemSLAMPoseLandmarks.h"
#include "../../src/rotation.hpp"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) \
    INFO(#x " =\n" << x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("MeasurementSLAMUniqueTagBundle::logLikelihood analytical gradient vs autodiff")
{
    GIVEN("A SLAM system with camera and ArUco landmark")
    {
        // Set up camera
        Camera cam;
        cam.cameraMatrix = (cv::Mat_<double>(3,3) <<
            800.0, 0.0, 320.0,
            0.0, 800.0, 240.0,
            0.0, 0.0, 1.0);
        cam.distCoeffs = cv::Mat::zeros(12, 1, CV_64F); // no distortion
        cam.flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;
        cam.imageSize = cv::Size(640, 480);
        
        // Set camera-to-body transformation
        cam.Tbc.rotationMatrix = Eigen::Matrix3d::Identity();
        cam.Tbc.translationVector = Eigen::Vector3d::Zero();

        // Create state vector: [vBNb(3), omegaBNb(3), rBNn(3), Thetanb(3), rJNn(3), Thetanj(3)]
        // State size: 12 (body) + 6 (one pose landmark: position + orientation) = 18
        Eigen::VectorXd x(18);
        x << 0.1, 0.2, 0.3,        // vBNb - body velocity
             0.05, 0.1, 0.15,      // omegaBNb - body angular velocity  
             1.0, 2.0, 3.0,        // rBNn - body position
             0.1, 0.2, 0.3,        // Thetanb - body orientation (roll, pitch, yaw)
             5.0, 6.0, 7.0,        // rJNn - landmark position
             0.0, 0.0, 0.0;        // Thetanj - landmark orientation

        // Create system density
        auto density = GaussianInfo<double>::fromSqrtMoment(Eigen::MatrixXd::Identity(18, 18));
        SystemSLAMPoseLandmarks system(density);
        
        // Add a known marker ID
        system.addKnownMarkerID(0);
        
        // Create measurement with detected ArUco corners
        // Simulate 4 corners of an ArUco marker in the image
        Eigen::Matrix<double, 8, Eigen::Dynamic> Y(8, 1);
        Y << 300.0, 310.0,  // Corner 0: (x, y)
             350.0, 305.0,  // Corner 1: (x, y)
             355.0, 355.0,  // Corner 2: (x, y)
             305.0, 360.0;  // Corner 3: (x, y)
        
        MeasurementSLAMUniqueTagBundle measurement(0.0, Y, cam);
        
        // Set marker IDs for the detection
        std::vector<int> markerIDs = {0};
        measurement.setFrameMarkerIDs(markerIDs);
        
        // Associate the detection with the landmark
        std::vector<std::size_t> idxLandmarks = {0};
        measurement.associate(system, idxLandmarks);

        // (1) Analytical gradient from your implementation
        Eigen::VectorXd g_an;
        double logLik_an = measurement.logLikelihood(x, system, g_an);

        // (2) Autodiff oracle using templated version
        using autodiff::dual;
        Eigen::VectorX<dual> x_dual = x.cast<dual>();

        // Compute gradient using autodiff
        autodiff::dual logLik_dual;
        Eigen::VectorXd g_exp = gradient(
            [&](const Eigen::VectorX<dual>& x_var) -> dual {
                return measurement.logLikelihoodTemplated<dual>(x_var, system);
            },
            autodiff::wrt(x_dual),
            autodiff::at(x_dual),
            logLik_dual
        );

        THEN("Analytical gradient matches autodiff within tolerance")
        {
            CAPTURE(logLik_an);
            CAPTURE(val(logLik_dual));
            CAPTURE_EIGEN(g_an);
            CAPTURE_EIGEN(g_exp);
            
            CHECK(std::abs(logLik_an - val(logLik_dual)) < 1e-6);
            CHECK(g_an.isApprox(g_exp, 1e-6));
        }
    }
}

SCENARIO("MeasurementSLAMUniqueTagBundle::logLikelihood analytical Hessian vs autodiff")
{
    GIVEN("A SLAM system with camera and ArUco landmark")
    {
        // Set up camera
        Camera cam;
        cam.cameraMatrix = (cv::Mat_<double>(3,3) <<
            800.0, 0.0, 320.0,
            0.0, 800.0, 240.0,
            0.0, 0.0, 1.0);
        cam.distCoeffs = cv::Mat::zeros(12, 1, CV_64F);
        cam.flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;
        cam.imageSize = cv::Size(640, 480);
        
        cam.Tbc.rotationMatrix = Eigen::Matrix3d::Identity();
        cam.Tbc.translationVector = Eigen::Vector3d::Zero();

        // Create state vector
        Eigen::VectorXd x(18);
        x << 0.1, 0.2, 0.3,
             0.05, 0.1, 0.15,
             1.0, 2.0, 3.0,
             0.1, 0.2, 0.3,
             5.0, 6.0, 7.0,
             0.0, 0.0, 0.0;

        auto density = GaussianInfo<double>::fromSqrtMoment(Eigen::MatrixXd::Identity(18, 18));
        SystemSLAMPoseLandmarks system(density);
        system.addKnownMarkerID(0);
        
        // Create measurement
        Eigen::Matrix<double, 8, Eigen::Dynamic> Y(8, 1);
        Y << 300.0, 310.0,
             350.0, 305.0,
             355.0, 355.0,
             305.0, 360.0;
        
        MeasurementSLAMUniqueTagBundle measurement(0.0, Y, cam);
        std::vector<int> markerIDs = {0};
        measurement.setFrameMarkerIDs(markerIDs);
        
        std::vector<std::size_t> idxLandmarks = {0};
        measurement.associate(system, idxLandmarks);

        // (1) Analytical Hessian from your implementation
        Eigen::VectorXd g_an;
        Eigen::MatrixXd H_an;
        double logLik_an = measurement.logLikelihood(x, system, g_an, H_an);

        // (2) Autodiff oracle for Hessian
        using autodiff::dual2nd;
        Eigen::VectorX<dual2nd> x_dual2 = x.cast<dual2nd>();

        autodiff::dual2nd logLik_dual2;
        Eigen::VectorXd g_exp;
        Eigen::MatrixXd H_exp = hessian(
            [&](const Eigen::VectorX<dual2nd>& x_var) -> dual2nd {
                return measurement.logLikelihoodTemplated<dual2nd>(x_var, system);
            },
            autodiff::wrt(x_dual2),
            autodiff::at(x_dual2),
            logLik_dual2,
            g_exp
        );

        THEN("Analytical Hessian matches autodiff within tolerance")
        {
            CAPTURE(logLik_an);
            CAPTURE(val(logLik_dual2));
            CAPTURE_EIGEN(g_an);
            CAPTURE_EIGEN(g_exp);
            CAPTURE_EIGEN(H_an);
            CAPTURE_EIGEN(H_exp);
            
            CHECK(std::abs(logLik_an - val(logLik_dual2)) < 1e-6);
            CHECK(g_an.isApprox(g_exp, 1e-6));
            CHECK(H_an.isApprox(H_exp, 1e-5));
        }
    }
}

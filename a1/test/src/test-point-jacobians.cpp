#include <doctest/doctest.h>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "../../src/Camera.h"
#include "../../src/MeasurementSLAMPointBundle.h"
#include "../../src/SystemSLAMPointLandmarks.h"
#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) \
    INFO(#x " =\n" << x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("Camera::vectorToPixel analytical Jacobian vs autodiff")
{
    GIVEN("A nominal calibrated camera and a 3D point")
    {
        Camera cam;

        // Initialise a simple camera matrix for testing
        cam.cameraMatrix = (cv::Mat_<double>(3,3) <<
            800.0, 0.0, 320.0,
            0.0, 800.0, 240.0,
            0.0, 0.0, 1.0);
        cam.distCoeffs = cv::Mat::zeros(12, 1, CV_64F); // no distortion
        cam.flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;

        Eigen::Vector3d rPCc(0.2, -0.1, 1.2);

        // (1) Analytical Jacobian from your implementation
        Eigen::Matrix<double,2,3> J_an;
        Eigen::Vector2d uv_an = cam.vectorToPixel(rPCc, J_an);

        // (2) Autodiff oracle
        using autodiff::dual;
        Eigen::Matrix<dual,3,1> xdual = rPCc.cast<dual>();

        // Evaluate using your templated version in Camera.h
        auto uv_dual = cam.vectorToPixel<dual>(xdual);

        // Gradient row-by-row
        Eigen::Matrix<double,2,3> J_exp;
        for(int i = 0; i < 2; ++i) {
            auto fi = [&](const Eigen::Matrix<dual,3,1>& x) -> dual {
                return cam.vectorToPixel<dual>(x)(i);
            };
            autodiff::dual f0;
            Eigen::VectorXd gi = gradient(fi, autodiff::wrt(xdual), autodiff::at(xdual), f0);
            J_exp.row(i) = gi.transpose();
        }

        THEN("Analytical matches autodiff within tolerance")
        {
            CAPTURE_EIGEN(J_an);
            CAPTURE_EIGEN(J_exp);
            CHECK(J_an.isApprox(J_exp, 1e-6));
        }
    }
}

SCENARIO("MeasurementPointBundle::predictFeature analytical Jacobian vs autodiff")
{
    GIVEN("A SLAM system with camera and landmark")
    {
        // Set up camera
        Camera cam;
        cam.cameraMatrix = (cv::Mat_<double>(3,3) <<
            800.0, 0.0, 320.0,
            0.0, 800.0, 240.0,
            0.0, 0.0, 1.0);
        cam.distCoeffs = cv::Mat::zeros(12, 1, CV_64F); // no distortion
        cam.flags = cv::CALIB_RATIONAL_MODEL | cv::CALIB_THIN_PRISM_MODEL;
        
        // Set camera-to-body transformation (identity for simplicity)
        cam.Tbc.rotationMatrix = Eigen::Matrix3d::Identity();
        cam.Tbc.translationVector = Eigen::Vector3d::Zero();

        // Create state vector: [vBNb(3), omegaBNb(3), rBNn(3), Thetanb(3), rL1Nn(3)]
        // State size: 12 (body) + 3 (one landmark) = 15
        Eigen::VectorXd x(15);
        x << 0.1, 0.2, 0.3,        // vBNb - body velocity
             0.05, 0.1, 0.15,      // omegaBNb - body angular velocity  
             1.0, 2.0, 3.0,        // rBNn - body position
             0.1, 0.2, 0.3,        // Thetanb - body orientation (roll, pitch, yaw)
             5.0, 6.0, 7.0;        // rL1Nn - landmark position

        // Create system density (mock - just need dimensions)
        auto density = GaussianInfo<double>::fromSqrtMoment(Eigen::MatrixXd::Identity(15, 15));
        SystemSLAMPointLandmarks system(density);
        
        // Create measurement
        Eigen::Matrix<double, 2, Eigen::Dynamic> Y_dummy(2, 1);
        Y_dummy.col(0) << 400.0, 300.0;  // dummy feature observation
        MeasurementPointBundle measurement(0.0, Y_dummy, cam);
        
        std::size_t idxLandmark = 0;  // Test first (and only) landmark

        // (1) Analytical Jacobian from predictFeature implementation
        Eigen::MatrixXd J_an;
        Eigen::Vector2d uv_an = measurement.predictFeature(x, J_an, system, idxLandmark);

        // (2) Autodiff oracle using templated version
        using autodiff::dual;
        Eigen::VectorX<dual> x_dual = x.cast<dual>();

        // Create lambda for each component of predictFeature output
        auto predictFeature_i = [&](const Eigen::VectorX<dual>& x_var, int i) -> dual {
            return measurement.predictFeature(x_var, system, idxLandmark)(i);
        };

        // Compute gradient for each output component
        Eigen::MatrixXd J_exp(2, x.size());
        for(int i = 0; i < 2; ++i) {
            auto fi = [&](const Eigen::VectorX<dual>& x_var) -> dual {
                return predictFeature_i(x_var, i);
            };
            autodiff::dual f0;
            Eigen::VectorXd gi = gradient(fi, autodiff::wrt(x_dual), autodiff::at(x_dual), f0);
            J_exp.row(i) = gi.transpose();
        }

        THEN("Analytical matches autodiff within tolerance")
        {
            CAPTURE_EIGEN(J_an);
            CAPTURE_EIGEN(J_exp);
            CHECK(J_an.isApprox(J_exp, 1e-6));
        }
    }
}

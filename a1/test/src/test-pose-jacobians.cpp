#include <doctest/doctest.h>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "../../src/Camera.h"
#include "../../src/MeasurementSLAMUniqueTagBundle.h"
#include "../../src/SystemSLAMPoseLandmarks.h"
#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) \
    INFO(#x " =\n" << x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("MeasurementSLAMUniqueTagBundle::predictFeature analytical Jacobian vs autodiff")
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

        // Create state vector: [vBNb(3), omegaBNb(3), rBNn(3), Thetanb(3), rL1Nn(3), Thetanj(3)]
        // State size: 12 (body) + 6 (one pose landmark: 3 position + 3 orientation) = 18
        Eigen::VectorXd x(18);
        x << 0.1, 0.2, 0.3,        // vBNb - body velocity
             0.05, 0.1, 0.15,      // omegaBNb - body angular velocity  
             1.0, 2.0, 3.0,        // rBNn - body position
             0.1, 0.2, 0.3,        // Thetanb - body orientation (roll, pitch, yaw)
             5.0, 6.0, 7.0,        // rL1Nn - landmark position
             0.1, 0.2, 0.3;        // Thetanj - landmark orientation (roll, pitch, yaw)

        // Create system density (mock - just need dimensions)
        auto density = GaussianInfo<double>::fromSqrtMoment(Eigen::MatrixXd::Identity(18, 18));
        SystemSLAMPoseLandmarks system(density);
        
        // Create measurement (8 measurements: 4 corners × 2 coordinates each)
        Eigen::Matrix<double, 8, Eigen::Dynamic> Y_dummy(8, 1);  // 8 rows, 1 col
        Y_dummy << 400.0,  // u coordinate for corner 1
                   300.0,  // v coordinate for corner 1
                   500.0,  // u coordinate for corner 2
                   400.0,  // v coordinate for corner 2
                   400.0,  // u coordinate for corner 3
                   300.0,  // v coordinate for corner 3
                   500.0,  // u coordinate for corner 4
                   400.0;  // v coordinate for corner 4
        MeasurementSLAMUniqueTagBundle measurement(0.0, Y_dummy, cam);
        
        std::size_t idxLandmark = 0;  // Test first (and only) landmark

        // (1) Analytical Jacobian from predictFeature implementation
        Eigen::MatrixXd J_an;
        Eigen::Matrix<double, 8, 1> uv_an = measurement.predictFeature(x, J_an, system, idxLandmark);

        // (2) Autodiff oracle using templated version
        using autodiff::dual;
        Eigen::VectorX<dual> x_dual = x.cast<dual>();

        // Create lambda for each component of predictFeature output
        auto predictFeature_i = [&](const Eigen::VectorX<dual>& x_var, int i) -> dual {
            return measurement.predictFeature(x_var, system, idxLandmark)(i);
        };

        // Compute gradient for each output component
        Eigen::MatrixXd J_exp(8, x.size());
        for(int i = 0; i < 8; ++i) {
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

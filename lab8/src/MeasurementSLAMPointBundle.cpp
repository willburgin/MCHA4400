#include <cstddef>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemSLAM.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMPointBundle.h"
#include "rotation.hpp"

MeasurementPointBundle::MeasurementPointBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(1.0) // TODO: Assignment(s)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM * MeasurementPointBundle::clone() const
{
    return new MeasurementPointBundle(*this);
}

Eigen::VectorXd MeasurementPointBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);

    // Select visible landmarks
    std::vector<std::size_t> idxLandmarks(systemSLAM.numberLandmarks());
    std::iota(idxLandmarks.begin(), idxLandmarks.end(), 0); // FIXME: This just selects all landmarks
    // TODO: Assignment(s)
    return 0.0;
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    g.resize(x.size());
    g.setZero();
    // TODO: Assignment(s)
    return logLikelihood(x, system);
}

double MeasurementPointBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    H.setZero();
    // TODO: Assignment(s)
    return logLikelihood(x, system, g);
}

void MeasurementPointBundle::update(SystemBase & system)
{
    SystemSLAM & systemSLAM = dynamic_cast<SystemSLAM &>(system);

    // TODO: Assignment(s)
    // Identify landmarks with matching features (data association)
    // Remove failed landmarks from map (consecutive failures to match)
    // Identify surplus features that do not correspond to landmarks in the map
    // Initialise up to Nmax – N new landmarks from best surplus features
    
    Measurement::update(system);    // Do the actual measurement update
}

// Image feature location for a given landmark and Jacobian
Eigen::Vector2d MeasurementPointBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Get camera pose from state
    Pose<double> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // Get landmark position from state
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3d rPNn = x.segment<3>(idx);

    // Transform to camera coordinates
    Eigen::Vector3d rPCc = Tnc.rotationMatrix.transpose() * (rPNn - Tnc.translationVector);

    // Get pixel coordinates with Jacobian w.r.t. camera coordinates
    Eigen::Matrix23d J_camera;  // ← Fixed: Different name from parameter
    Eigen::Vector2d rQOi = camera_.vectorToPixel(rPCc, J_camera);

    // Compute full Jacobian ∂h/∂x using chain rule and Appendix B expressions
    J.resize(2, x.size());  // ← Fixed: Use parameter name J, not h
    J.setZero();

    // Extract state components
    Eigen::Vector3d rBNn = x.segment<3>(6);
    Eigen::Vector3d Thetanb = x.segment<3>(9);

    // Get body pose
    Pose<double> Tnb;
    Tnb.rotationMatrix = rpy2rot(Thetanb);
    Tnb.translationVector = rBNn;

    // Get camera-to-body transformation
    Pose<double> Tbc = camera_.Tbc;
    Eigen::Matrix3d Rnb = Tnb.rotationMatrix;
    Eigen::Matrix3d Rbc = Tbc.rotationMatrix;

    // Equation (27a): ∂h_j/∂r^n_{j/N} 
    Eigen::Matrix<double, 2, 3> dhj_drPNn = J_camera * Rbc.transpose() * Rnb.transpose();
    J.block<2, 3>(0, idx) = dhj_drPNn;

    // Equation (27b): ∂h_j/∂r^n_{B/N}
    Eigen::Matrix<double, 2, 3> dhj_drBNn = -J_camera * Rbc.transpose() * Rnb.transpose();
    J.block<2, 3>(0, 6) = dhj_drBNn;

    Eigen::Vector3d rPNn_minus_rBNn = rPNn - rBNn;

    // ∂R^n_b/∂φ (roll) - Fixed: These should be 3×3 matrices
    Eigen::Matrix3d dRx_dphi;
    rotx(Thetanb(0), dRx_dphi);
    Eigen::Matrix3d dRnb_dphi = rotz(Thetanb(2)) * roty(Thetanb(1)) * dRx_dphi;

    // ∂R^n_b/∂θ (pitch)  
    Eigen::Matrix3d dRy_dtheta;
    roty(Thetanb(1), dRy_dtheta);
    Eigen::Matrix3d dRnb_dtheta = rotz(Thetanb(2)) * dRy_dtheta * rotx(Thetanb(0));

    // ∂R^n_b/∂ψ (yaw)
    Eigen::Matrix3d dRz_dpsi;
    rotz(Thetanb(2), dRz_dpsi);
    Eigen::Matrix3d dRnb_dpsi = dRz_dpsi * roty(Thetanb(1)) * rotx(Thetanb(0));

    // // // Apply equation (27c) for each Euler angle
    J.col(9)  = J_camera * Rbc.transpose() * dRnb_dphi.transpose() * rPNn_minus_rBNn;     // ∂h_j/∂φ
    J.col(10) = J_camera * Rbc.transpose() * dRnb_dtheta.transpose() * rPNn_minus_rBNn;  // ∂h_j/∂θ  
    J.col(11) = J_camera * Rbc.transpose() * dRnb_dpsi.transpose() * rPNn_minus_rBNn;    // ∂h_j/∂ψ
    // std::cout << "J: " << J << std::endl;

    return rQOi;
}

// Density of image feature location for a given landmark
GaussianInfo<double> MeasurementPointBundle::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2;

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeature(x, J, system, idxLandmark) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };
    
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd MeasurementPointBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = system.density.dim();
    assert(x.size() == nx);

    Eigen::VectorXd h(2*nL);
    J.resize(2*nL, nx);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::MatrixXd Jfeature;
        Eigen::Vector2d rQOi = predictFeature(x, Jfeature, system, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 9
        // Set pair of rows of J
        // TODO: Lab 9
    }
    return h;
}

// Density of image features for a set of landmarks
GaussianInfo<double> MeasurementPointBundle::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 2*idxLandmarks.size();

    //   y   =   h(x) + v  
    // \___/   \__________/
    //   ya  =   ha(x, v)
    //
    // Helper function to evaluate ha(x, v) and its Jacobian Ja = [dha/dx, dha/dv]
    const auto func = [&](const Eigen::VectorXd & xv, Eigen::MatrixXd & Ja)
    {
        assert(xv.size() == nx + ny);
        Eigen::VectorXd x = xv.head(nx);
        Eigen::VectorXd v = xv.tail(ny);
        Eigen::MatrixXd J;
        Eigen::VectorXd ya = predictFeatureBundle(x, J, system, idxLandmarks) + v;
        Ja.resize(ny, nx + ny);
        Ja << J, Eigen::MatrixXd::Identity(ny, ny);
        return ya;
    };

    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

#include "association_util.h"
const std::vector<int> & MeasurementPointBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    GaussianInfo<double> featureBundleDensity = predictFeatureBundleDensity(system, idxLandmarks);
    snn(system, featureBundleDensity, idxLandmarks, Y_, camera_, idxFeatures_);
    return idxFeatures_;
}

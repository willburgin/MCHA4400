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
#include "MeasurementSLAMDuckBundle.h"
#include "rotation.hpp"
#include "association_util.h"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

MeasurementDuckBundle::MeasurementDuckBundle(double time, const Eigen::Matrix<double, 3, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_c_(10.0) // TODO: Assignment(s)
    , sigma_a_(50.0) // TODO: Assignment(s)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM * MeasurementDuckBundle::clone() const
{
    return new MeasurementDuckBundle(*this);
}

Eigen::VectorXd MeasurementDuckBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

// build the log-likelihood function
double MeasurementDuckBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    return logLikelihoodTemplated<double>(x, systemSLAM);
}

double MeasurementDuckBundle::logLikelihood(
    const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    // Forward-mode autodiff
    autodiff::dual logLik_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    
    g = gradient(
        [&](const Eigen::VectorX<autodiff::dual> & x_ad) { 
            return logLikelihoodTemplated<autodiff::dual>(x_ad, systemSLAM); 
        },
        wrt(x_dual), 
        at(x_dual), 
        logLik_dual
    );
    
    return val(logLik_dual);
}

double MeasurementDuckBundle::logLikelihood(
    const Eigen::VectorXd & x, const SystemEstimator & system, 
    Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    // Forward-mode autodiff with dual2nd for Hessian
    autodiff::dual2nd logLik_dual;
    Eigen::VectorX<autodiff::dual2nd> x_dual = x.cast<autodiff::dual2nd>();
    
    H = hessian(
        [&](const Eigen::VectorX<autodiff::dual2nd> & x_ad) { 
            return logLikelihoodTemplated<autodiff::dual2nd>(x_ad, systemSLAM); 
        },
        wrt(x_dual), 
        at(x_dual), 
        logLik_dual,
        g
    );
    
    return val(logLik_dual);
}

void MeasurementDuckBundle::update(SystemBase & system)
{
    SystemSLAM & systemSLAM = dynamic_cast<SystemSLAM &>(system);
    
    // Filter to only visible landmarks using FOV check
    Eigen::VectorXd x = systemSLAM.density.mean();
    Eigen::Vector3d rCNn = systemSLAM.cameraPositionDensity(camera_).mean();
    Eigen::Vector3d Thetanc = systemSLAM.cameraOrientationEulerDensity(camera_).mean();
    Eigen::Matrix3d Rnc = rpy2rot(Thetanc);
    
    visibleLandmarks_.clear();
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i)
    {
        std::size_t idx = systemSLAM.landmarkPositionIndex(i);
        Eigen::Vector3d rJNn = x.segment<3>(idx);
        
        // Transform to camera frame
        Eigen::Vector3d rJCc = Rnc.transpose() * (rJNn - rCNn);
        
        cv::Vec3d rJCc_cv(rJCc(0), rJCc(1), rJCc(2));
        if (camera_.isVectorWithinFOV(rJCc_cv)) {
            visibleLandmarks_.push_back(i);
        }
    }
    
    idxFeatures_ = associate(systemSLAM, visibleLandmarks_);
    
    // Find which detections are already associated
    std::vector<bool> detectionUsed(Y_.cols(), false);
    for (int idx : idxFeatures_) {
        if (idx >= 0) {
            detectionUsed[idx] = true;
        }
    }
    // Landmark initialization constraints
    int maxVisibleLandmarksPerFrame = 8;     // Max landmarks visible at once
    int maxTotalLandmarks = 60;              // Total capacity
    int currentVisible = visibleLandmarks_.size();
    int currentTotal = systemSLAM.numberLandmarks();
    
    // Calculate how many we can initialize
    int spotsAvailableInFrame = maxVisibleLandmarksPerFrame - currentVisible;
    int spotsAvailableTotal = maxTotalLandmarks - currentTotal;
    int landmarksToInitialize = std::min(spotsAvailableInFrame, spotsAvailableTotal);
    
    if (landmarksToInitialize <= 0) {
        Measurement::update(system);
        return;
    }
    
    int landmarksInitializedThisFrame = 0;
    bool hasSmallDucks = false;
    for (int detectionIdx = 0; detectionIdx < Y_.cols(); ++detectionIdx) {
        double area_pixels = Y_(2, detectionIdx);
        if (area_pixels <= 6000) {
            hasSmallDucks = true;
            break;
        }
    }
    
    // Loop over all detected ducks and initialize new landmarks
    for (std::size_t detectionIdx = 0; detectionIdx < Y_.cols(); ++detectionIdx)
    {
        double area_pixels = Y_(2, detectionIdx);
        if (landmarksInitializedThisFrame >= landmarksToInitialize) {
            break;
        }
        
        if (!detectionUsed[detectionIdx])
        {
            Eigen::Vector2d centroid_pixel = Y_.block<2, 1>(0, detectionIdx);
            
            // is detection too close to image border?
            double borderMargin = 100.0;  // pixels
            bool tooCloseToEdge = (centroid_pixel(0) < borderMargin || 
                                centroid_pixel(0) > camera_.imageSize.width - borderMargin ||
                                centroid_pixel(1) < borderMargin || 
                                centroid_pixel(1) > camera_.imageSize.height - borderMargin);
            
            if (tooCloseToEdge) {
                continue;  // skip this detection
            }
            
            // is this detection within 4-sigma confidence region of existing landmarks?
            bool withinConfidenceRegion = false;
            
            for (size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) {
                // get predicted feature density for this landmark (2D in pixel space)
                GaussianInfo<double> featureDensity = predictFeatureDensity(systemSLAM, i);
                
                // check if detection is within 4-sigma confidence region
                double nSigma = 4.0;
                if (featureDensity.isWithinConfidenceRegion(centroid_pixel, nSigma)) {
                    withinConfidenceRegion = true;
                    break;
                }
            }
            
            if (!withinConfidenceRegion) {                    
                // Initialize if: (small duck) OR (no small ducks exist in frame)
                if (area_pixels <= 6000 || !hasSmallDucks) {
                    double duck_radius = 0.03;
                    double fx = camera_.cameraMatrix.at<double>(0, 0);
                    double fy = camera_.cameraMatrix.at<double>(1, 1);
                    
                    double estimated_depth = std::sqrt((fx * fy * duck_radius * duck_radius) / area_pixels);
                    estimated_depth = std::clamp(estimated_depth, 0.1, 10.0);
                    std::cout << "  Estimated Depth " << estimated_depth << std::endl;

                    // Backproject centroid to 3D at estimated depth
                    cv::Vec2d centroid_cv(centroid_pixel(0), centroid_pixel(1));
                    cv::Vec3d rJCc_cv = camera_.pixelToVector(centroid_cv);
                    Eigen::Vector3d rJCc_unit(rJCc_cv[0], rJCc_cv[1], rJCc_cv[2]);
                    Eigen::Vector3d rJCc = rJCc_unit.normalized() * estimated_depth;
                    
                    // Transform to world frame
                    Eigen::Vector3d rJNn = Rnc * rJCc + rCNn;
                    
                    // Create new landmark with prior
                    Eigen::VectorXd mu_new = rJNn;
                    double epsilon = 50.0;
                    Eigen::MatrixXd Xi_new = epsilon * Eigen::MatrixXd::Identity(3, 3);
                    Eigen::VectorXd nu_new = Xi_new * mu_new;
                    
                    GaussianInfo<double> newLandmarkDensity = GaussianInfo<double>::fromSqrtInfo(nu_new, Xi_new);
                    systemSLAM.density *= newLandmarkDensity;
                    landmarksInitializedThisFrame++;
                }
            }
        }
    }
    
    std::cout << "=== BEFORE Measurement::update() ===" << std::endl;
    Eigen::VectorXd x_before = systemSLAM.density.mean();
    std::cout << "  Camera position before: " << x_before.segment<3>(6).transpose() << std::endl;
    
    // Measurement update
    Measurement::update(system);
    
    Eigen::VectorXd x_after = systemSLAM.density.mean();
    std::cout << "=== AFTER Measurement::update() ===" << std::endl;
    std::cout << "  Camera position after: " << x_after.segment<3>(6).transpose() << std::endl;
    std::cout << "  Position change: " << (x_after.segment<3>(6) - x_before.segment<3>(6)).norm() << std::endl;
}

// Image feature location for a given landmark and Jacobian
Eigen::Vector2d MeasurementDuckBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
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
    Eigen::Matrix23d J_camera;
    Eigen::Vector2d rQOi = camera_.vectorToPixel(rPCc, J_camera);

    J.resize(2, x.size());
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

    // Equation (27a):
    Eigen::Matrix<double, 2, 3> dhj_drPNn = J_camera * Rbc.transpose() * Rnb.transpose();
    J.block<2, 3>(0, idx) = dhj_drPNn;

    // Equation (27b):
    Eigen::Matrix<double, 2, 3> dhj_drBNn = -J_camera * Rbc.transpose() * Rnb.transpose();
    J.block<2, 3>(0, 6) = dhj_drBNn;

    Eigen::Vector3d rPNn_minus_rBNn = rPNn - rBNn;
    Eigen::Matrix3d dRx_dphi;
    rotx(Thetanb(0), dRx_dphi);
    Eigen::Matrix3d dRnb_dphi = rotz(Thetanb(2)) * roty(Thetanb(1)) * dRx_dphi;

    Eigen::Matrix3d dRy_dtheta;
    roty(Thetanb(1), dRy_dtheta);
    Eigen::Matrix3d dRnb_dtheta = rotz(Thetanb(2)) * dRy_dtheta * rotx(Thetanb(0));

    Eigen::Matrix3d dRz_dpsi;
    rotz(Thetanb(2), dRz_dpsi);
    Eigen::Matrix3d dRnb_dpsi = dRz_dpsi * roty(Thetanb(1)) * rotx(Thetanb(0));

    // Apply equation (27c) for each Euler angle
    J.col(9)  = J_camera * Rbc.transpose() * dRnb_dphi.transpose() * rPNn_minus_rBNn;    
    J.col(10) = J_camera * Rbc.transpose() * dRnb_dtheta.transpose() * rPNn_minus_rBNn;  
    J.col(11) = J_camera * Rbc.transpose() * dRnb_dpsi.transpose() * rPNn_minus_rBNn;   
    // std::cout << "J: " << J << std::endl;

    return rQOi;
}

// Density of image feature location for a given landmark
GaussianInfo<double> MeasurementDuckBundle::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
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
    
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_c_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd MeasurementDuckBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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
        h.segment<2>(2*i) = rQOi;
        // Set pair of rows of J
        // TODO: Lab 9
        J.block<2, Eigen::Dynamic>(2*i, 0, 2, nx) = Jfeature; 
    }
    return h;
}

// Density of image features for a set of landmarks
GaussianInfo<double> MeasurementDuckBundle::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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

    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_c_*Eigen::MatrixXd::Identity(ny, ny));
    auto pxv = system.density*pv;   // p(x, v) = p(x)*p(v)
    return pxv.affineTransform(func);
}

#include "association_util.h"
const std::vector<int> & MeasurementDuckBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    // guard to prevent snn being called with an empty idxLandmarks
    if (idxLandmarks.empty()) {
        idxFeatures_.clear();
        return idxFeatures_;
    }
    GaussianInfo<double> featureBundleDensity = predictFeatureBundleDensity(system, idxLandmarks);
    // Pass only the centroid rows (first 2 rows) to snn, not the area
    Eigen::Matrix<double, 2, Eigen::Dynamic> Y_centroids = Y_.topRows<2>();
    snn(system, featureBundleDensity, idxLandmarks, Y_centroids, camera_, idxFeatures_);
    return idxFeatures_;
}

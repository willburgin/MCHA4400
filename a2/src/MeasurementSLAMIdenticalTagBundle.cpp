#include <cstddef>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"
#include "Camera.h"
#include "Measurement.h"
#include "MeasurementSLAM.h"
#include "MeasurementSLAMIdenticalTagBundle.h"
#include "rotation.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

MeasurementSLAMIdenticalTagBundle::MeasurementSLAMIdenticalTagBundle(double time, const Eigen::Matrix<double, 8, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(5.2) // TODO: Assignment(s)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM * MeasurementSLAMIdenticalTagBundle::clone() const
{
    return new MeasurementSLAMIdenticalTagBundle(*this);
}

Eigen::VectorXd MeasurementSLAMIdenticalTagBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

// build the log-likelihood function
double MeasurementSLAMIdenticalTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const SystemVisualNav & systemVisualNav = dynamic_cast<const SystemVisualNav &>(system);
    return logLikelihoodTemplated<double>(x, systemVisualNav);
}

// gradient version using GaussianInfo log function with chain rule
double MeasurementSLAMIdenticalTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    const SystemVisualNav & systemVisualNav = dynamic_cast<const SystemVisualNav &>(system);
    
    // compute log-likelihood using the base function
    double logLik = logLikelihood(x, system);
    
    // initialize gradient
    g = Eigen::VectorXd::Zero(x.size());
    
    // create measurement noise model for a single corner
    Eigen::MatrixXd S = sigma_ * Eigen::MatrixXd::Identity(2, 2);
    GaussianInfo<double> measurementModel = GaussianInfo<double>::fromSqrtMoment(S);
    
    // compute gradient for all associated feature/landmark pairs
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        if (idxFeatures_[j] >= 0) {  
            int detectionIdx = idxFeatures_[j];
            
            // predict all 4 corners with Jacobian
            Eigen::MatrixXd J_h;
            Eigen::Matrix<double, 8, 1> h_pred = predictFeature(x, J_h, systemVisualNav, j);
            
            // sum over all 4 corners
            for (int c = 0; c < 4; ++c) {
                // get measured corner position
                Eigen::Vector2d y_ic = Y_.block<2, 1>(2*c, detectionIdx);
                
                // get predicted corner position
                Eigen::Vector2d h_ic = h_pred.segment<2>(2*c);
                
                // compute residual
                Eigen::Vector2d residual = y_ic - h_ic;
                
                // use GaussianInfo log function to get gradient w.r.t. residual
                Eigen::VectorXd g_residual;
                measurementModel.log(residual, g_residual);
                
                // chain rule
                Eigen::MatrixXd J_corner = J_h.block(2*c, 0, 2, x.size());
                g += -J_corner.transpose() * g_residual;
            }
        }
    }
    
    // no gradient contribution from penalty term (constant w.r.t. x)
    
    return logLik;
}

// hessian version using forward-mode autodiff
double MeasurementSLAMIdenticalTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    const SystemVisualNav & systemVisualNav = dynamic_cast<const SystemVisualNav &>(system);
    
    // forward-mode autodifferentiation with dual2nd for Hessian
    autodiff::dual2nd logLik_dual;
    Eigen::VectorX<autodiff::dual2nd> x_dual = x.cast<autodiff::dual2nd>();
    
    H = hessian(
        [&](const Eigen::VectorX<autodiff::dual2nd> & x_ad) { 
            return logLikelihoodTemplated<autodiff::dual2nd>(x_ad, systemVisualNav); 
        },
        wrt(x_dual), 
        at(x_dual), 
        logLik_dual,
        g
    );
    
    return val(logLik_dual);
}

void MeasurementIdenticalTagBundle::update(SystemBase & system)
{
    SystemVisualNav & systemVisualNav = dynamic_cast<SystemVisualNav &>(system);
    
    // Get current state estimate
    Eigen::VectorXd x = systemVisualNav.density.mean();
    Eigen::Vector3d rCNn = systemVisualNav.cameraPositionDensity(camera_).mean();
    Eigen::Vector3d Thetanc = systemVisualNav.cameraOrientationEulerDensity(camera_).mean();
    Eigen::Matrix3d Rnc = rpy2rot(Thetanc);
    
    // Filter to only visible landmarks using FOV check
    visibleLandmarks_.clear();
    for (std::size_t i = 0; i < systemVisualNav.numberLandmarks(); ++i)
    {
        std::size_t idx = systemVisualNav.landmarkPositionIndex(i);
        Eigen::Vector3d rJNn = x.segment<3>(idx);
        
        // Transform to camera frame (use landmark position, not orientation for FOV check)
        Eigen::Vector3d rJCc = Rnc.transpose() * (rJNn - rCNn);
        
        // Check if in front of camera and within FOV
        if (rJCc(2) > 0.0) {
            cv::Vec3d rJCc_cv(rJCc(0), rJCc(1), rJCc(2));
            if (camera_.isVectorWithinFOV(rJCc_cv)) {
                visibleLandmarks_.push_back(i);
            }
        }
    }
    
    // Associate detected tags with visible landmarks
    idxFeatures_ = associate(systemVisualNav, visibleLandmarks_);
    
    // Find which detections are already associated
    std::vector<bool> detectionUsed(frameMarkerCorners_.size(), false);
    for (int idx : idxFeatures_) {
        if (idx >= 0) {
            detectionUsed[idx] = true;
        }
    }
    
    // Loop over all detected tags and initialize new landmarks for unassociated detections
    for (std::size_t detectionIdx = 0; detectionIdx < frameMarkerCorners_.size(); ++detectionIdx)
    {
        if (!detectionUsed[detectionIdx])
        {
            // Get the 4 corners for this detection
            std::vector<cv::Point2f> imageCorners;
            for (int c = 0; c < 4; ++c) {
                imageCorners.push_back(cv::Point2f(Y_(2*c, detectionIdx), Y_(2*c+1, detectionIdx)));
            }
            
            // Filter 1: Check if any corner is too close to image border
            double borderMargin = 50.0;  // pixels
            bool tooCloseToEdge = false;
            for (const auto& corner : imageCorners) {
                if (corner.x < borderMargin || 
                    corner.x > camera_.imageSize.width - borderMargin ||
                    corner.y < borderMargin || 
                    corner.y > camera_.imageSize.height - borderMargin) {
                    tooCloseToEdge = true;
                    break;
                }
            }
            
            if (tooCloseToEdge) {
                continue;
            }
            
            // Get body pose for transformation
            Eigen::Vector3d rBNn = systemVisualNav.density.mean().segment<3>(6);
            Eigen::Vector3d Thetanb = systemVisualNav.density.mean().segment<3>(9);
            Pose<double> Tnb(rpy2rot(Thetanb), rBNn);
            
            // Define marker corners in marker frame (edge length = 166mm)
            double l_half = 0.166 / 2.0;
            std::vector<cv::Point3f> markerCorners3D = {
                cv::Point3f(-l_half,  l_half, 0.0f),
                cv::Point3f( l_half,  l_half, 0.0f),
                cv::Point3f( l_half, -l_half, 0.0f),
                cv::Point3f(-l_half, -l_half, 0.0f)
            };
            
            // Solve PnP to get marker pose relative to camera
            cv::Mat rvec, tvec;
            bool success = cv::solvePnP(markerCorners3D, imageCorners, 
                                       camera_.cameraMatrix, camera_.distCoeffs, 
                                       rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
            
            if (!success) {
                std::cout << "PnP failed for detection " << detectionIdx << std::endl;
                continue;
            }
            
            // Convert to pose in world frame
            Pose<double> Tcj(rvec, tvec);  
            Pose<double> Tnj = Tnb * camera_.Tbc * Tcj; 
            
            // Extract position and orientation
            Eigen::Vector3d posInit = Tnj.translationVector;
            Eigen::Vector3d oriInit = rot2rpy(Tnj.rotationMatrix);
            
                        
            // Create landmark state: [position, orientation]
            Eigen::VectorXd mu_new(6);
            mu_new << posInit, oriInit;
            
            // Set prior information (tune epsilon based on PnP accuracy)
            double epsilon = 10.0;  
            Eigen::MatrixXd Xi_new = epsilon * Eigen::MatrixXd::Identity(6, 6);
            Eigen::VectorXd nu_new = Xi_new * mu_new;
            
            // Add new landmark to the map
            GaussianInfo<double> newLandmarkDensity = 
                GaussianInfo<double>::fromSqrtInfo(nu_new, Xi_new);
            systemVisualNav.density *= newLandmarkDensity;
            
            std::cout << "Initialized new tag landmark #" << systemVisualNav.numberLandmarks() - 1
                      << " at position: " << posInit.transpose() 
                      << " (depth: " << posInit(2) << " m)" << std::endl;
            
            // Add newly initialized landmark to tracking lists
            std::size_t newLandmarkIdx = systemVisualNav.numberLandmarks() - 1;
            visibleLandmarks_.push_back(newLandmarkIdx);
            idxFeatures_.push_back(static_cast<int>(detectionIdx));
        }
    }
    
    // Perform measurement update with all associated detections
    Measurement::update(system);
}

// image feature location for a given landmark (ArUco marker) and Jacobian
Eigen::Matrix<double, 8, 1> MeasurementSLAMIdenticalTagBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, std::size_t idxLandmark) const
{
    // get camera pose from state
    Pose<double> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // get landmark pose from state 
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3d rJNn = x.segment<3>(idx);       // position
    Eigen::Vector3d Thetanj = x.segment<3>(idx + 3); // orientation

    // create landmark pose
    Pose<double> Tnj;
    Tnj.translationVector = rJNn;
    Tnj.rotationMatrix = rpy2rot(Thetanj);  

    // aruco marker corner positions in marker frame 
    double l_half = 0.166 / 2.0; // half edge length in meters
    std::vector<Eigen::Vector3d> rJcJj = {
        Eigen::Vector3d(-l_half, l_half, 0.0),
        Eigen::Vector3d( l_half, l_half, 0.0), 
        Eigen::Vector3d( l_half,  -l_half, 0.0),
        Eigen::Vector3d(-l_half,  -l_half, 0.0)  
    };

    // compute predicted pixel coordinates for all 4 corners
    Eigen::Matrix<double, 8, 1> h; // 4 corners × 2 coordinates = 8 
    
    // initialize Jacobian 
    J.resize(8, x.size()); // 8 measurement outputs × state size
    J.setZero();

    // extract body state components
    Eigen::Vector3d rBNn = x.segment<3>(6);
    Eigen::Vector3d Thetanb = x.segment<3>(9);

    // get body pose
    Pose<double> Tnb;
    Tnb.rotationMatrix = rpy2rot(Thetanb);
    Tnb.translationVector = rBNn;

    // get camera-to-body transformation
    Pose<double> Tbc = camera_.Tbc;
    Eigen::Matrix3d Rnb = Tnb.rotationMatrix;
    Eigen::Matrix3d Rbc = Tbc.rotationMatrix;
    
    for (int i = 0; i < 4; ++i)
    {
        // transform corner from marker frame to world frame: rJcNn = Rnj * rJcJj + rJNn
        Eigen::Vector3d rJcNn = Tnj.rotationMatrix * rJcJj[i] + rJNn;
        
        // transform to camera coordinates
        Eigen::Vector3d rJcCn = Tnc.rotationMatrix.transpose() * (rJcNn - Tnc.translationVector);
        
        // get pixel coordinates with Jacobian w.r.t. camera coordinates
        Eigen::Matrix23d J_camera;  
        Eigen::Vector2d rQOi = camera_.vectorToPixel(rJcCn, J_camera);
        
        // store in output vector
        h.segment<2>(2*i) = rQOi;
        
        // compute partial Jacobians for this corner
        // (landmark position derivatives) 
        Eigen::Matrix<double, 2, 3> dhi_drJNn = J_camera * Rbc.transpose() * Rnb.transpose();
        J.block<2, 3>(2*i, idx) = dhi_drJNn; 

        // (landmark orientation derivatives) 
        Eigen::Matrix3d dRxj_dphi;
        rotx(Thetanj(0), dRxj_dphi);
        Eigen::Matrix3d dRnj_dphi = rotz(Thetanj(2)) * roty(Thetanj(1)) * dRxj_dphi;

        Eigen::Matrix3d dRyj_dtheta;
        roty(Thetanj(1), dRyj_dtheta);
        Eigen::Matrix3d dRnj_dtheta = rotz(Thetanj(2)) * dRyj_dtheta * rotx(Thetanj(0));
 
        Eigen::Matrix3d dRzj_dpsi;
        rotz(Thetanj(2), dRzj_dpsi);
        Eigen::Matrix3d dRnj_dpsi = dRzj_dpsi * roty(Thetanj(1)) * rotx(Thetanj(0));

        // (body position derivatives)  
        Eigen::Matrix<double, 2, 3> dhi_drBNn = -J_camera * Rbc.transpose() * Rnb.transpose();
        J.block<2, 3>(2*i, 6) = dhi_drBNn;  

        // (body orientation derivatives)
        Eigen::Vector3d rJcNn_minus_rBNn = rJcNn - rBNn;  

        Eigen::Matrix3d dRx_dphi;
        rotx(Thetanb(0), dRx_dphi);
        Eigen::Matrix3d dRnb_dphi = rotz(Thetanb(2)) * roty(Thetanb(1)) * dRx_dphi;

        Eigen::Matrix3d dRy_dtheta;
        roty(Thetanb(1), dRy_dtheta);
        Eigen::Matrix3d dRnb_dtheta = rotz(Thetanb(2)) * dRy_dtheta * rotx(Thetanb(0));
 
        Eigen::Matrix3d dRz_dpsi;
        rotz(Thetanb(2), dRz_dpsi);
        Eigen::Matrix3d dRnb_dpsi = dRz_dpsi * roty(Thetanb(1)) * rotx(Thetanb(0));        

        // apply body orientation derivative
        J.block<2, 1>(2*i, 9)  = J_camera * Rbc.transpose() * dRnb_dphi.transpose() * rJcNn_minus_rBNn;    
        J.block<2, 1>(2*i, 10) = J_camera * Rbc.transpose() * dRnb_dtheta.transpose() * rJcNn_minus_rBNn;  
        J.block<2, 1>(2*i, 11) = J_camera * Rbc.transpose() * dRnb_dpsi.transpose() * rJcNn_minus_rBNn;   

        // (landmark orientation derivatives)
        Eigen::Vector3d rJcJj_i = rJcJj[i]; 
        J.block<2, 1>(2*i, idx+3) = J_camera * Rbc.transpose() * Rnb.transpose() * dRnj_dphi * rJcJj_i;     
        J.block<2, 1>(2*i, idx+4) = J_camera * Rbc.transpose() * Rnb.transpose() * dRnj_dtheta * rJcJj_i;     
        J.block<2, 1>(2*i, idx+5) = J_camera * Rbc.transpose() * Rnb.transpose() * dRnj_dpsi * rJcJj_i;     
    }
    return h;
}

// density of image feature location for a given landmark
GaussianInfo<double> MeasurementSLAMIdenticalTagBundle::predictFeatureDensity(const SystemVisualNav & system, std::size_t idxLandmark) const
{
    const std::size_t & nx = system.density.dim();
    
    // get the 3D position marginal from the state 
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    auto idxPos = Eigen::seqN(idx, 3);
    GaussianInfo<double> p3D = system.density.marginal(idxPos);
    
    // helper to compute the 2D pixel center and its Jacobian
    const auto projectCenter = [&](const Eigen::VectorXd & rJNn, Eigen::MatrixXd & J) -> Eigen::Vector2d
    {
        // get camera pose from state mean
        Eigen::VectorXd x = system.density.mean();
        Eigen::Vector3d rCNn = system.cameraPosition(camera_, x);
        Eigen::Matrix3d Rnc = system.cameraOrientation(camera_, x);
        
        // transform landmark center to camera frame
        Eigen::Vector3d rJcCc = Rnc.transpose() * (rJNn - rCNn);
        
        // project to pixels with Jacobian
        Eigen::Matrix23d J_camera;
        Eigen::Vector2d pixel = camera_.vectorToPixel(rJcCc, J_camera);
        
        // chain rule
        J = J_camera * Rnc.transpose();
        
        return pixel;
    };
    
    // add measurement noise
    const std::size_t ny = 2;
    auto pv = GaussianInfo<double>::fromSqrtMoment(sigma_ * Eigen::MatrixXd::Identity(ny, ny));
    auto p3Dv = p3D * pv;
    
    // project through camera
    return p3Dv.affineTransform([&](const Eigen::VectorXd & rJNnv, Eigen::MatrixXd & Ja) -> Eigen::Vector2d {
        Eigen::Vector3d rJNn = rJNnv.head(3);
        Eigen::Vector2d v = rJNnv.tail(2);
        
        Eigen::MatrixXd J;
        Eigen::Vector2d pixel = projectCenter(rJNn, J);
        
        Ja.resize(2, 5);
        Ja << J, Eigen::Matrix2d::Identity();
        
        return pixel + v;
    });
}

// Image feature locations for a bundle of landmarks
Eigen::VectorXd MeasurementSLAMIdenticalTagBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nL = idxLandmarks.size();
    const std::size_t & nx = system.density.dim();
    assert(x.size() == nx);

    Eigen::VectorXd h(8*nL); // 8 coordinates per landmark (4 corners × 2 coordinates)
    J.resize(8*nL, nx);
    J.setZero();
    
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::MatrixXd Jfeature;
        Eigen::Matrix<double, 8, 1> corners = predictFeature(x, Jfeature, system, idxLandmarks[i]);
        
        // Set 8 elements of h for this landmark (4 corners)
        h.segment<8>(8*i) = corners;
        
        // Set 8 rows of J for this landmark 
        J.block(8*i, 0, 8, nx) = Jfeature;
    }
    return h;
}

// Density of image features for a set of landmarks
GaussianInfo<double> MeasurementSLAMIdenticalTagBundle::predictFeatureBundleDensity(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const
{
    const std::size_t & nx = system.density.dim();
    const std::size_t ny = 8*idxLandmarks.size(); // 8 coordinates per landmark

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

const std::vector<int> & MeasurementSLAMIdenticalTagBundle::associate(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks)
{
    // guard to prevent snn being called with an empty idxLandmarks
    if (idxLandmarks.empty()) {
        idxFeatures_.clear();
        return idxFeatures_;
    }
    GaussianInfo<double> featureBundleDensity = predictFeatureBundleDensity(system, idxLandmarks);
    snn(system, featureBundleDensity, idxLandmarks, Y_, camera_, idxFeatures_);
    return idxFeatures_;
}

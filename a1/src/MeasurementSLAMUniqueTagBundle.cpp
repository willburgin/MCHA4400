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
#include "MeasurementSLAMUniqueTagBundle.h"
#include "SystemSLAMPoseLandmarks.h"
#include "rotation.hpp"
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

MeasurementSLAMUniqueTagBundle::MeasurementSLAMUniqueTagBundle(double time, const Eigen::Matrix<double, 8, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(5.0) // TODO: Assignment(s)
{
    // updateMethod_ = UpdateMethod::BFGSLMSQRT;
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
}

MeasurementSLAM * MeasurementSLAMUniqueTagBundle::clone() const
{
    return new MeasurementSLAMUniqueTagBundle(*this);
}

Eigen::VectorXd MeasurementSLAMUniqueTagBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y(Y_.size());
    throw std::runtime_error("Not implemented");
    return y;
}

// Templated version for autodiff
template <typename Scalar>
Scalar MeasurementSLAMUniqueTagBundle::logLikelihoodTemplated(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    Scalar logLik = Scalar(0.0);
    
    // Count unassociated landmarks for penalty term
    int numUnassociated = 0;
    for (int assoc : idxFeatures_) {
        if (assoc < 0) numUnassociated++;
    }
    
    // Sum log-likelihoods over all associated feature/landmark pairs
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        if (idxFeatures_[j] >= 0) {  // This landmark is associated
            int detectionIdx = idxFeatures_[j];
            // std::cout << "Using landmark " << j << " with detection " << detectionIdx 
            // << " (marker ID " << frameMarkerIDs_[detectionIdx] << ")" << std::endl;
            
            // Predict all 4 corners for this landmark
            Eigen::Matrix<Scalar, 8, 1> h_pred = predictFeature(x, systemSLAM, j);
            
            // Sum over all 4 corners
            for (int c = 0; c < 4; ++c) {
                // std::cout << "  Corner " << c << ": (" 
                // << Y_(2*c, detectionIdx) << ", " 
                // << Y_(2*c+1, detectionIdx) << ")" << std::endl;
                // Get measured corner position (always double)
                Eigen::Vector2d y_ic = Y_.block<2, 1>(2*c, detectionIdx);
                
                // Get predicted corner position
                Eigen::Vector2<Scalar> h_ic = h_pred.template segment<2>(2*c);
                
                // Compute residual
                Eigen::Vector2<Scalar> residual;
                residual(0) = Scalar(y_ic(0)) - h_ic(0);
                residual(1) = Scalar(y_ic(1)) - h_ic(1);
                
                // Add log-likelihood: log N(y; h, σ²I) = -||y-h||²/(2σ²) - log(2πσ²)
                logLik += -Scalar(0.5) * residual.squaredNorm() / Scalar(sigma_ * sigma_);
                logLik += -Scalar(std::log(2.0 * M_PI * sigma_ * sigma_));
            }
        }
    }
    
    // // Add penalty term for unassociated visible landmarks
    double imageArea = camera_.imageSize.width * camera_.imageSize.height;
    logLik -= Scalar(4.0 * numUnassociated * std::log(imageArea));
    
    return logLik;
}

// Regular version (no derivatives)
double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    return logLikelihoodTemplated<double>(x, systemSLAM);
}

// Gradient version using forward-mode autodiff
double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    // Forward-mode autodifferentiation
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

// Hessian version using forward-mode autodiff
double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    // Forward-mode autodifferentiation with dual2nd for Hessian
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

void MeasurementSLAMUniqueTagBundle::update(SystemBase & system)
{
    SystemSLAMPoseLandmarks & systemSLAM = dynamic_cast<SystemSLAMPoseLandmarks &>(system);
    
    // associate detected features with existing landmarks
    std::vector<std::size_t> idxLandmarks;
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) 
    {
        idxLandmarks.push_back(i);
    }
    idxFeatures_ = associate(systemSLAM, idxLandmarks);
    
    // loop over all detected markers
    for (std::size_t detectionIdx = 0; detectionIdx < frameMarkerIDs_.size(); ++detectionIdx)
    {
        int detectedMarkerID = frameMarkerIDs_[detectionIdx];
        
        if (!systemSLAM.isMarkerKnown(detectedMarkerID))
        {
            systemSLAM.addKnownMarkerID(detectedMarkerID);
            
            // Get camera pose
            auto camPos = systemSLAM.cameraPositionDensity(camera_).mean();
            auto camRpy = systemSLAM.cameraOrientationEulerDensity(camera_).mean();
            Eigen::Matrix3d Rnc = rpy2rot(camRpy);
            
            // Define marker corners in marker frame (same as in your associate function)
            double l_half = 0.166 / 2.0;
            std::vector<cv::Point3f> markerCorners3D = {
                cv::Point3f(-l_half,  l_half, 0.0f),
                cv::Point3f( l_half,  l_half, 0.0f),
                cv::Point3f( l_half, -l_half, 0.0f),
                cv::Point3f(-l_half, -l_half, 0.0f)
            };
            
            // Get detected corners for this marker
            std::vector<cv::Point2f> imageCorners;
            for (int c = 0; c < 4; ++c) {
                imageCorners.push_back(cv::Point2f(Y_(2*c, detectionIdx), Y_(2*c+1, detectionIdx)));
            }
            Eigen::Vector3d rBNn = systemSLAM.density.mean().segment<3>(6);
            Eigen::Vector3d Thetanb = systemSLAM.density.mean().segment<3>(9);
            Pose<double> Tnb(rpy2rot(Thetanb), rBNn); // body pose
            
            // solve PnP to get marker pose relative to camera
            cv::Mat rvec, tvec;
            cv::solvePnP(markerCorners3D, imageCorners, camera_.cameraMatrix, 
                        camera_.distCoeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
            
            // convert to pose
            Pose<double> Tcj(rvec, tvec);  // marker relative to camera
            Pose<double> Tnj = Tnb * camera_.Tbc * Tcj;  // marker in world frame
            
            // extract position and orientation
            Eigen::Vector3d posInit = Tnj.translationVector;
            Eigen::Vector3d oriInit = rot2rpy(Tnj.rotationMatrix);
            
            Eigen::VectorXd mu_new(6);
            mu_new << posInit, oriInit;
            
            double epsilon = 40;
            Eigen::MatrixXd Xi_new = epsilon * Eigen::MatrixXd::Identity(6, 6);
            Eigen::VectorXd nu_new = Xi_new * mu_new;
            
            GaussianInfo<double> newLandmarkDensity = 
                GaussianInfo<double>::fromSqrtInfo(nu_new, Xi_new);
            
            systemSLAM.density *= newLandmarkDensity;
        }
    }
    
    // measurement update with associated detections
    Measurement::update(system);
}

// image feature location for a given landmark (ArUco marker) and Jacobian
Eigen::Matrix<double, 8, 1> MeasurementSLAMUniqueTagBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
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
    Tnj.rotationMatrix = rpy2rot(Thetanj);  // rotation matrix

    // aruco marker corner positions in marker frame 
    double l_half = 0.166 / 2.0; // half edge length in meters
    std::vector<Eigen::Vector3d> rJcJj = {
        Eigen::Vector3d(-l_half, l_half, 0.0),
        Eigen::Vector3d( l_half, l_half, 0.0), 
        Eigen::Vector3d( l_half,  -l_half, 0.0),
        Eigen::Vector3d(-l_half,  -l_half, 0.0)  
    };

    // compute predicted pixel coordinates for all 4 corners
    Eigen::Matrix<double, 8, 1> h; // 4 corners × 2 coordinates = 8 values
    
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
        Eigen::Vector3d rJcJj_i = rJcJj[i]; // corner position in marker frame
        J.block<2, 1>(2*i, idx+3) = J_camera * Rbc.transpose() * Rnb.transpose() * dRnj_dphi * rJcJj_i;     
        J.block<2, 1>(2*i, idx+4) = J_camera * Rbc.transpose() * Rnb.transpose() * dRnj_dtheta * rJcJj_i;     
        J.block<2, 1>(2*i, idx+5) = J_camera * Rbc.transpose() * Rnb.transpose() * dRnj_dpsi * rJcJj_i;     
    }
    return h;
}

// Density of image feature location for a given landmark
GaussianInfo<double> MeasurementSLAMUniqueTagBundle::predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const
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
Eigen::VectorXd MeasurementSLAMUniqueTagBundle::predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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
GaussianInfo<double> MeasurementSLAMUniqueTagBundle::predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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

const std::vector<int> & MeasurementSLAMUniqueTagBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    const SystemSLAMPoseLandmarks & systemSLAM = dynamic_cast<const SystemSLAMPoseLandmarks &>(system);
    const std::vector<int>& knownMarkerIDs = systemSLAM.getKnownMarkerIDs();
    
    // Get current state for visibility checking
    Eigen::VectorXd x = systemSLAM.density.mean();
    Eigen::Vector3d rCNn = systemSLAM.cameraPositionDensity(camera_).mean();
    Eigen::Vector3d Thetanc = systemSLAM.cameraOrientationEulerDensity(camera_).mean();
    Pose<double> Tnc;
    Tnc.translationVector = rCNn;
    Tnc.rotationMatrix = rpy2rot(Thetanc);
    
    // Initialize
    idxFeatures_.clear();
    idxFeatures_.resize(knownMarkerIDs.size(), -1);
    visibleLandmarks_.clear();
    visibleLandmarks_.resize(knownMarkerIDs.size(), false);
    
    // Check visibility for ALL known landmarks (based on CENTER position)
    for (std::size_t landmarkIdx = 0; landmarkIdx < knownMarkerIDs.size(); ++landmarkIdx)
    {
        std::size_t stateIdx = systemSLAM.landmarkPositionIndex(landmarkIdx);
        Eigen::Vector3d rJNn = x.segment<3>(stateIdx);  // Marker center position
        
        // Transform center to camera frame
        Eigen::Vector3d rJcCc = Tnc.rotationMatrix.transpose() * (rJNn - rCNn);
        
        // Check if center is in front of camera
        if (rJcCc(2) > 0.0) {
            // Project center to image
            Eigen::Vector2d pixel = camera_.vectorToPixel(rJcCc);
            
            // Check if center is within image bounds
            if (pixel(0) >= 0 && pixel(0) < camera_.imageSize.width &&
                pixel(1) >= 0 && pixel(1) < camera_.imageSize.height) {
                visibleLandmarks_[landmarkIdx] = true;
            }
        }
    }
    
    // Associate detected markers with visible landmarks
    for (std::size_t detectionIdx = 0; detectionIdx < frameMarkerIDs_.size(); ++detectionIdx)
    {
        int detectedMarkerID = frameMarkerIDs_[detectionIdx];
        auto findLandmark = std::find(knownMarkerIDs.begin(), knownMarkerIDs.end(), detectedMarkerID);
        
        if (findLandmark != knownMarkerIDs.end())
        {
            std::size_t landmarkIdx = std::distance(knownMarkerIDs.begin(), findLandmark);
            idxFeatures_[landmarkIdx] = static_cast<int>(detectionIdx);
        }
    }
    
    return idxFeatures_;
}

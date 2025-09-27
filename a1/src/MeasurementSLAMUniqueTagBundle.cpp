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
#include "rotation.hpp"

MeasurementSLAMUniqueTagBundle::MeasurementSLAMUniqueTagBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera)
    : MeasurementSLAM(time, camera)
    , Y_(Y)
    , sigma_(1.0) // TODO: Assignment(s)
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

double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    // get all landmark indices
    std::vector<std::size_t> idxLandmarks;
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) 
    {
        idxLandmarks.push_back(i);
    }
    
    double logLik = 0.0;
    
    // use frozen associations 
    const std::vector<int> & idxFeatures = idxFeatures_;

    // count unassociated visible landmarks 
    std::size_t numUnassociated = 0;
    for (std::size_t j = 0; j < idxLandmarks.size(); ++j)
    {
        if (idxFeatures[j] < 0)  // landmark j is visible but not associated
        {
            numUnassociated++;
        }
    }

    // process associated landmarks:
    for (std::size_t j = 0; j < idxLandmarks.size(); ++j)  // loop over landmarks
    {
        if (idxFeatures[j] >= 0)  // landmark j is associated with feature i
        {
            // predict 8 corners for landmark j
            Eigen::MatrixXd J;
            Eigen::VectorXd predictedCorners = predictFeature(x, J, systemSLAM, idxLandmarks[j]);
            
            // extract the corresponding measurements from Y_
            int featureIdx = idxFeatures[j];
            Eigen::VectorXd measuredCorners = Y_.col(featureIdx);  // 8×1 vector
            
            // compute residual for all 8 corner coordinates  
            Eigen::VectorXd residual = measuredCorners - predictedCorners;
            
            // add gaussian likelihood for all 8 corners
            // for 8 coordinates total: log N(y; h(x), σ²I_8×8)
            double variance = sigma_ * sigma_;
            double squaredResidualNorm = residual.squaredNorm() / variance;
            double logNormalizationConstant = 4.0 * std::log(2.0 * M_PI * variance); // 8 measurements
            
            logLik += -0.5 * squaredResidualNorm - 0.5 * logNormalizationConstant;
        }
    }
    
    // add penalty for unassociated features: -|U| log |Y|  
    double imageArea = static_cast<double>(camera_.imageSize.width * camera_.imageSize.height);
    logLik -= static_cast<double>(numUnassociated) * std::log(imageArea);
    
    return logLik;
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    // initialize gradient
    g.resize(x.size());
    g.setZero();
    
    // get all landmark indices
    std::vector<std::size_t> idxLandmarks;
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) 
    {
        idxLandmarks.push_back(i);
    }
    
    // use frozen associations
    const std::vector<int> & idxFeatures = idxFeatures_;
    
    // compute gradient for associated landmarks only
    double measurementVariance = sigma_ * sigma_;
    
    for (std::size_t j = 0; j < idxLandmarks.size(); ++j)
    {
        if (idxFeatures[j] >= 0)  // landmark j is associated
        {
            // predict corners and get Jacobian
            Eigen::MatrixXd J;
            Eigen::VectorXd predictedCorners = predictFeature(x, J, systemSLAM, idxLandmarks[j]);
            
            // get measurements
            int featureIdx = idxFeatures[j];
            Eigen::VectorXd measuredCorners = Y_.col(featureIdx);
            
            // compute residual
            Eigen::VectorXd residual = measuredCorners - predictedCorners;
            
            g += J.transpose() * residual / measurementVariance;
        }
    }
    
    // penalty term has no gradient w.r.t. x (it's constant)
    
    return logLikelihood(x, system);
}

double MeasurementSLAMUniqueTagBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    const SystemSLAM & systemSLAM = dynamic_cast<const SystemSLAM &>(system);
    
    // initialize gradient and Hessian
    g.resize(x.size());
    g.setZero();
    H.resize(x.size(), x.size());
    H.setZero();
    
    // get all landmark indices
    std::vector<std::size_t> idxLandmarks;
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) 
    {
        idxLandmarks.push_back(i);
    }
    
    // use frozen associations
    const std::vector<int> & idxFeatures = idxFeatures_;
    
    double measurementVariance = sigma_ * sigma_;
    
    for (std::size_t j = 0; j < idxLandmarks.size(); ++j)
    {
        if (idxFeatures[j] >= 0)  // landmark j is associated
        {
            // predict corners and get Jacobian
            Eigen::MatrixXd J;
            Eigen::VectorXd predictedCorners = predictFeature(x, J, systemSLAM, idxLandmarks[j]);
            
            // get measurements
            int featureIdx = idxFeatures[j];
            Eigen::VectorXd measuredCorners = Y_.col(featureIdx);
            
            // compute residual
            Eigen::VectorXd residual = measuredCorners - predictedCorners;
            
            // add gradient contribution
            g += J.transpose() * residual / measurementVariance;
            
            H -= J.transpose() * J / measurementVariance;
        }
    }
    
    return logLikelihood(x, system, g);
}

void MeasurementSLAMUniqueTagBundle::update(SystemBase & system)
{
    SystemSLAM & systemSLAM = dynamic_cast<SystemSLAM &>(system);
    const std::size_t numDetectedMarkers = frameMarkerIDs_.size();
    
    for (std::size_t detectionIdx = 0; detectionIdx < numDetectedMarkers; ++detectionIdx)
    {
        const int currentMarkerID = frameMarkerIDs_[detectionIdx];
        
        // check if this detection was associated
        bool wasAssociated = std::any_of(idxFeatures_.begin(), idxFeatures_.end(),
            [detectionIdx](int association) { return association == static_cast<int>(detectionIdx); });
        
        if (!wasAssociated)
        {
            // add marker ID to our list
            knownMarkerIDs_.push_back(currentMarkerID);
            
            // initialize new landmark with proper prior
            const auto camPos = systemSLAM.cameraPositionDensity(camera_).mean();
            const auto camRpy = systemSLAM.cameraOrientationEulerDensity(camera_).mean();
            const Eigen::Matrix3d Rnc = rpy2rot(camRpy);
            const Eigen::Vector3d forward = Rnc * Eigen::Vector3d(0,0,0.30); // 30 cm forward

            Eigen::VectorXd landmarkMean(6);
            landmarkMean.head<3>() = camPos + forward;
            landmarkMean.tail<3>() = camRpy;

            // weak prior
            Eigen::Matrix<double,6,6> landmarkCov = Eigen::Matrix<double,6,6>::Zero();
            landmarkCov.diagonal() << 
                25.0, 25.0, 25.0,            // (5 m)^2 on position
                (30.0*M_PI/180.0)*(30.0*M_PI/180.0),  // (30°)^2 on roll
                (30.0*M_PI/180.0)*(30.0*M_PI/180.0),  // (30°)^2 on pitch
                (30.0*M_PI/180.0)*(30.0*M_PI/180.0);  // (30°)^2 on yaw

            GaussianInfo<double> newLandmarkDensity = GaussianInfo<double>::fromMoment(landmarkMean, landmarkCov);
            systemSLAM.density *= newLandmarkDensity;  // adds +6 dims for the landmark
            
            std::cout << "Initialized new landmark with ID: " << currentMarkerID << std::endl;
        }
    }
    
    // freeze associations before optimization
    std::vector<std::size_t> idxLandmarks;
    for (std::size_t i = 0; i < systemSLAM.numberLandmarks(); ++i) 
    {
        idxLandmarks.push_back(i);
    }
    
    // compute associations at current system state and freeze them
    idxFeatures_ = associate(systemSLAM, idxLandmarks);
    
    // measurement update
    Measurement::update(system);
}

// image feature location for a given landmark (ArUco marker) and Jacobian
Eigen::Matrix<double, 8, 1> MeasurementSLAMUniqueTagBundle::predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // get camera pose from state
    Pose<double> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // Get landmark pose from state (6D: position + orientation)
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3d rJNn = x.segment<3>(idx);       // Position
    Eigen::Vector3d Thetanj = x.segment<3>(idx + 3); // Orientation

    // Create landmark pose
    Pose<double> Tnj;
    Tnj.translationVector = rJNn;
    Tnj.rotationMatrix = rpy2rot(Thetanj);  // Rnj rotation matrix

    // ArUco marker corner positions in marker frame 
    double l_half = 166e-3 / 2.0; // Half edge length in meters
    std::vector<Eigen::Vector3d> rJcJj = {
        Eigen::Vector3d(-l_half, -l_half, 0.0),
        Eigen::Vector3d( l_half, -l_half, 0.0), 
        Eigen::Vector3d( l_half,  l_half, 0.0),
        Eigen::Vector3d(-l_half,  l_half, 0.0)  
    };

    // Compute predicted pixel coordinates for all 4 corners
    Eigen::Matrix<double, 8, 1> h; // 4 corners × 2 coordinates = 8 values
    
    // Initialize Jacobian ∂h/∂x 
    J.resize(8, x.size()); // 8 measurement outputs × state size
    J.setZero();

    // Extract body state components
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
    
    for (int i = 0; i < 4; ++i)
    {
        // Transform corner from marker frame to world frame: rJcNn = Rnj * rJcJj + rJNn
        Eigen::Vector3d rJcNn = Tnj.rotationMatrix * rJcJj[i] + rJNn;
        
        // Transform to camera coordinates
        Eigen::Vector3d rJcCn = Tnc.rotationMatrix.transpose() * (rJcNn - Tnc.translationVector);
        
        // Get pixel coordinates with Jacobian w.r.t. camera coordinates
        Eigen::Matrix23d J_camera;  
        Eigen::Vector2d rQOi = camera_.vectorToPixel(rJcCn, J_camera);
        
        // Store in output vector
        h.segment<2>(2*i) = rQOi;
        
        // Compute partial Jacobians for this corner
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

        // Apply body orientation derivative
        J.block<2, 1>(2*i, 9)  = J_camera * Rbc.transpose() * dRnb_dphi.transpose() * rJcNn_minus_rBNn;    
        J.block<2, 1>(2*i, 10) = J_camera * Rbc.transpose() * dRnb_dtheta.transpose() * rJcNn_minus_rBNn;  
        J.block<2, 1>(2*i, 11) = J_camera * Rbc.transpose() * dRnb_dpsi.transpose() * rJcNn_minus_rBNn;   

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
    const std::size_t ny = 8; // 4 corners × 2 coordinates

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

#include "association_util.h"
const std::vector<int> & MeasurementSLAMUniqueTagBundle::associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks)
{
    const std::size_t numDetectedMarkers = frameMarkerIDs_.size();
    assert(Y_.cols() == 4 * static_cast<int>(numDetectedMarkers));
    
    // Start with no associations
    idxFeatures_.assign(idxLandmarks.size(), -1);

    // Get current camera pose from system (once)
    GaussianInfo<double> cameraPosDensity = system.cameraPositionDensity(camera_);
    GaussianInfo<double> cameraOrientDensity = system.cameraOrientationEulerDensity(camera_);
    
    Pose<double> cameraPose;
    cameraPose.translationVector = cameraPosDensity.mean();
    cameraPose.rotationMatrix = rpy2rot(cameraOrientDensity.mean());

    // Process each detection
    for (std::size_t detectionIdx = 0; detectionIdx < numDetectedMarkers; ++detectionIdx) 
    {
        const int currentMarkerID = frameMarkerIDs_[detectionIdx];
        
        // Find if this marker ID exists in our known landmarks
        auto searchResult = std::find(knownMarkerIDs_.begin(), knownMarkerIDs_.end(), currentMarkerID);
        if (searchResult != knownMarkerIDs_.end()) 
        {
            std::size_t landmarkIdx = static_cast<std::size_t>(std::distance(knownMarkerIDs_.begin(), searchResult));
            
            // FOV check: get landmark position and check if visible
            Eigen::VectorXd systemState = system.density.mean();
            std::size_t posIdx = system.landmarkPositionIndex(landmarkIdx);
            Eigen::Vector3d landmarkPos = systemState.segment<3>(posIdx);
            cv::Vec3d landmarkPos_cv(landmarkPos[0], landmarkPos[1], landmarkPos[2]);
            
            if (camera_.isWorldWithinFOV(landmarkPos_cv, cameraPose)) 
            {
                // Find this landmark in the candidate list and associate
                auto candidateIt = std::find(idxLandmarks.begin(), idxLandmarks.end(), landmarkIdx);
                if (candidateIt != idxLandmarks.end()) 
                {
                    std::size_t candidateIdx = static_cast<std::size_t>(std::distance(idxLandmarks.begin(), candidateIt));
                    idxFeatures_[candidateIdx] = static_cast<int>(detectionIdx);
                }
            }
        }
        // New markers (not in knownMarkerIDs_) will be handled in update()
    }
    
    return idxFeatures_;
}


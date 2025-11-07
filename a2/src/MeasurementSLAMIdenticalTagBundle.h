#ifndef MEASUREMENTIDENTICALTAGBUNDLE_H
#define MEASUREMENTIDENTICALTAGBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"

class MeasurementSLAMIdenticalTagBundle : public MeasurementSLAM
{
public:
    MeasurementSLAMIdenticalTagBundle(double time, const cv::Mat & image, const Camera & camera, int maxNumMarkers = 100);
    MeasurementSLAM * clone() const override;
    
    // Get the visualization image with detected markers drawn
    const cv::Mat& getVisualizationImage() const { return visualizationImage_; }
    
    // Get the measurement matrix (for checking if any markers were detected)
    const Eigen::Matrix<double, 8, Eigen::Dynamic>& getY() const { return Y_; }
    
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    template <typename Scalar> Eigen::Matrix<Scalar, 8, 1> predictFeature(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, std::size_t idxLandmark) const;
    Eigen::Matrix<double, 8, 1> predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, std::size_t idxLandmark) const;
    virtual GaussianInfo<double> predictFeatureDensity(const SystemVisualNav & system, std::size_t idxLandmark) const override;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const override;

    virtual const std::vector<int> & associate(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) override;
    void setFrameMarkerIDs(const std::vector<int>& markerIDs) { frameMarkerIDs_ = markerIDs; }
    const std::vector<int>& getAssociations() const { return idxFeatures_; }
    const std::vector<size_t>& getVisibleLandmarks() const { return visibleLandmarks_; }
    
    // Templated log-likelihood for autodiff support (public for testing)
    template <typename Scalar>
    Scalar logLikelihoodTemplated(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system) const;
protected:
    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 8, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_;                                  // Feature error standard deviation (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
    std::vector<int> frameMarkerIDs_;               // Marker IDs for each frame
    std::vector<size_t> visibleLandmarks_;            // Visibility status for each landmark
    cv::Mat visualizationImage_;                     // Image with detected markers drawn
};

// Image feature location for a given landmark (ArUco marker with 4 corners)
template <typename Scalar>
Eigen::Matrix<Scalar, 8, 1> MeasurementSLAMIdenticalTagBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, std::size_t idxLandmark) const
{
    // Obtain camera pose from state
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // Obtain landmark pose from state (position + orientation)
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rJNn = x.template segment<3>(idx);     // Position
    Eigen::Vector3<Scalar> Thetanj = x.template segment<3>(idx + 3); // Orientation

    // Create landmark pose
    Pose<Scalar> Tnj;
    Tnj.translationVector = rJNn;
    Tnj.rotationMatrix = rpy2rot(Thetanj);  // Rnj rotation matrix

    // ArUco marker corner positions in marker frame 
    Scalar l_half = Scalar(0.166 / 2.0); // Half edge length in meters
    std::vector<Eigen::Vector3<Scalar>> rJcJj = {
        Eigen::Vector3<Scalar>(-l_half, l_half, Scalar(0)),
        Eigen::Vector3<Scalar>( l_half, l_half, Scalar(0)), 
        Eigen::Vector3<Scalar>( l_half,  -l_half, Scalar(0)),
        Eigen::Vector3<Scalar>(-l_half,  -l_half, Scalar(0))  
    };

    // Predict pixel coordinates for all 4 corners
    Eigen::Matrix<Scalar, 8, 1> h; // 4 corners × 2 coordinates = 8 values
    
    for (int j = 0; j < 4; ++j)
    {
        // Transform corner from marker frame to world frame: rJcNn = Rnj * rJcJj + rJNn
        Eigen::Vector3<Scalar> rJcNn = Tnj.rotationMatrix * rJcJj[j] + rJNn;
        
        // Transform to camera coordinates
        Eigen::Vector3<Scalar> rJcCn = Tnc.rotationMatrix.transpose() * (rJcNn - Tnc.translationVector);
        
        // Project to pixels
        Eigen::Vector2<Scalar> rQOi = camera_.vectorToPixel(rJcCn); 
        
        // Store in output vector
        h.template segment<2>(2*j) = rQOi;
    }
    
    return h;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementSLAMIdenticalTagBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const
{
    assert(x.size() == system.density.dim());

    const std::size_t & nL = idxLandmarks.size();
    Eigen::VectorX<Scalar> h(8*nL); // 8 coordinates per landmark (4 corners × 2 coordinates)
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::Matrix<Scalar, 8, 1> corners = predictFeature(x, system, idxLandmarks[i]);
        // Set 8 elements of h for this landmark
        h.template segment<8>(8*i) = corners;
    }
    return h;
}

// Templated log-likelihood for autodiff support
template <typename Scalar>
Scalar MeasurementSLAMIdenticalTagBundle::logLikelihoodTemplated(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system) const
{
    const SystemVisualNav & systemVisualNav = dynamic_cast<const SystemVisualNav &>(system);
    
    Scalar logLik = Scalar(0.0);
    
    // Count unassociated landmarks for penalty term
    int numUnassociated = 0;
    for (int assoc : idxFeatures_) {
        if (assoc < 0) numUnassociated++;
    }
    
    // Create measurement noise model
    Eigen::MatrixX<Scalar> S = Scalar(sigma_) * Eigen::MatrixX<Scalar>::Identity(2, 2);
    GaussianInfo<Scalar> measurementModel = GaussianInfo<Scalar>::fromSqrtMoment(S);
    
    // Sum log-likelihoods over all associated feature/landmark pairs
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        if (idxFeatures_[j] >= 0) {  // This landmark is associated
            size_t landmarkIdx = visibleLandmarks_[j];
            int detectionIdx = idxFeatures_[j];
            
            // Predict all 4 corners for this landmark
            Eigen::Matrix<Scalar, 8, 1> h_pred = predictFeature(x, systemVisualNav, landmarkIdx);
            
            // Sum over all 4 corners
            for (int c = 0; c < 4; ++c) {
                // Get measured corner position (always double)
                Eigen::Vector2d y_ic = Y_.block<2, 1>(2*c, detectionIdx);
                
                // Get predicted corner position
                Eigen::Vector2<Scalar> h_ic = h_pred.template segment<2>(2*c);
                Eigen::Vector2<Scalar> residual;
                residual(0) = Scalar(y_ic(0)) - h_ic(0);
                residual(1) = Scalar(y_ic(1)) - h_ic(1);
                
                // Use our tested GaussianInfo log function
                logLik += measurementModel.log(residual);
            }
        }
    }
    
    // Add penalty term for unassociated visible landmarks (not critical for this scenario)
    double imageArea = camera_.imageSize.width * camera_.imageSize.height;
    logLik -= Scalar(4.0 * numUnassociated * std::log(imageArea));
    
    return logLik;
}

#endif

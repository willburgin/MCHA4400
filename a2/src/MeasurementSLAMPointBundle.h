#ifndef MEASUREMENTSLAMPOINTBUNDLE_H
#define MEASUREMENTSLAMPOINTBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"

class MeasurementPointBundle : public MeasurementSLAM
{
public:
    MeasurementPointBundle(double time, const cv::Mat & image, const Camera & camera, int maxNumFeatures = 100);
    MeasurementSLAM * clone() const override;
    
    // Get the visualization image with detected features drawn
    const cv::Mat& getVisualizationImage() const { return visualizationImage_; }
    
    // Get the measurement matrix (for checking if any features were detected)
    const Eigen::Matrix<double, 2, Eigen::Dynamic>& getY() const { return Y_; }
    
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    template <typename Scalar> Eigen::Vector2<Scalar> predictFeature(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, std::size_t idxLandmark) const;
    Eigen::Vector2d predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, std::size_t idxLandmark) const;
    GaussianInfo<double> predictFeatureDensity(const SystemVisualNav & system, std::size_t idxLandmark) const;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const;
    GaussianInfo<double> predictFeatureBundleDensity(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const;

    const std::vector<int> & associate(const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks);
    template <typename Scalar>
    Scalar logLikelihoodTemplated(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system) const;
    
    // Accessors for plotting
    const std::vector<std::size_t>& getVisibleLandmarks() const { return visibleLandmarks_; }
    const std::vector<int>& getAssociations() const { return idxFeatures_; }

protected:
    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 2, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_;                                  // Feature error standard deviation (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
    std::vector<std::size_t> visibleLandmarks_;     // Visible landmarks
    cv::Mat visualizationImage_;                     // Image with detected features drawn
};

// Image feature location for a given landmark
template <typename Scalar>
Eigen::Vector2<Scalar> MeasurementPointBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, std::size_t idxLandmark) const
{
    // Obtain camera pose from state
    Pose<Scalar> Tnc;
    Tnc.translationVector = SystemVisualNav::cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = SystemVisualNav::cameraOrientation(camera_, x); // Rnc

    // Obtain landmark position from state
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rPNn = x.template segment<3>(idx);

    // Camera vector
    Eigen::Vector3<Scalar> rPCc;
    rPCc = Tnc.rotationMatrix.transpose() * (rPNn - Tnc.translationVector);

    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi;
    rQOi = camera_.vectorToPixel(rPCc);  
    return rQOi;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementPointBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system, const std::vector<std::size_t> & idxLandmarks) const
{
    assert(x.size() == system.density.dim());

    const std::size_t & nL = idxLandmarks.size();
    Eigen::VectorX<Scalar> h(2*nL);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::Vector2<Scalar> rQOi = predictFeature(x, system, idxLandmarks[i]);
        h.template segment<2>(2*i) = rQOi;
    }
    return h;
}

// Templated log-likelihood for autodiff support
template <typename Scalar>
Scalar MeasurementPointBundle::logLikelihoodTemplated(const Eigen::VectorX<Scalar> & x, const SystemVisualNav & system) const
{
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
            int detectionIdx = idxFeatures_[j];
            size_t landmarkIdx = visibleLandmarks_[j]; 
            // Predict single point feature for this landmark
            Eigen::Vector2<Scalar> h_pred = predictFeature(x, system, landmarkIdx);
            
            // Get measured point position (always double)
            Eigen::Vector2d y_i = Y_.col(detectionIdx);
            
            // Compute residual
            Eigen::Vector2<Scalar> residual;
            residual(0) = Scalar(y_i(0)) - h_pred(0);
            residual(1) = Scalar(y_i(1)) - h_pred(1);
            
            // Add log-likelihood contribution
            logLik += measurementModel.log(residual);
        }
    }
    
    // Add penalty term for unassociated visible landmarks
    double imageArea = camera_.imageSize.width * camera_.imageSize.height;
    logLik -= Scalar(numUnassociated * std::log(imageArea));
    
    return logLik;
}

#endif

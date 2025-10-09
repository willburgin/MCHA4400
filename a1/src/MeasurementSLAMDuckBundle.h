#ifndef MEASUREMENTSLAMDUCKBUNDLE_H
#define MEASUREMENTSLAMDUCKBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"

class MeasurementDuckBundle : public MeasurementSLAM
{
public:
    MeasurementDuckBundle(double time, const Eigen::Matrix<double, 3, Eigen::Dynamic> & Y, const Camera & camera);
    MeasurementSLAM * clone() const override;
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    template <typename Scalar> Eigen::Vector2<Scalar> predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const;
    Eigen::Vector2d predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const;
    virtual GaussianInfo<double> predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const override;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const override;

    virtual const std::vector<int> & associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) override;
    
    // Accessors for associations and visibility (used by Plot)
    const std::vector<int>& getAssociations() const { return idxFeatures_; }
    const std::vector<size_t>& getVisibleLandmarks() const { return visibleLandmarks_; }
    
    // Templated log-likelihood for autodiff support (public for testing)
    template <typename Scalar>
    Scalar logLikelihoodTemplated(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system) const;
protected:
    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 3, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_c_;                                  // Feature error standard deviation for centroid (in pixels)
    double sigma_a_;                                  // Feature error standard deviation for area (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
    std::vector<size_t> visibleLandmarks_;              // Landmarks associated with visible features
};

// Image feature location for a given landmark
template <typename Scalar>
Eigen::Vector2<Scalar> MeasurementDuckBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Obtain camera pose from state
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // Obtain landmark position from state
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rPNn = x.template segment<3>(idx);

    // Camera vector
    Eigen::Vector3<Scalar> rPCc;
    // TODO: Lab 8
    rPCc = Tnc.rotationMatrix.transpose() * (rPNn - Tnc.translationVector);

    // Pixel coordinates
    Eigen::Vector2<Scalar> rQOi;
    // TODO: Lab 8
    rQOi = camera_.vectorToPixel(rPCc);  
    return rQOi;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementDuckBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
{
    assert(x.size() == system.density.dim());

    const std::size_t & nL = idxLandmarks.size();
    Eigen::VectorX<Scalar> h(2*nL);
    for (std::size_t i = 0; i < nL; ++i)
    {
        Eigen::Vector2<Scalar> rQOi = predictFeature(x, system, idxLandmarks[i]);
        // Set pair of elements of h
        // TODO: Lab 9
        h.template segment<2>(2*i) = rQOi;
    }
    return h;
}

template <typename Scalar>
Scalar MeasurementDuckBundle::logLikelihoodTemplated(
    const Eigen::VectorX<Scalar> & x, const SystemSLAM & system) const
{
    Scalar logLik = Scalar(0.0);
    const Scalar fx = Scalar(camera_.cameraMatrix.at<double>(0, 0));
    const Scalar fy = Scalar(camera_.cameraMatrix.at<double>(1, 1));
    const Scalar duck_radius = Scalar(0.03);
    
    // Measurement noise models
    Eigen::MatrixX<Scalar> S_centroid = Scalar(sigma_c_) * Eigen::MatrixX<Scalar>::Identity(2, 2);
    GaussianInfo<Scalar> centroidModel = GaussianInfo<Scalar>::fromSqrtMoment(S_centroid);

    // Area likelihood
    Eigen::MatrixX<Scalar> S_area = Scalar(sigma_a_) * Eigen::MatrixX<Scalar>::Identity(1, 1);
    GaussianInfo<Scalar> areaModel = GaussianInfo<Scalar>::fromSqrtMoment(S_area);
    
    // Sum over all associated duck measurements
    for (std::size_t j = 0; j < idxFeatures_.size(); ++j) {
        if (idxFeatures_[j] >= 0) {
            size_t landmarkIdx = visibleLandmarks_[j];

            int detectionIdx = idxFeatures_[j];
            
            // CENTROID LIKELIHOOD
            // Predict centroid in pixels
            Eigen::Vector2<Scalar> predicted_centroid = predictFeature(x, system, landmarkIdx);

            // Get measured centroid (extract from Y_)
            Eigen::Vector2d measured_centroid = Y_.block<2, 1>(0, detectionIdx); // First 2 rows
            
            // Compute residual
            Eigen::Vector2<Scalar> centroid_residual;
            centroid_residual(0) = Scalar(measured_centroid(0)) - predicted_centroid(0);
            centroid_residual(1) = Scalar(measured_centroid(1)) - predicted_centroid(1);
            
            logLik += centroidModel.log(centroid_residual);
            
            // AREA LIKELIHOOD
            // Predicted area
            std::size_t idx = system.landmarkPositionIndex(landmarkIdx);
            Eigen::Vector3<Scalar> rJNn = x.template segment<3>(idx);
            Eigen::Vector3<Scalar> rCNn = system.cameraPosition(camera_, x);
            Scalar distance_squared = (rCNn - rJNn).squaredNorm();
            Scalar predicted_area = (Scalar(fx) * Scalar(fy) * Scalar(M_PI) * Scalar(duck_radius * duck_radius)) / distance_squared;
            
            // Get measured area (extract from Y_)
            double measured_area = Y_(2, detectionIdx); // Third row
            
            // Compute residual
            Eigen::Matrix<Scalar, 1, 1> area_residual;
            area_residual(0) = Scalar(measured_area) - predicted_area;
            
            logLik += areaModel.log(area_residual);
        }
    }
    return logLik;
}

#endif

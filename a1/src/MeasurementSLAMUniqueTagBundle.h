#ifndef MEASUREMENTSLAMUNIQUETAGBUNDLE_H
#define MEASUREMENTSLAMUNIQUETAGBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"

class MeasurementSLAMUniqueTagBundle : public MeasurementSLAM
{
public:
    MeasurementSLAMUniqueTagBundle(double time, const Eigen::Matrix<double, 8, Eigen::Dynamic> & Y, const Camera & camera);
    MeasurementSLAM * clone() const override;
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    template <typename Scalar> Eigen::Matrix<Scalar, 8, 1> predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const;
    Eigen::Matrix<double, 8, 1> predictFeature(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, std::size_t idxLandmark) const;
    virtual GaussianInfo<double> predictFeatureDensity(const SystemSLAM & system, std::size_t idxLandmark) const override;

    template <typename Scalar> Eigen::VectorX<Scalar> predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    Eigen::VectorXd predictFeatureBundle(const Eigen::VectorXd & x, Eigen::MatrixXd & J, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const;
    virtual GaussianInfo<double> predictFeatureBundleDensity(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const override;

    virtual const std::vector<int> & associate(const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) override;
    void setFrameMarkerIDs(const std::vector<int>& markerIDs) { frameMarkerIDs_ = markerIDs; }
    const std::vector<int>& getAssociations() const { return idxFeatures_; }
    const std::vector<bool>& getVisibility() const { return visibleLandmarks_; }
protected:
    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 8, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_;                                  // Feature error standard deviation (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
    std::vector<int> frameMarkerIDs_;               // Marker IDs for each frame
    std::vector<bool> visibleLandmarks_;            // Visibility status for each landmark
private:
    template <typename Scalar>
    Scalar logLikelihoodTemplated(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system) const;
};

// Image feature location for a given landmark (ArUco marker with 4 corners)
template <typename Scalar>
Eigen::Matrix<Scalar, 8, 1> MeasurementSLAMUniqueTagBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const
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
    Eigen::Matrix<Scalar, 8, 1> h; // 4 corners × 2 coordinates = 8 values (
    
    for (int j = 0; j < 4; ++j)
    {
        // Transform corner from marker frame to world frame: rJcNn = Rnj * rJcJj + rJNn
        Eigen::Vector3<Scalar> rJcNn = Tnj.rotationMatrix * rJcJj[j] + rJNn;
        
        // Transform to camera coordinates
        Eigen::Vector3<Scalar> rJcCn = Tnc.rotationMatrix.transpose() * (rJcNn - Tnc.translationVector);
        
        // Project to pixels
        Eigen::Vector2<Scalar> rQOi = camera_.vectorToPixel(rJcCn); // position vector from image origin to corner j of landmark i
        
        // Store in output vector
        h.template segment<2>(2*j) = rQOi;
    }
    
    return h;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementSLAMUniqueTagBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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

#endif

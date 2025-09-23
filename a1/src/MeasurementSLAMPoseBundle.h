#ifndef MEASUREMENTSLAMPOSEBUNDLE_H
#define MEASUREMENTSLAMPOSEBUNDLE_H

#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Camera.h"
#include "Pose.hpp"
#include "MeasurementSLAM.h"

class MeasurementPoseBundle : public MeasurementSLAM
{
public:
    MeasurementPoseBundle(double time, const Eigen::Matrix<double, 2, Eigen::Dynamic> & Y, const Camera & camera);
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
protected:
    virtual void update(SystemBase & system) override;
    Eigen::Matrix<double, 2, Eigen::Dynamic> Y_;    // Feature bundle
    double sigma_;                                  // Feature error standard deviation (in pixels)
    std::vector<int> idxFeatures_;                  // Features associated with visible landmarks
};

// Image feature location for a given landmark (ArUco marker with 4 corners)
template <typename Scalar>
Eigen::Matrix<Scalar, 8, 1> MeasurementPoseBundle::predictFeature(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, std::size_t idxLandmark) const
{
    // Obtain camera pose from state
    Pose<Scalar> Tnc;
    Tnc.translationVector = system.cameraPosition(camera_, x); // rCNn
    Tnc.rotationMatrix = system.cameraOrientation(camera_, x); // Rnc

    // Obtain landmark pose from state (6D: position + orientation)
    std::size_t idx = system.landmarkPositionIndex(idxLandmark);
    Eigen::Vector3<Scalar> rPNn = x.template segment<3>(idx);     // Position
    Eigen::Vector3<Scalar> Thetapn = x.template segment<3>(idx + 3); // Orientation

    // Create landmark pose
    Pose<Scalar> Tpn;
    Tpn.translationVector = rPNn;
    Tpn.rotationMatrix = rpy2rot(Thetapn);  // RPn rotation matrix

    // ArUco marker corner positions in marker frame (assume 166mm edge length)
    Scalar l_half = Scalar(166e-3 / 2.0); // Half edge length in meters
    std::vector<Eigen::Vector3<Scalar>> rPij_J = {
        Eigen::Vector3<Scalar>(-l_half, -l_half, Scalar(0)), // Bottom-left
        Eigen::Vector3<Scalar>( l_half, -l_half, Scalar(0)), // Bottom-right  
        Eigen::Vector3<Scalar>( l_half,  l_half, Scalar(0)), // Top-right
        Eigen::Vector3<Scalar>(-l_half,  l_half, Scalar(0))  // Top-left
    };

    // Predict pixel coordinates for all 4 corners
    Eigen::Matrix<Scalar, 8, 1> h; // 4 corners × 2 coordinates = 8 values
    
    for (int i = 0; i < 4; ++i)
    {
        // Transform corner from marker frame to world frame: rPij/N = RPn * rPij/J + rPNn
        Eigen::Vector3<Scalar> rPij_N = Tpn.rotationMatrix * rPij_J[i] + Tpn.translationVector;
        
        // Transform to camera coordinates
        Eigen::Vector3<Scalar> rPij_C = Tnc.rotationMatrix.transpose() * (rPij_N - Tnc.translationVector);
        
        // Project to pixels
        Eigen::Vector2<Scalar> rQij = camera_.vectorToPixel(rPij_C); // position vector from image origin to corner j of landmark i
        
        // Store in output vector
        h.template segment<2>(2*i) = rQij;
    }
    
    return h;
}

// Image feature locations for a bundle of landmarks
template <typename Scalar>
Eigen::VectorX<Scalar> MeasurementPoseBundle::predictFeatureBundle(const Eigen::VectorX<Scalar> & x, const SystemSLAM & system, const std::vector<std::size_t> & idxLandmarks) const
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

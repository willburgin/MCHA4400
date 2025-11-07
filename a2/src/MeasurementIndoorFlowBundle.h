#ifndef MEASUREMENTINDOORFLOWBUNDLE_H
#define MEASUREMENTINDOORFLOWBUNDLE_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "SystemEstimator.h"
#include "Pose.hpp"
#include "Camera.h"
#include "Measurement.h"
#include <iostream>

class MeasurementIndoorFlowBundle : public Measurement
{
public:
    MeasurementIndoorFlowBundle(double time, const Camera & camera, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1);
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const override;
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const override;

    // Helper functions for log likelihood and visualisation
    template <typename Scalar> Eigen::Matrix<Scalar, 3, Eigen::Dynamic> predictFlowImpl(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pk) const;    template <typename Scalar> Scalar logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const;
    Eigen::Matrix<double, 2, Eigen::Dynamic> predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const;
    GaussianInfo<double> predictFeatureDensity(const SystemEstimator & system, std::size_t pointIdx) const;

    // Note: costOdometry is used only in Lab 11.
    //       Assignment 2 uses costJointDensity instead.
    template <typename Scalar> Scalar costOdometryImpl(const Eigen::VectorX<Scalar> & etak, const Eigen::VectorXd & etakm1) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g) const;
    double costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g, Eigen::MatrixXd & H) const;

    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedPreviousFeatures() const;
    const Eigen::Matrix<double, 2, Eigen::Dynamic> & trackedCurrentFeatures() const;
    const std::vector<unsigned char> & inlierMask() const;
    
    // void update(SystemBase & system) override;

protected:
    const Camera & camera_;

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOikm1_;      // Measured features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_;        // Measured features for current frame

    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOikm1_;   // Undistorted features for previous frame
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_;     // Undistorted features for current frame

    std::vector<unsigned char> mask_;                       // Inlier mask

    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1_;         // Inlier undistorted homogeneous points in previous frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_;           // Inlier undistorted homogeneous points in current frame

    double sigma_;                                          // Feature error standard deviation (in pixels)
};

template <typename Scalar>
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MeasurementIndoorFlowBundle::predictFlowImpl(
    const Eigen::VectorX<Scalar> & x, 
    const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pkm1, 
    const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pk) const
{
    // this function is useless for indoor flow bundle
    return Eigen::Matrix<Scalar, 3, Eigen::Dynamic>::Zero(3, pk.cols());
}

template <typename Scalar>
Scalar MeasurementIndoorFlowBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
{
    assert(pkm1_.cols() == pk_.cols());
    int m = pkm1_.cols();

    // Extract translation for fundamental matrix computation
    Eigen::Vector3<Scalar> rBNn_k = x.template segment<3>(6);
    Eigen::Vector3<Scalar> rBNn_km1 = x.template segment<3>(12);
    Eigen::Vector3<Scalar> translation = rBNn_km1 - rBNn_k;
    // std::cout << "translation: " << translation.transpose() << std::endl;
    
    // ===== COMPUTE FUNDAMENTAL MATRIX F =====
    Eigen::Vector3<Scalar> thetaNB_k = x.template segment<3>(9);
    Eigen::Vector3<Scalar> thetaNB_km1 = x.template segment<3>(15);
    
    Eigen::Matrix3<Scalar> Rnb_k = rpy2rot(thetaNB_k);
    Eigen::Matrix3<Scalar> Rnb_km1 = rpy2rot(thetaNB_km1);
    
    Eigen::Matrix3d Rbc = camera_.Tbc.rotationMatrix;
    Eigen::Matrix3<Scalar> Rbc_scalar = Rbc.cast<Scalar>();
    
    Eigen::Matrix3<Scalar> Rnc_k = Rnb_k * Rbc_scalar;
    Eigen::Matrix3<Scalar> Rnc_km1 = Rnb_km1 * Rbc_scalar;
    
    Eigen::Matrix3d K;
    cv::cv2eigen(camera_.cameraMatrix, K);
    Eigen::Matrix3<Scalar> K_scalar = K.cast<Scalar>();
    
    Scalar fx = K_scalar(0, 0);
    Scalar fy = K_scalar(1, 1);
    Scalar cx = K_scalar(0, 2);
    Scalar cy = K_scalar(1, 2);
    
    Eigen::Matrix3<Scalar> K_inv;
    K_inv << Scalar(1)/fx, Scalar(0), -cx/fx,
             Scalar(0), Scalar(1)/fy, -cy/fy,
             Scalar(0), Scalar(0), Scalar(1);
    
    Eigen::Matrix3<Scalar> S_t;
    S_t << Scalar(0), -translation(2), translation(1),
           translation(2), Scalar(0), -translation(0),
           -translation(1), translation(0), Scalar(0);
    
    Eigen::Matrix3<Scalar> F = K_inv.transpose() * Rnc_k.transpose() * S_t * Rnc_km1 * K_inv;
    
    // Create 1D measurement model: N(0, sigma^2)
    Eigen::Matrix<Scalar, 1, 1> S;
    S(0, 0) = Scalar(sigma_);
    GaussianInfo<Scalar> measurementModel = GaussianInfo<Scalar>::fromSqrtMoment(S);
    
    Scalar logLik = Scalar(0);
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pkm1_scalar = pkm1_.template cast<Scalar>();
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_scalar = pk_.template cast<Scalar>();
    
    for (int j = 0; j < m; ++j) 
    {
        // Compute epipolar line: l[k] = F * p[k-1]
        Eigen::Vector3<Scalar> l = F * pkm1_scalar.col(j);
        
        // Normalize line: n_l(l) = l / sqrt(a^2 + b^2)
        Scalar l_norm = sqrt(l(0)*l(0) + l(1)*l(1));
        
        Eigen::Vector3<Scalar> n_l = l / l_norm;
        
        // Normalize point: n_p(p[k]) = p[k] / p[k](2)
        Eigen::Vector3<Scalar> n_p = pk_scalar.col(j) / pk_scalar(2, j);
        
        // Compute scalar measurement: n_p(p[k])^T * n_l(F * p[k-1])
        Scalar measurement_scalar = n_p.dot(n_l);
        
        // Wrap in 1D vector for GaussianInfo::log()
        Eigen::Matrix<Scalar, 1, 1> measurement;
        measurement(0, 0) = measurement_scalar;
        
        // Evaluate log N(measurement_scalar; 0, sigma^2)
        logLik += measurementModel.log(measurement);
    }
    
    return logLik;
}

// Note: costOdometry is used only in Lab 11, not Assignment 2.
template <typename Scalar>
Scalar MeasurementIndoorFlowBundle::costOdometryImpl(const Eigen::VectorX<Scalar> & etak, const Eigen::VectorXd & etakm1) const
{
    Eigen::VectorX<Scalar> x(18);
    x.template segment<6>(6) = etak;
    x.template segment<6>(12) = etakm1;
    return -logLikelihoodImpl(x);
}

#endif

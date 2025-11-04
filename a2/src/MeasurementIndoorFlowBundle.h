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
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MeasurementIndoorFlowBundle::predictFlowImpl(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pk) const
{
    assert(x.rows() >= 18);
    assert(x.cols() == 1);
    assert(pkm1.cols() == pk.cols());
    
    return pk_hat;
}

template <typename Scalar>
Scalar MeasurementIndoorFlowBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
{
    assert(pkm1_.cols() == pk_.cols());
    
    // Cast measured features to Scalar type for autodiff
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pkm1_scalar = pkm1_.template cast<Scalar>();
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_scalar = pk_.template cast<Scalar>();
    
    // Predict flow in homogeneous coordinates
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1_scalar, pk_scalar);
    
    // Apply r() to both predicted AND measured features
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> rQbarOik_hat = 
        pk_hat.template topRows<2>().array().rowwise() / pk_hat.row(2).array();
    
    Eigen::Matrix<Scalar, 2, Eigen::Dynamic> rQbarOik_measured = 
        pk_scalar.template topRows<2>().array().rowwise() / pk_scalar.row(2).array();
    
    Scalar logLik = Scalar(0);
    
    // Measurement covariance - cast sigma_ to Scalar
    Eigen::Matrix2<Scalar> S = Scalar(sigma_) * Eigen::Matrix2<Scalar>::Identity();
    GaussianInfo<Scalar> measurementModel = GaussianInfo<Scalar>::fromSqrtMoment(S);
    
    int nFeatures = pk_.cols();
    for (int i = 0; i < nFeatures; ++i) {
        // Residual: measured - predicted
        Eigen::Vector2<Scalar> residual = 
            rQbarOik_measured.col(i) - rQbarOik_hat.col(i);
        
        // Evaluate log likelihood
        logLik += measurementModel.log(residual);
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

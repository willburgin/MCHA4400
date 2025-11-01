#ifndef MEASUREMENTOUTDOORFLOWBUNDLE_H
#define MEASUREMENTOUTDOORFLOWBUNDLE_H

#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "SystemEstimator.h"
#include "Pose.hpp"
#include "Camera.h"
#include "Measurement.h"

class MeasurementOutdoorFlowBundle : public Measurement
{
public:
    MeasurementOutdoorFlowBundle(double time, const Camera & camera, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1);
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
    
    void update(SystemBase & system) override;

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
Eigen::Matrix<Scalar, 3, Eigen::Dynamic> MeasurementOutdoorFlowBundle::predictFlowImpl(const Eigen::VectorX<Scalar> & x, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pkm1, const Eigen::Matrix<Scalar, 3, Eigen::Dynamic> & pk) const
{
    assert(x.rows() >= 18);
    assert(x.cols() == 1);
    assert(pkm1.cols() == pk.cols());
    
    Eigen::Matrix<Scalar, 3, Eigen::Dynamic> pk_hat(3, pkm1.cols());
    // TODO: Lab 11

    // extract states - body pose same as camera pose
    Eigen::Vector3<Scalar> rBNn_k = x.template segment<3>(6);
    Eigen::Vector3<Scalar> thetaNB_k = x.template segment<3>(9);

    Eigen::Vector3<Scalar> rBNn_km1 = x.template segment<3>(12);
    Eigen::Vector3<Scalar> thetaNB_km1 = x.template segment<3>(15);

    // Get rotation matrices (body frame)
    Eigen::Matrix3<Scalar> Rnb_k = rpy2rot(thetaNB_k);
    Eigen::Matrix3<Scalar> Rnb_km1 = rpy2rot(thetaNB_km1);

    // Body-to-camera transformation
    Eigen::Matrix3d Rbc = camera_.Tbc.rotationMatrix;
    Eigen::Matrix3<Scalar> Rbc_scalar = Rbc.cast<Scalar>();

    // Camera orientations in nav frame
    Eigen::Matrix3<Scalar> Rnc_k = Rnb_k * Rbc_scalar;
    Eigen::Matrix3<Scalar> Rnc_km1 = Rnb_km1 * Rbc_scalar;

    // Camera positions in nav frame (same as body since they coincide)
    Eigen::Vector3<Scalar> rCNn_k = rBNn_k;
    Eigen::Vector3<Scalar> rCNn_km1 = rBNn_km1;

    // Camera parameters
    Eigen::Matrix3d K;
    cv::cv2eigen(camera_.cameraMatrix, K);
    Eigen::Matrix3<Scalar> K_scalar = K.cast<Scalar>();

    // Compute K^-1 directly
    Scalar fx = K_scalar(0, 0);
    Scalar fy = K_scalar(1, 1);
    Scalar cx = K_scalar(0, 2);
    Scalar cy = K_scalar(1, 2);

    Eigen::Matrix3<Scalar> K_inv;
    K_inv << Scalar(1)/fx,  Scalar(0),     -cx/fx,
            Scalar(0),     Scalar(1)/fy,  -cy/fy,
            Scalar(0),     Scalar(0),      Scalar(1);

    // e3 vector (pointing down in NED frame)
    Eigen::Vector3<Scalar> e3(Scalar(0), Scalar(0), Scalar(1));

    // Translation (camera positions)
    Eigen::Vector3<Scalar> translation = rCNn_km1 - rCNn_k;
    Scalar altitude_km1 = e3.dot(rCNn_km1); 

    // H_z: Ground plane homography (equation 3)
    Eigen::Matrix3<Scalar> H_z = K_scalar * Rnc_k.transpose() * 
        (Eigen::Matrix3<Scalar>::Identity() - (translation * e3.transpose()) / altitude_km1) * 
        Rnc_km1 * K_inv;

    // H_inf: Infinity homography (equation 4)
    Eigen::Matrix3<Scalar> H_inf = K_scalar * Rnc_k.transpose() * Rnc_km1 * K_inv;
    
    for (int i = 0; i < pkm1.cols(); ++i) {
        Eigen::Vector3<Scalar> ray_direction = Rnc_k * K_inv * pk.col(i).template cast<Scalar>();
        Scalar vertical_component = e3.dot(ray_direction);
        
        // Apply appropriate homography to pkm1 to predict pk_hat
        if (vertical_component > Scalar(0)) {
            // Below horizon - use ground plane homography
            pk_hat.col(i) = H_z * pkm1.col(i).template cast<Scalar>();
        } else {
            // Above horizon - use infinity homography
            pk_hat.col(i) = H_inf * pkm1.col(i).template cast<Scalar>();
        }
    }
    
    return pk_hat;
}

template <typename Scalar>
Scalar MeasurementOutdoorFlowBundle::logLikelihoodImpl(const Eigen::VectorX<Scalar> & x) const
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
Scalar MeasurementOutdoorFlowBundle::costOdometryImpl(const Eigen::VectorX<Scalar> & etak, const Eigen::VectorXd & etakm1) const
{
    Eigen::VectorX<Scalar> x(18);
    x.template segment<6>(6) = etak;
    x.template segment<6>(12) = etakm1;
    return -logLikelihoodImpl(x);
}

#endif

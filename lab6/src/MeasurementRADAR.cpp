#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include "Gaussian.hpp"
#include "MeasurementGaussianLikelihood.h"
#include "MeasurementRADAR.h"

const double MeasurementRADAR::r1 = 5000;    // Horizontal position of sensor [m]
const double MeasurementRADAR::r2 = 5000;    // Vertical position of sensor [m]

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y)
    : MeasurementGaussianLikelihood(time, y)
{
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRTINV;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::MeasurementRADAR(double time, const Eigen::VectorXd & y, int verbosity)
    : MeasurementGaussianLikelihood(time, y, verbosity)
{
    updateMethod_ = UpdateMethod::BFGSTRUSTSQRTINV;
    // updateMethod_ = UpdateMethod::SR1TRUSTEIG;
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;

    // updateMethod_ = UpdateMethod::AFFINE;
    // updateMethod_ = UpdateMethod::GAUSSNEWTON;
    // updateMethod_ = UpdateMethod::LEVELBERGMARQUARDT;
}

MeasurementRADAR::~MeasurementRADAR() = default;

std::string MeasurementRADAR::getProcessString() const
{
    return "RADAR measurement update:";
}

// Evaluate h(x) from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd h(1);
    // TODO
    h << std::hypot(r1, x(0) - r2);     
    return h;
}

// Evaluate h(x) and its Jacobian J = dh/fx from the measurement model y = h(x) + v
Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx) const
{
    Eigen::VectorXd h = predict(x, system);

    //              dh_i
    // dhdx(i, j) = ----
    //              dx_j
    dhdx.resize(h.size(), x.size());
    dhdx.setZero();
    // TODO: Set non-zero elements of dhdx
    dhdx(0, 0) = (x(0) - r2) / h(0);
    dhdx(0, 1) = 0;
    dhdx(0, 2) = 0;
    return h;
}

Eigen::VectorXd MeasurementRADAR::predict(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::MatrixXd & dhdx, Eigen::Tensor<double, 3> & d2hdx2) const
{
    Eigen::VectorXd h = predict(x, system, dhdx);

    //                    d^2 h_i     d 
    // d2hdx2(i, j, k) = --------- = ---- dhdx(i, j)
    //                   dx_j dx_k   dx_k
    d2hdx2.resize(h.size(), x.size(), x.size());
    d2hdx2.setZero();

    // Only nonzero is d2hdx2(0,0,0)
    // From MATLAB: d2hdx2(1, 1, 1) = obj.r1^2/h(1)^3;
    d2hdx2(0, 0, 0) = std::pow(r1, 2) / std::pow(h(0), 3);

    return h;
}

Gaussian<double> MeasurementRADAR::noiseDensity(const SystemEstimator & system) const
{
    // SR is an upper triangular matrix such that SR.'*SR = R is the measurement noise covariance
    Eigen::MatrixXd SR(1, 1);
    // TODO
    SR << 50;
    return Gaussian<double>::fromSqrtMoment(SR);
}


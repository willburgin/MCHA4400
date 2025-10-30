#include "MeasurementAltimeter.h"
#include <cmath>

MeasurementAltimeter::MeasurementAltimeter(double time, const Eigen::VectorXd& y)
    : MeasurementGaussianLikelihood(time, y) {
    updateMethod_ = UpdateMethod::AFFINE;
}

MeasurementAltimeter::MeasurementAltimeter(double time, const Eigen::VectorXd& y, int verbosity)
    : MeasurementGaussianLikelihood(time, y, verbosity) {
    updateMethod_ = UpdateMethod::AFFINE;
}

MeasurementAltimeter::~MeasurementAltimeter() = default;

std::string MeasurementAltimeter::getProcessString() const {
    return "Altimeter measurement update:";
}

// Evaluate h(x) from the measurement model y = h(x) + v
// For altimeter: h(x) = -e3^T*rCNn (altimeter position from state)
Eigen::VectorXd MeasurementAltimeter::predict(const Eigen::VectorXd& x,
                                                const SystemEstimator& system) const {
    // Extract altimeter position from state
    Eigen::Vector3d e3(0, 0, 1);
    Eigen::Vector3d rBNn = x.segment<3>(6);
    Eigen::Vector3d rCNn = rBNn;
    Eigen::VectorXd h(1);
    h(0) = -e3.dot(rCNn);
    return h;
}

// Evaluate h(x) and its Jacobian J = dh/dx
Eigen::VectorXd MeasurementAltimeter::predict(const Eigen::VectorXd& x,
                                                const SystemEstimator& system,
                                                Eigen::MatrixXd& dhdx) const {
    Eigen::VectorXd h = predict(x, system);

    // Jacobian for altimeter measurement
    dhdx.resize(1, x.size());
    dhdx.setZero();

    dhdx(0, 8) = -1.0;  // down component of state
    return h;
}

// Evaluate h(x), Jacobian, and Hessian
Eigen::VectorXd MeasurementAltimeter::predict(const Eigen::VectorXd& x,
                                                const SystemEstimator& system,
                                                Eigen::MatrixXd& dhdx,
                                                Eigen::Tensor<double, 3>& d2hdx2) const {
    Eigen::VectorXd h = predict(x, system, dhdx);

    // Hessian for altimeter measurement
    // Since h(x) is linear (identity mapping), all second derivatives are zero
    d2hdx2.resize(1, x.size(), x.size());
    d2hdx2.setZero();

    return h;
}

// Define measurement noise covariance
GaussianInfo<double> MeasurementAltimeter::noiseDensity(
    const SystemEstimator& system) const {

    // SR is an upper triangular matrix such that SR^T * SR = R
    // where R is the measurement noise covariance
    Eigen::MatrixXd SR = sigma_altimeter_ * Eigen::MatrixXd::Identity(1, 1);

    return GaussianInfo<double>::fromSqrtMoment(SR);
}

/**
 * @file MeasurementAltimeter.h
 * @brief Defines the MeasurementAltimeter class for altimeter measurements
 */
#ifndef MEASUREMENTALTIMETER_H
#define MEASUREMENTALTIMETER_H

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "MeasurementGaussianLikelihood.h"

/**
 * @class MeasurementAltimeter
 * @brief Class for altimeter measurements (altitude)
 *
 * This class represents altimeter measurements in the SLAM system.
 * It inherits from MeasurementGaussianLikelihood and implements altimeter-specific functionality.
 */
class MeasurementAltimeter : public MeasurementGaussianLikelihood {
public:
    /**
     * @brief Construct a new MeasurementAltimeter object
     * @param time The time at which the measurement occurs
     * @param y The measurement vector (altitude in meters)
     */
    MeasurementAltimeter(double time, const Eigen::VectorXd& y);

    /**
     * @brief Construct a new MeasurementAltimeter object with verbosity
     * @param time The time at which the measurement occurs
     * @param y The measurement vector (altitude in meters)
     * @param verbosity The verbosity level for logging and output
     */
    MeasurementAltimeter(double time, const Eigen::VectorXd& y, int verbosity);

    /**
     * @brief Destroy the MeasurementAltimeter object
     */
    virtual ~MeasurementAltimeter() override;

    // Inherited virtual functions
    virtual Eigen::VectorXd predict(const Eigen::VectorXd& x, const SystemEstimator& system) const override;

    virtual Eigen::VectorXd predict(const Eigen::VectorXd& x,
                                    const SystemEstimator& system,
                                    Eigen::MatrixXd& J) const override;

    virtual Eigen::VectorXd predict(const Eigen::VectorXd& x,
                                    const SystemEstimator& system,
                                    Eigen::MatrixXd& J,
                                    Eigen::Tensor<double, 3>& H) const override;

    virtual GaussianInfo<double> noiseDensity(const SystemEstimator& system) const override;

protected:
    /**
     * @brief Get a string representation of the altimeter measurement process
     * @return std::string A string describing the altimeter measurement process
     */
    virtual std::string getProcessString() const override;

private:
    static constexpr double sigma_altimeter_ = 20.0;  ///< Altimeter noise std dev [meters]
};

#endif  // MEASUREMENTALTIMETER_H

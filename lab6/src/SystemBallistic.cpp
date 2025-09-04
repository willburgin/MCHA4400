#include <cmath>
#include <cassert>
#include <Eigen/Core>
#include "Gaussian.hpp"
#include "SystemEstimator.h"
#include "SystemBallistic.h"

const double SystemBallistic::p0 = 101.325e3;            // Air pressure at sea level [Pa]
const double SystemBallistic::M  = 0.0289644;            // Molar mass of dry air [kg/mol]
const double SystemBallistic::R  = 8.31447;              // Gas constant [J/(mol.K)]
const double SystemBallistic::L  = 0.0065;               // Temperature gradient [K/m]
const double SystemBallistic::T0 = 288.15;               // Temperature at sea level [K]
const double SystemBallistic::g  = 9.81;                 // Acceleration due to gravity [m/s^2]

SystemBallistic::SystemBallistic(const Gaussian<double> & density)
    : SystemEstimator(density)
{}

Gaussian<double> SystemBallistic::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise
    // For process noise only in the 2nd and 3rd state equations, SQ should be 2x2
    Eigen::MatrixXd SQ(2, 2);
    // Set SQ where Q = diag([1e-20, 25e-12])
    SQ << 1e-10, 0,
          0,     5e-6;

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return Gaussian<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemBallistic::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    // TODO: Continuous-time process noise in 2nd and 3rd state equations
    idxQ.push_back(1);
    idxQ.push_back(2);
    return idxQ;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    Eigen::VectorXd f(x.size());
    
    // Calculate air properties at current altitude
    double T = SystemBallistic::T0 - SystemBallistic::L * x(0);           
    double gM_RL = SystemBallistic::g * SystemBallistic::M / SystemBallistic::R / SystemBallistic::L;
    double T_T0 = T / SystemBallistic::T0;
    double p = SystemBallistic::p0 * std::pow(T_T0, gM_RL);              
    double rho = p * SystemBallistic::M / SystemBallistic::R / T;        
    
    double d = 0.5 * rho * x(1) * x(1) * x(2);                         
    
    f << x(1),                    
         d - SystemBallistic::g,   
         0;                       
    
    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemBallistic::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(t, x, u);

    J.resize(f.size(), x.size());
    // TODO: Set J
    double T = SystemBallistic::T0 - SystemBallistic::L * x(0);
    double gM_RL = SystemBallistic::g * SystemBallistic::M / SystemBallistic::R / SystemBallistic::L;
    double T_T0 = T / SystemBallistic::T0;
    
    J.setZero();
    J(0, 1) = 1.0;
    J(1, 0) = (SystemBallistic::M * SystemBallistic::p0 * SystemBallistic::L * SystemBallistic::R * (1 - gM_RL) * std::pow(T_T0, gM_RL) * std::pow(x(1), 2) * x(2)) / (2 * std::pow(SystemBallistic::R, 2) * std::pow(T, 2));
    J(1, 1) = (SystemBallistic::M * SystemBallistic::p0 * std::pow(T_T0, gM_RL) * x(1) * x(2)) / (SystemBallistic::R * T);
    J(1, 2) = (SystemBallistic::M * SystemBallistic::p0 * std::pow(T_T0, gM_RL) * std::pow(x(1), 2)) / (2 * SystemBallistic::R * T);
    return f;
}

Eigen::VectorXd SystemBallistic::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}


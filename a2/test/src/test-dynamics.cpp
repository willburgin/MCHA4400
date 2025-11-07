#include <doctest/doctest.h>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

#include "../../src/SystemVisualNav.h"
#include "../../src/SystemVisualNavPointLandmarks.h"

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) \
    INFO(#x " =\n" << x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

SCENARIO("SystemVisualNav::dynamics analytical Jacobian vs autodiff")
{
    GIVEN("A Visual Nav system with state vector")
    {
        // Create state vector: [vBNb(3), omegaBNb(3), rBNn(3), Thetanb(3), rL1Nn(3)]
        // State size: 12 (body) + 3 (one landmark) = 15
        Eigen::VectorXd x(15);
        x << 0.1, 0.2, 0.3,        // vBNb - body translational velocity
             0.05, 0.1, 0.15,      // omegaBNb - body angular velocity  
             1.0, 2.0, 3.0,        // rBNn - body position
             0.1, 0.2, 0.3,        // Thetanb - body orientation (roll, pitch, yaw)
             5.0, 6.0, 7.0;        // rL1Nn - landmark position

        // Create input vector (not used but required for dynamics signature)
        Eigen::VectorXd u = Eigen::VectorXd::Zero(6);

        // Create system density (mock - just need dimensions)
        auto density = GaussianInfo<double>::fromSqrtMoment(Eigen::MatrixXd::Identity(15, 15));
        SystemVisualNavPointLandmarks system(density);

        // (1) Analytical Jacobian from dynamics implementation
        Eigen::MatrixXd J_an;
        Eigen::VectorXd f_an = system.dynamics(0.0, x, u, J_an);

        // (2) Autodiff oracle using templated version
        using autodiff::dual;
        Eigen::VectorX<dual> x_dual = x.cast<dual>();

        // Create lambda for each component of dynamics output
        auto dynamics_i = [&](const Eigen::VectorX<dual>& x_var, int i) -> dual {
            return SystemVisualNav::dynamics(x_var)(i);
        };

        // Compute gradient for each output component
        Eigen::MatrixXd J_exp(x.size(), x.size());
        for(int i = 0; i < x.size(); ++i) {
            auto fi = [&](const Eigen::VectorX<dual>& x_var) -> dual {
                return dynamics_i(x_var, i);
            };
            autodiff::dual f0;
            Eigen::VectorXd gi = gradient(fi, autodiff::wrt(x_dual), autodiff::at(x_dual), f0);
            J_exp.row(i) = gi.transpose();
        }

        THEN("Analytical matches autodiff within tolerance")
        {
            CAPTURE_EIGEN(J_an);
            CAPTURE_EIGEN(J_exp);
            CHECK(J_an.isApprox(J_exp, 1e-6));
        }

        THEN("Dynamics function output matches templated version")
        {
            Eigen::VectorX<dual> f_dual = SystemVisualNav::dynamics(x_dual);
            Eigen::VectorXd f_exp = f_dual.cast<double>();
            
            CAPTURE_EIGEN(f_an);
            CAPTURE_EIGEN(f_exp);
            CHECK(f_an.isApprox(f_exp, 1e-10));
        }
    }
}


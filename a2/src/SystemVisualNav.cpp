#include <cstddef>
#include <cmath>
#include <print>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/mat.hpp>
#include "GaussianInfo.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"

SystemVisualNav::SystemVisualNav(const GaussianInfo<double> & density)
    : SystemEstimator(density)
{

}

SystemVisualNav * SystemVisualNav::clone() const
{
    return new SystemVisualNav(*this);
}

void SystemVisualNav::setFlowEvent(bool isEvent)
{
    isFlowEvent_ = isEvent;
}

void SystemVisualNav::predict(double time)
{
    double dt = time - time_;
    assert(dt >= 0);
    if (dt == 0.0) return;
    
    auto pdw = processNoiseDensity(dt);
    
    auto pxdw = density * pdw;

    bool isFlowEvent = isFlowEvent_;

    // Phi maps [ x[k]; dw(idxQ)[k] ] to x[k+1]
    auto Phi = [&, isFlowEvent](const Eigen::VectorXd & xdw, Eigen::MatrixXd & J)
    {
        int nx = xdw.size() - 6; // size 18
        
        // Extract z[k] = [nu[k]; eta[k]] (first 12 elements of state)
        Eigen::VectorXd zdw(18);
        zdw.head(12) = xdw.head(12); // state
        zdw.tail(6) = xdw.tail(6); // noise
        
        // Integrate z[k] -> z[k+1] using RK4
        Eigen::MatrixXd J_zdw;
        Eigen::VectorXd z_kp1 = RK4SDEHelper(zdw, dt, J_zdw);
        
        // Extract poses for cloning
        Eigen::Vector6d eta_k = xdw.segment<6>(6);
        Eigen::Vector6d zeta_k = xdw.segment<6>(12);
        Eigen::Vector6d eta_kp1 = z_kp1.segment<6>(6);
        
        // Build full state x[k+1]
        Eigen::VectorXd x_kp1(nx);
        x_kp1.head(12) = z_kp1;
        
        if (isFlowEvent) {
            x_kp1.segment<6>(12) = eta_k;
        } else {
            x_kp1.segment<6>(12) = zeta_k;
        }
        
        int nm = nx - 18;
        if (nm > 0) {
            x_kp1.tail(nm) = xdw.segment(18, nm);
        }
        
        // BUILD JACOBIAN 
        J.resize(nx, xdw.size());  // 18 × 24
        J.setZero();
        
        // Jacobian for [nu[k+1]; eta[k+1]] (rows 0-11)
        // These only depend on [nu[k]; eta[k]] and dw, NOT on zeta[k]
        J.block(0, 0, 12, 12) = J_zdw.leftCols(12);     // d[nu,eta]_{k+1}/d[nu,eta]_k
        J.block(0, 12, 12, 6).setZero();                // d[nu,eta]_{k+1}/dzeta_k = 0
        // Columns 18-23 are for landmarks (if any), set to zero
        if (nm > 0) {
            J.block(0, 18, 12, nm).setZero();           // d[nu,eta]_{k+1}/dm_k = 0
        }
        J.block(0, nx, 12, 6) = J_zdw.rightCols(6);     // d[nu,eta]_{k+1}/dw
        
        // Jacobian for zeta[k+1] (rows 12-17)
        if (isFlowEvent) {
            // zeta[k+1] = eta[k]
            J.block(12, 0, 6, 6).setZero();              // dzeta_{k+1}/dnu_k = 0
            J.block<6, 6>(12, 6).setIdentity();          // dzeta_{k+1}/deta_k = I
            J.block(12, 12, 6, 6).setZero();             // dzeta_{k+1}/dzeta_k = 0
            if (nm > 0) {
                J.block(12, 18, 6, nm).setZero();        // dzeta_{k+1}/dm_k = 0
            }
            J.block(12, nx, 6, 6).setZero();             // dzeta_{k+1}/dw = 0
        } else {
            // zeta[k+1] = zeta[k]
            J.block(12, 0, 6, 12).setZero();             // dzeta_{k+1}/d[nu,eta]_k = 0
            J.block<6, 6>(12, 12).setIdentity();         // dzeta_{k+1}/dzeta_k = I
            if (nm > 0) {
                J.block(12, 18, 6, nm).setZero();        // dzeta_{k+1}/dm_k = 0
            }
            J.block(12, nx, 6, 6).setZero();             // dzeta_{k+1}/dw = 0
        }
        
        // Jacobian for map states (rows 18+)
        if (nm > 0) {
            J.block(18, 0, nm, 18).setZero();            // dm_{k+1}/d[nu,eta,zeta]_k = 0
            J.block(18, 18, nm, nm).setIdentity();       // dm_{k+1}/dm_k = I
            J.block(18, nx, nm, 6).setZero();            // dm_{k+1}/dw = 0
        }
        
        return x_kp1;
    };

    density = pxdw.affineTransform(Phi);

    time_ = time;
    isFlowEvent_ = false;
}

// Evaluate f(x) from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u) const
{
    assert(density.dim() == x.size());
    //
    //  dnu/dt =          0 + dwnu/dt
    // deta/dt = JK(eta)*nu +       0
    //   dm/dt =          0 +       0
    // \_____/   \________/   \_____/
    //  dx/dt  =    f(x)    +  dw/dt
    //
    //        [          0 ]
    // f(x) = [ JK(eta)*nu ]
    //        [          0 ] for all map states
    //
    //        [                    0 ]
    //        [                    0 ]
    // f(x) = [    Rnb(thetanb)*vBNb ]
    //        [ TK(thetanb)*omegaBNb ]
    //        [                    0 ] for all map states
    //
    Eigen::VectorXd f(x.size());
    f.setZero();
    // TODO: Implement in Assignment(s)
    // set nu and eta
    Eigen::Vector6d nu = x.segment<6>(0);
    Eigen::Vector6d eta = x.segment<6>(6);
    Eigen::Matrix<double, 6, 6> J_eta = eulerKinematicTransformation(eta);
    f.segment<6>(6) = J_eta * nu;
    return f;
}

// Evaluate f(x) and its Jacobian J = df/fx from the SDE dx = f(x)*dt + dw
Eigen::VectorXd SystemVisualNav::dynamics(double t, const Eigen::VectorXd & x, const Eigen::VectorXd & u, Eigen::MatrixXd & J) const
{
    Eigen::VectorXd f = dynamics(t, x, u);

    // Jacobian J = df/dx
    //    
    //     [  0                  0 0 ]
    // J = [ JK d(JK(eta)*nu)/deta 0 ]
    //     [  0                  0 0 ]
    //
    J.resize(f.size(), x.size());
    J.setZero();
    // build dRnb(thetanb)vBNb/dvBNb
    // extract states
    double ustate = x(0);
    double v = x(1);
    double w = x(2);
    double q = x(4);
    double r = x(5);
    double phi = x(9);
    double theta = x(10);
    double psi = x(11);
    Eigen::Vector3d Thetanb = Eigen::Vector3d(phi, theta, psi);
    double cphi = std::cos(phi);
    double sphi = std::sin(phi);
    double ctheta = std::cos(theta);
    double stheta = std::sin(theta);
    double cpsi = std::cos(psi);
    double spsi = std::sin(psi);
    double ttheta = std::tan(theta);
    double sec_theta = 1 / std::cos(theta);
    // build matrix
    Eigen::Matrix3d dRnb_dvBNb = rpy2rot(Thetanb);
    Eigen::Matrix3d dRnb_dThetanb = Eigen::Matrix3d::Zero();
    dRnb_dThetanb(0,0) = (spsi * sphi + cpsi * stheta * cphi) * v + (spsi * cphi - cpsi * sphi * stheta) * w; // df1/dphi   
    dRnb_dThetanb(0,1) = -cpsi * stheta * ustate + cpsi * ctheta * sphi * v + cpsi * cphi * ctheta * w; // df1/dtheta
    dRnb_dThetanb(0,2) = -spsi * ctheta * ustate - (cpsi * cphi + spsi * stheta * sphi) * v + (cpsi * sphi - spsi * cphi * stheta) * w; // df1/dpsi
    dRnb_dThetanb(1,0) = (-cpsi * sphi + cphi * stheta * spsi) * v + (-cpsi * cphi - spsi * sphi * stheta) * w; // df2/dphi
    dRnb_dThetanb(1,1) = -spsi * stheta * ustate + sphi * ctheta * spsi * v + spsi * cphi * ctheta * w; // df2/dtheta
    dRnb_dThetanb(1,2) = cpsi * ctheta * ustate -(spsi * cphi - sphi * stheta * cpsi) * v + (spsi * sphi + cpsi * cphi * stheta) * w; // df2/dpsi
    dRnb_dThetanb(2,0) =  ctheta * cphi * v - ctheta * sphi * w; // df3/dphi
    dRnb_dThetanb(2,1) = -ctheta * ustate - stheta * sphi * v - stheta * cphi * w; // df3/dtheta
    dRnb_dThetanb(2,2) = 0; // df3/dpsi
    // build matrix dTK(thetanb)/dThetanb
    Eigen::Matrix3d dTk_dwBNb = TK(Thetanb);
    Eigen::Matrix3d dTK_dThetanb = Eigen::Matrix3d::Zero();
    dTK_dThetanb(0,0) = cphi * ttheta * q - sphi * ttheta * r; // df1/dphi
    dTK_dThetanb(0,1) = (sphi / (ctheta * ctheta)) * q + (cphi / (ctheta * ctheta)) * r; // df1/dtheta
    dTK_dThetanb(1,0) = -sphi * q - cphi * r; // df2/dphi
    dTK_dThetanb(2,0) = (cphi / ctheta) * q - (sphi / ctheta) * r; // df3/dphi
    dTK_dThetanb(2,1) = sphi * ttheta * (1/ctheta) * q + cphi * ttheta * (1/ctheta) * r; // df3/dtheta
    // TODO: Implement in Assignment(s) - implement this analytically.
    // build jacobian from above
    J.block<3,3>(6, 0) = dRnb_dvBNb;
    J.block<3,3>(6, 9) = dRnb_dThetanb;
    J.block<3,3>(9, 3) = dTk_dwBNb;
    J.block<3,3>(9, 9) = dTK_dThetanb;
    return f;
}

Eigen::VectorXd SystemVisualNav::input(double t, const Eigen::VectorXd & x) const
{
    return Eigen::VectorXd(0);
}

GaussianInfo<double> SystemVisualNav::processNoiseDensity(double dt) const
{
    // SQ is an upper triangular matrix such that SQ.'*SQ = Q is the power spectral density of the continuous time process noise

    // // TODO: Assignment(s) Tuning parameters
    const double sigma_vx = 0.2;  // m/s / sqrt(s)
    const double sigma_vy = 0.2;  // m/s / sqrt(s)
    const double sigma_vz = 0.04;  // m/s / sqrt(s)
    const double sigma_p  = 0.3;  // rad/s / sqrt(s)
    const double sigma_q  = 0.3;  // rad/s / sqrt(s)
    const double sigma_r  = 0.09;  // rad/s / sqrt(s)

    Eigen::Matrix<double, 6, 6> SQ = Eigen::Matrix<double, 6, 6>::Zero();
    SQ(0,0) = sigma_vx;
    SQ(1,1) = sigma_vy;
    SQ(2,2) = sigma_vz;
    SQ(3,3) = sigma_p;
    SQ(4,4) = sigma_q;
    SQ(5,5) = sigma_r;
    

    // Distribution of noise increment dw ~ N(0, Q*dt) for time increment dt
    return GaussianInfo<double>::fromSqrtMoment(SQ*std::sqrt(dt));
}

std::vector<Eigen::Index> SystemVisualNav::processNoiseIndex() const
{
    // Indices of process model equations where process noise is injected
    std::vector<Eigen::Index> idxQ;
    // TODO: Assignment(s)
    idxQ = {0, 1, 2, 3, 4, 5};
    return idxQ;
}

cv::Mat & SystemVisualNav::view()
{
    return view_;
};

const cv::Mat & SystemVisualNav::view() const
{
    return view_;
};

std::size_t SystemVisualNav::numberLandmarks() const
{
    return (density.dim() - 18)/3;
}

std::size_t SystemVisualNav::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 18 + 3*idxLandmark;    
}

GaussianInfo<double> SystemVisualNav::bodyPositionDensity() const
{
    return density.marginal(Eigen::seqN(6, 3));
}

GaussianInfo<double> SystemVisualNav::bodyOrientationDensity() const
{
    return density.marginal(Eigen::seqN(9, 3));
}

GaussianInfo<double> SystemVisualNav::bodyTranslationalVelocityDensity() const
{
    return density.marginal(Eigen::seqN(0, 3));
}

GaussianInfo<double> SystemVisualNav::bodyAngularVelocityDensity() const
{
    return density.marginal(Eigen::seqN(3, 3));
}

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

Eigen::Vector3d SystemVisualNav::cameraPosition(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> rCNn_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraPosition<autodiff::dual>, wrt(x_dual), at(camera, x_dual), rCNn_dual);
    return rCNn_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraPositionDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraPosition(camera, x, J); };
    return density.affineTransform(f);
}

Eigen::Vector3d SystemVisualNav::cameraOrientationEuler(const Camera & camera, const Eigen::VectorXd & x, Eigen::MatrixXd & J)
{
    Eigen::Vector3<autodiff::dual> Thetanc_dual;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    J = jacobian(cameraOrientationEuler<autodiff::dual>, wrt(x_dual), at(camera, x_dual), Thetanc_dual);
    return Thetanc_dual.cast<double>();
};

GaussianInfo<double> SystemVisualNav::cameraOrientationEulerDensity(const Camera & camera) const
{
    auto f = [&](const Eigen::VectorXd & x, Eigen::MatrixXd & J) { return cameraOrientationEuler(camera, x, J); };
    return density.affineTransform(f);    
}

GaussianInfo<double> SystemVisualNav::landmarkPositionDensity(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    std::size_t idx = landmarkPositionIndex(idxLandmark);
    return density.marginal(Eigen::seqN(idx, 3));
}

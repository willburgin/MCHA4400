#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    
    // Rotation around X-axis:
    // R = [1   0      0    ]
    //     [0  cos(x) -sin(x)]
    //     [0  sin(x)  cos(x)]
    R(1,1) =  cos(x);
    R(1,2) = -sin(x);
    R(2,1) =  sin(x);
    R(2,2) =  cos(x);
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx            =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(1,1)       = -sin(x);
    dRdx(2,1)       =  cos(x);

    dRdx(1,2)       = -cos(x);
    dRdx(2,2)       = -sin(x);
    return rotx(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    // TODO: Lab 8
    // Rotation around Y-axis:
    // R = [ cos(y)  0  sin(y)]
    //     [ 0      1  0    ]
    //     [-sin(y) 0  cos(y)]
    R(0,0) =  cos(x);
    R(0,2) =  sin(x);
    R(2,0) = -sin(x);
    R(2,2) =  cos(x);
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(2,0)    = -cos(x);

    dRdx(0,2)    =  cos(x);
    dRdx(2,2)    = -sin(x);
    return roty(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    // TODO: Lab 8
    // Rotation around Z-axis:
    // R = [ cos(z) -sin(z) 0]
    //     [ sin(z)  cos(z) 0]
    //     [ 0       0      1]
    R(0,0) =  cos(x);
    R(0,1) = -sin(x);
    R(1,0) =  sin(x);
    R(1,1) =  cos(x);
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(1,0)    =  cos(x);

    dRdx(0,1)    = -cos(x);
    dRdx(1,1)    = -sin(x);
    return rotz(x);
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> rpy2rot(const Eigen::MatrixBase<Derived> & Theta)
{
    using Scalar = typename Derived::Scalar;
    // R = Rz*Ry*Rx
    Eigen::Matrix3<Scalar> R;
    // TODO: Lab 8
    // R = Rz*Ry*Rx
    R = rotz(Theta(2)) * roty(Theta(1)) * rotx(Theta(0));
    return R;
}

template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> rot2rpy(const Eigen::MatrixBase<Derived> & R)
{
    using Scalar = typename Derived::Scalar;
    using std::atan2, std::hypot;
    Eigen::Vector3<Scalar> Theta;
    // TODO: Lab 8
    // Theta = [atan2(R(2,1), R(2,2)), atan2(-R(2,0), sqrt(R(2,1)^2 + R(2,2)^2)), atan2(R(1,0), R(0,0))]
    Theta(0) = atan2(R(2,1), R(2,2));
    Theta(1) = atan2(-R(2,0), hypot(R(2,1), R(2,2)));
    Theta(2) = atan2(R(1,0), R(0,0));
    return Theta;
}

// implement the kinematic transformation matrix given by:
// J(eta) = [Rnb(thetanb)    0]
//          [0               T(thetanb)]
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> TK(const Eigen::Matrix<Scalar, 3, 1> & Thetanb)
{
    Scalar phi = Thetanb(0);
    Scalar theta = Thetanb(1);
    Eigen::Matrix<Scalar, 3, 3> TK = Eigen::Matrix<Scalar, 3, 3>::Zero();
    using std::cos, std::sin, std::tan;
    Scalar cphi = cos(phi);
    Scalar sphi = sin(phi);
    Scalar ctheta = cos(theta);
    Scalar ttheta = tan(theta);
    TK(0, 0) = 1;
    TK(0, 1) = sphi * ttheta;
    TK(0, 2) = cphi * ttheta;
    TK(1, 1) = cphi;
    TK(1, 2) = -sphi;
    TK(2, 1) = sphi / ctheta;
    TK(2, 2) = cphi / ctheta;
    return TK;
}
// build the kinematic transformation matrix
template <typename Scalar>
Eigen::Matrix<Scalar, 6, 6> eulerKinematicTransformation(const Eigen::Matrix<Scalar, 6, 1> & eta)
{
    Eigen::Matrix<Scalar, 3, 1> thetanb = eta.template segment<3>(3);
    Eigen::Matrix<Scalar, 3, 3> Rnb = rpy2rot(thetanb);
    Eigen::Matrix<Scalar, 3, 3> T = TK(thetanb);

    Eigen::Matrix<Scalar, 6, 6> J = Eigen::Matrix<Scalar, 6, 6>::Zero();
    J.template block<3,3>(0,0) = Rnb;
    J.template block<3,3>(3,3) = T;
    return J;
}

#endif

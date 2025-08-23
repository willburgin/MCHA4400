// Tip: Only include headers needed to parse this implementation only
#include <cassert>
#include <Eigen/Core>
#include <Eigen/QR>

#include "Gaussian.h"

Gaussian::Gaussian()
{

}

Gaussian::Gaussian(const Eigen::VectorXd & mu, const Eigen::MatrixXd & S)
    : mu_(mu)
    , S_(S)
{
    assert(mu_.size() == S_.cols());
    assert(S_.isUpperTriangular());
}

Eigen::VectorXd Gaussian::mean() const
{
    return mu_;
}

Eigen::MatrixXd Gaussian::sqrtCov() const
{
    return S_;
}

Eigen::MatrixXd Gaussian::cov() const
{
    return S_.transpose()*S_;
}
// TODO: Add member function implementations
Gaussian Gaussian::add(const Gaussian & other) const
{
    // compute mu
    Eigen::VectorXd mu_out = mu_ + other.mu_;

    // compute matrix A (the tall matrix)
    Eigen::MatrixXd A(S_.rows() + other.S_.rows(), S_.cols());
    A.topRows(S_.rows()) = S_;
    A.bottomRows(other.S_.rows()) = other.S_;

    // construct R from S using the householder QR decomposition
    const int n = A.cols(); // state size
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>(); // upper triangular matrix
    // compute S_out
    Eigen::MatrixXd S_out = R.topLeftCorner(n, n); // we want an nxn matrix 

    // return new Gaussian
    Gaussian out(mu_out, S_out);
    return out;
}

Gaussian Gaussian::marginalHead(int na) const
{
    // add the state size n and precondition for na relative to n
    const int n = S_.rows();
    // compute mu_out
    Eigen::VectorXd mu_out = mu_.head(na);
    // compute tall matrix A again (keep all rows and take the first na columns)
    Eigen::MatrixXd A = S_.topLeftCorner(n, na);
    // call householder QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>(); // upper triangular matrix
    Eigen::MatrixXd S_out = R.topLeftCorner(na, na); // we want an nxn matrix 

    Gaussian out(mu_out, S_out);
    return out;
}

Gaussian Gaussian::conditionalTailGivenHead(const Eigen::VectorXd & xa) const
{   
    // partition the mean mu into mu_a and mu_b
    const int n = S_.rows();
    const int na = xa.size();
    const int nb = n - na;
    Eigen::VectorXd mu_a = mu_.head(na);
    Eigen::VectorXd mu_b = mu_.tail(nb);

    // partition the covariance matrix S into S1 S2 and S3
    Eigen::MatrixXd S1 = S_.topLeftCorner(na, na);
    Eigen::MatrixXd S2 = S_.topRightCorner(na, nb);
    Eigen::MatrixXd S3 = S_.bottomRightCorner(nb, nb);

    // compute the conditional mean mu_b_a
    Eigen::VectorXd diff = xa - mu_a;
    Eigen::VectorXd mub_a = mu_b + S2.transpose() * S1.transpose().triangularView<Eigen::Lower>().solve(diff);  

    // compute the conditional covariance Sb_a 
    Eigen::MatrixXd Sb_a = S3;

    Gaussian out(mub_a, Sb_a);
    return out;
}

Gaussian Gaussian::permute(const Eigen::ArrayXi & idx) const
{
    // permute the mean mu
    Eigen::VectorXd mu_out = mu_(idx);    
    // construct tall matrix A  
    Eigen::MatrixXd A = S_(Eigen::all, idx);
    // qr time
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

    // extract S_out
    Eigen::MatrixXd S_out = R.topLeftCorner(idx.size(), idx.size());

    Gaussian out(mu_out, S_out);
    return out;
}

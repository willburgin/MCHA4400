#include <cstddef>
#include <print>
#include <numeric>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "GaussianInfo.hpp"
#include "rotation.hpp"
#include "SystemEstimator.h"
#include "SystemVisualNav.h"
#include "MeasurementOutdoorFlowBundle.h"

MeasurementOutdoorFlowBundle::MeasurementOutdoorFlowBundle(double time, const Camera & camera, const cv::Mat & imgk_raw, const cv::Mat & imgkm1_raw, const Eigen::Matrix<double, 2, Eigen::Dynamic> & rQOikm1)
    : Measurement(time)
    , camera_(camera)
    , rQOikm1_(rQOikm1)
    , rQOik_()
    , rQbarOikm1_()
    , rQbarOik_()
    , mask_()
    , pkm1_()
    , pk_()
    , sigma_(2.0) // TODO: Assignment(s)
{
    // TODO: Assignment(s)
    // updateMethod_ = UpdateMethod::NEWTONTRUSTEIG;
    // updateMethod_ = UpdateMethod::BFGSTRUSTSQRT;

    // TODO: Lab 11
    const int divisor               = 2;                // Image scaling factor
    const int maxNumFeatures        = 900;                // Maximum number of features per frame
    const int minNumFeatures        = 850;                // Minimum number of feature per frame

    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,30,0.01);
    cv::Size subPixWinSize(11, 11);     // Window size for subpixel refinement
    cv::Size winSize(21, 21);           // Window size for optical flow

    // Convert images to grayscale
    cv::Mat imgk_gray;
    cv::Mat imgkm1_gray;
    cv::cvtColor(imgk_raw, imgk_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgkm1_raw, imgkm1_gray, cv::COLOR_BGR2GRAY);

    // Scale images
    cv::Mat imgk_scaled;
    cv::Mat imgkm1_scaled;
    cv::resize(imgk_gray, imgk_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);
    cv::resize(imgkm1_gray, imgkm1_scaled, cv::Size(), 1.0/divisor, 1.0/divisor);

    std::vector<cv::Point2f> rQOikm1_scaled;
    std::vector<cv::Point2f> rQOik_scaled;
    std::vector<uchar> status;
    std::vector<float> err;
    
    if (rQOikm1_.cols() < minNumFeatures)
    {
        std::println("Adding new features to reach minimum count.");
        
        // Convert existing features to scaled coordinates
        rQOikm1_scaled.resize(rQOikm1_.cols());
        for (int j = 0; j < rQOikm1_.cols(); ++j)
        {
            rQOikm1_scaled[j].x = rQOikm1_(0, j)/divisor;
            rQOikm1_scaled[j].y = rQOikm1_(1, j)/divisor;
        }
        
        // Create mask to avoid detecting features near existing ones
        cv::Mat mask = cv::Mat::ones(imgk_scaled.size(), CV_8U) * 255;
        for (const auto& pt : rQOikm1_scaled) {
            cv::circle(mask, pt, 20, cv::Scalar(0), -1);  // Exclude 20px radius
        }
        
        // Detect new features
        int numNewFeatures = maxNumFeatures - rQOikm1_.cols();
        std::vector<cv::Point2f> rQOik_new;
        cv::goodFeaturesToTrack(imgk_scaled, rQOik_new, numNewFeatures, 0.01, 40, mask);
        cv::cornerSubPix(imgk_scaled, rQOik_new, cv::Size(5, 5), cv::Size(-1, -1), termcrit);
        std::println("Found {} new features.", rQOik_new.size());
        
        // Append new features to existing ones (as if they were in previous frame)
        rQOikm1_scaled.insert(rQOikm1_scaled.end(), rQOik_new.begin(), rQOik_new.end());
    }
    else
    {
        // Just use existing features
        rQOikm1_scaled.resize(rQOikm1_.cols());
        for (int j = 0; j < rQOikm1_.cols(); ++j)
        {
            rQOikm1_scaled[j].x = rQOikm1_(0, j)/divisor;
            rQOikm1_scaled[j].y = rQOikm1_(1, j)/divisor;
        }
    }

    // Now track ALL features (existing + new) from frame k-1 to frame k
    cv::calcOpticalFlowPyrLK(imgkm1_scaled, imgk_scaled, rQOikm1_scaled, rQOik_scaled, status, err, winSize);

    // Keep points that have been matched between both frames
    std::vector<cv::Point2f> rQOik_filtered, rQOikm1_filtered;
    for (size_t i = 0; i < status.size(); i++) {
        if (status[i]) {
            rQOik_filtered.push_back(rQOik_scaled[i]);
            rQOikm1_filtered.push_back(rQOikm1_scaled[i]);
        }
    }
    
    int np = rQOik_filtered.size();
    // TODO: Lab 11 
    std::println("After filtering by status, there are {} associations.", np);

    // Calculate where flow points would be in the unscaled image
    rQOik_.resize(2, np);
    rQOikm1_.resize(2, np);
    for (int j = 0; j < np; ++j)
    {
        rQOik_(0, j) = rQOik_filtered[j].x*divisor;
        rQOik_(1, j) = rQOik_filtered[j].y*divisor;

        rQOikm1_(0, j) = rQOikm1_filtered[j].x*divisor;
        rQOikm1_(1, j) = rQOikm1_filtered[j].y*divisor;
    }

    // Calculate the undistorted location of features
    rQbarOik_ = camera_.undistort(rQOik_);
    rQbarOikm1_ = camera_.undistort(rQOikm1_);
    // TODO: Lab 11

    // Use RANSAC to find fundamental matrix and determine inliers (mask_)
    //
    // Note: We don't actually use the fundamental matrix computed here, it is just used to
    //       determine which undistorted flow vectors are consistent with the epipolar constraint.

    // TODO: Lab 11
    std::vector<cv::Point2f> rQbarOik_cv, rQbarOikm1_cv;
    for (int i = 0; i < rQbarOik_.cols(); i++) {
        rQbarOik_cv.push_back(cv::Point2f(rQbarOik_(0, i), rQbarOik_(1, i)));
        rQbarOikm1_cv.push_back(cv::Point2f(rQbarOikm1_(0, i), rQbarOikm1_(1, i)));
    }
    
    // Find fundamental matrix on undistorted points
    cv::Mat mask_cv;
    double threshold = 3.0;  // Epipolar error threshold in pixels
    if (rQbarOik_cv.size() >= 8) {
        cv::Mat F = cv::findFundamentalMat(rQbarOikm1_cv, rQbarOik_cv, 
                                           cv::FM_RANSAC, threshold, 0.99, mask_cv);
        
        // Convert mask to std::vector<bool>
        mask_.resize(mask_cv.rows);
        for (int i = 0; i < mask_cv.rows; i++) {
            mask_[i] = (mask_cv.at<uchar>(i) != 0);
        }
    } else {
        // Not enough points for RANSAC, accept all
        mask_.resize(np, true);
    }
    
    int nInliers = std::count(mask_.begin(), mask_.end(), true);
    std::println("No. inliers = {}, No. outliers  = {}", nInliers, mask_.size() - nInliers);

    // Inlier undistorted homogeneous points (pkm1_ and pk_)
    pk_     = Eigen::MatrixXd::Ones(3, nInliers);
    pkm1_   = Eigen::MatrixXd::Ones(3, nInliers);
    // TODO: Lab 11
    int inlierIdx = 0;
    for (int j = 0; j < np; ++j) {
        if (mask_[j]) {
            pkm1_(0, inlierIdx) = rQbarOikm1_(0, j);
            pkm1_(1, inlierIdx) = rQbarOikm1_(1, j);
            pkm1_(2, inlierIdx) = 1.0;
            
            pk_(0, inlierIdx) = rQbarOik_(0, j);
            pk_(1, inlierIdx) = rQbarOik_(1, j);
            pk_(2, inlierIdx) = 1.0;
            
            inlierIdx++;
        }
    }
}

Eigen::VectorXd MeasurementOutdoorFlowBundle::simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    Eigen::VectorXd y;
    throw std::runtime_error("Not implemented");
    return y;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedPreviousFeatures() const
{
    return rQOikm1_;
}

const Eigen::Matrix<double, 2, Eigen::Dynamic> & MeasurementOutdoorFlowBundle::trackedCurrentFeatures() const
{
    return rQOik_;
}

const std::vector<unsigned char> & MeasurementOutdoorFlowBundle::inlierMask() const
{
    return mask_;
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    return logLikelihoodImpl(x);
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const
{
    // Evaluate gradient for Newton and quasi-Newton methods
    double logLik;
    Eigen::VectorX<autodiff::dual> x_dual = x.cast<autodiff::dual>();
    
    // Create a lambda that computes the log likelihood
    auto f = [&](const Eigen::VectorX<autodiff::dual>& x_) -> autodiff::dual {
        return logLikelihoodImpl<autodiff::dual>(x_);
    };
    
    // Compute gradient
    autodiff::dual logLik_dual;
    g = gradient(f, wrt(x_dual), at(x_dual), logLik_dual);
    logLik = val(logLik_dual);
    
    return logLik;
}

double MeasurementOutdoorFlowBundle::logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Evaluate Hessian for Newton method
    H.resize(x.size(), x.size());
    
    double logLik;
    Eigen::VectorX<autodiff::dual2nd> x_dual2nd = x.cast<autodiff::dual2nd>();
    
    // Create a lambda that computes the log likelihood
    auto f = [&](const Eigen::VectorX<autodiff::dual2nd>& x_) -> autodiff::dual2nd {
        return logLikelihoodImpl<autodiff::dual2nd>(x_);
    };
    
    // Compute gradient and Hessian
    autodiff::dual2nd logLik_dual2nd;
    H = hessian(f, wrt(x_dual2nd), at(x_dual2nd), logLik_dual2nd, g);
    logLik = val(logLik_dual2nd);
    
    return logLik;
}
Eigen::Matrix<double, 2, Eigen::Dynamic> MeasurementOutdoorFlowBundle::predictedFeatures(const Eigen::VectorXd & x, const SystemEstimator & system) const
{
    std::size_t np = rQOik_.cols();

    // std::println("Debug predictedFeatures:");
    // std::println("  rBNn_k: {} {} {}", x(6), x(7), x(8));
    // std::println("  thetaNB_k: {} {} {}", x(9), x(10), x(11));
    // std::println("  rBNn_km1: {} {} {}", x(12), x(13), x(14));
    // std::println("  thetaNB_km1: {} {} {}", x(15), x(16), x(17));

    // Predict undistorted homogeneous image points in current frame
    Eigen::Matrix<double, 3, Eigen::Dynamic> pk(3, np);
    Eigen::Matrix<double, 3, Eigen::Dynamic> pkm1(3, np);
    pkm1.topRows<2>() = rQbarOikm1_;
    pkm1.row(2).setOnes();
    pk.topRows<2>() = rQbarOik_;
    pk.row(2).setOnes();

    Eigen::Matrix<double, 3, Eigen::Dynamic> pk_hat = predictFlowImpl(x, pkm1, pk);
    for (int i = 0; i < std::min(5, (int)pk_hat.cols()); ++i) {
        std::println("  pk_hat col {}: {} {} {}", i, pk_hat(0,i), pk_hat(1,i), pk_hat(2,i));
    }
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQbarOik_hat = pk_hat.topRows<2>().array().rowwise()/pk_hat.row(2).array();
    assert(rQbarOik_hat.cols() == np);
    
    // Compute image coordinates (with lens distortion)
    // TODO: Lab 11 - Apply lens distortion to undistorted features
    Eigen::Matrix<double, 2, Eigen::Dynamic> rQOik_hat = camera_.distort(rQbarOik_hat);
    return rQOik_hat;
}

void MeasurementOutdoorFlowBundle::update(SystemBase & system_)
{
    // Set the flow event flag before the base class calls predict()
    SystemVisualNav & system = dynamic_cast<SystemVisualNav &>(system_);
    system.setFlowEvent(true);
    std::println("Debug update: setFlowEvent(true)");
    
    // Call base class implementation (which calls predict() and does optimization)
    Measurement::update(system_);
}

// Note: costOdometry is used only in Lab 11, not Assignment 2.
double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1) const
{
    return costOdometryImpl(etak, etakm1);
}

double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g) const
{
    // Forward-mode autodifferentiation
    Eigen::Matrix<autodiff::dual, Eigen::Dynamic, 1> etakdual = etak.cast<autodiff::dual>();
    autodiff::dual fdual;
    g = gradient(&MeasurementOutdoorFlowBundle::costOdometryImpl<autodiff::dual>, wrt(etakdual), at(this, etakdual, etakm1), fdual);
    return val(fdual);
}

double MeasurementOutdoorFlowBundle::costOdometry(const Eigen::VectorXd & etak, const Eigen::VectorXd & etakm1, Eigen::VectorXd & g, Eigen::MatrixXd & H) const
{
    // Forward-mode autodifferentiation
    Eigen::Matrix<autodiff::dual2nd, Eigen::Dynamic, 1> etakdual = etak.cast<autodiff::dual2nd>();
    autodiff::dual2nd fdual;
    H = hessian(&MeasurementOutdoorFlowBundle::costOdometryImpl<autodiff::dual2nd>, wrt(etakdual), at(this, etakdual, etakm1), fdual, g);
    return val(fdual);
}


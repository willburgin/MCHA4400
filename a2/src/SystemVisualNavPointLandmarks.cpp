#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"
#include "SystemVisualNavPointLandmarks.h"

SystemVisualNavPointLandmarks::SystemVisualNavPointLandmarks(const GaussianInfo<double> & density)
    : SystemVisualNav(density)
{

}

SystemVisualNav * SystemVisualNavPointLandmarks::clone() const
{
    return new SystemVisualNavPointLandmarks(*this);
}


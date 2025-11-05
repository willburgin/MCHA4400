#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"
#include "SystemVisualNavPoseLandmarks.h"

SystemVisualNavPoseLandmarks::SystemVisualNavPoseLandmarks(const GaussianInfo<double> & density)
    : SystemVisualNav(density)
{

}

SystemVisualNav * SystemVisualNavPoseLandmarks::clone() const
{
    return new SystemVisualNavPoseLandmarks(*this);
}

std::size_t SystemVisualNavPoseLandmarks::numberLandmarks() const
{
    return (density.dim() - 18)/6;
}

std::size_t SystemVisualNavPoseLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 18 + 6*idxLandmark;    
}

void SystemVisualNavPoseLandmarks::addKnownMarkerID(int markerID)
{
    if (std::find(knownMarkerIDs_.begin(), knownMarkerIDs_.end(), markerID) == knownMarkerIDs_.end())
    {
        knownMarkerIDs_.push_back(markerID);
    }
}

bool SystemVisualNavPoseLandmarks::isMarkerKnown(int markerID) const
{
    return std::find(knownMarkerIDs_.begin(), knownMarkerIDs_.end(), markerID) != knownMarkerIDs_.end();
}

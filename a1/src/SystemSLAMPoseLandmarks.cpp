#include <cmath>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemSLAM.h"
#include "SystemSLAMPoseLandmarks.h"

SystemSLAMPoseLandmarks::SystemSLAMPoseLandmarks(const GaussianInfo<double> & density)
    : SystemSLAM(density)
{

}

SystemSLAM * SystemSLAMPoseLandmarks::clone() const
{
    return new SystemSLAMPoseLandmarks(*this);
}

std::size_t SystemSLAMPoseLandmarks::numberLandmarks() const
{
    return (density.dim() - 12)/6;
}

std::size_t SystemSLAMPoseLandmarks::landmarkPositionIndex(std::size_t idxLandmark) const
{
    assert(idxLandmark < numberLandmarks());
    return 12 + 6*idxLandmark;    
}

void SystemSLAMPoseLandmarks::addKnownMarkerID(int markerID)
{
    if (std::find(knownMarkerIDs_.begin(), knownMarkerIDs_.end(), markerID) == knownMarkerIDs_.end())
    {
        knownMarkerIDs_.push_back(markerID);
    }
}

bool SystemSLAMPoseLandmarks::isMarkerKnown(int markerID) const
{
    return std::find(knownMarkerIDs_.begin(), knownMarkerIDs_.end(), markerID) != knownMarkerIDs_.end();
}

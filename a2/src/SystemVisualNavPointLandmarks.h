#ifndef SYSTEMVISUALNAVPOINTLANDMARKS_H
#define SYSTEMVISUALNAVPOINTLANDMARKS_H

#include <vector>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemVisualNav.h"

/*
 * State containing body velocities, body pose, previous pose, and 3D point landmarks
 *
 *     [ nu   ] Body translational and angular velocities (body-fixed)
 * x = [ eta  ] Body position and orientation (world-fixed)
 *     [ zeta ] Body position and orientation at previous time step (world-fixed)
 *     [ rP1Nn ] 3D Point landmark 1 position (world-fixed)
 *     [ rP2Nn ] 3D Point landmark 2 position (world-fixed)
 *     [ ...  ] ...
 *
 */
class SystemVisualNavPointLandmarks : public SystemVisualNav
{
public:
    explicit SystemVisualNavPointLandmarks(const GaussianInfo<double> & density);
    SystemVisualNav * clone() const override;
    
    // Landmark tracking
    std::vector<int> consecutiveFailures_;  // Track consecutive association failures per landmark
};

#endif


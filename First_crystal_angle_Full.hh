#pragma once

/*
 * First_crystal_angle_Full.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef FIRST_CRYSTAL_ANGLE_FULL_HH_
#define FIRST_CRYSTAL_ANGLE_FULL_HH_

#include <math.h>
#include <vector>


static std::vector<double> getFullAngle(double r1x, double r1y, double r1z, double n1x, double n1y, double n1z){
        double inter_pro, angle, r2x, r2y, r2z;

        inter_pro = r1x * n1x + r1y * n1y + r1z * n1z;

        angle = asin(-inter_pro);
        r2x = r1x - 2 * inter_pro * n1x;
        r2y = r1y - 2 * inter_pro * n1y;
        r2z = r1z - 2 * inter_pro * n1z;

        std::vector<double> res;
        res.push_back(angle);
        res.push_back(r2x);
        res.push_back(r2y);
        res.push_back(r2z);

        return res;
}


#endif /* FIRST_CRYSTAL_ANGLE_FULL_HH_ */

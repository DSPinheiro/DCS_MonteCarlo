/*
 * Second_crystal_angle_full.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef SECOND_CRYSTAL_ANGLE_FULL_HH_
#define SECOND_CRYSTAL_ANGLE_FULL_HH_

#include <math.h>
#include <vector>

static std::vector<double> getFullAngle2(double r2x, double r2y, double r2z, double n2x, double n2y, double n2z){

        double inter_pro, angle, r3x, r3y, r3z;

        inter_pro = r2x * n2x + r2y * n2y + r2z * n2z;

        angle = asin(-inter_pro);
        r3x = r2x - 2 * inter_pro * n2x;
        r3y = r2y - 2 * inter_pro * n2y;
        r3z = r2z - 2 * inter_pro * n2z;

        std::vector<double> res;
        res.push_back(angle);
        res.push_back(r3x);
        res.push_back(r3y);
        res.push_back(r3z);

        return res;

}


#endif /* SECOND_CRYSTAL_ANGLE_FULL_HH_ */

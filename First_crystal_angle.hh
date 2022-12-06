#pragma once

/*
 * First_crystal_angle.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef FIRST_CRYSTAL_ANGLE_HH_
#define FIRST_CRYSTAL_ANGLE_HH_

#include <math.h>

static double getFirstApproxAngle(double tetaref, double tetadir, double sin_fi, double cos_fi, double tilt_C1, double squa_tilt1){

        double temp_sin, sinte;

        temp_sin = sin(tetadir + tetaref) * cos_fi;
        sinte = temp_sin * (1 - squa_tilt1) + sin_fi * tilt_C1;

        return asin(sinte);
}


#endif /* FIRST_CRYSTAL_ANGLE_HH_ */

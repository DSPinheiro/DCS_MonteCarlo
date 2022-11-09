/*
 * First_crystal_angle_1_Approximation.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef FIRST_CRYSTAL_ANGLE_1_APPROXIMATION_HH_
#define FIRST_CRYSTAL_ANGLE_1_APPROXIMATION_HH_

#include <math.h>

static double getFullApproximationAngle(double tetaref, double tetadir, double cos_e, double tan_e, double fidir, double tilt_C1){
        return tetaref + tetadir - (pow(fidir, 2) + pow(tilt_C1, 2)) * tan_e + fidir * tilt_C1 / cos_e;
}


#endif /* FIRST_CRYSTAL_ANGLE_1_APPROXIMATION_HH_ */

/*
 * Second_crystal_angle_aproximation.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef SECOND_CRYSTAL_ANGLE_APROXIMATION_HH_
#define SECOND_CRYSTAL_ANGLE_APROXIMATION_HH_


#include <math.h>

static double getFullApproximationAngle2(double tetaref, double tetadir, double delrot, double cos_e, double tan_e, double cos2_e, double fidir, double tilt_C1, double tilt_C2, bool Parallel){

        if(Parallel)
                return tetaref + tetadir - delrot - tan_e * (pow(fidir, 2) + pow(tilt_C2, 2) + 4 * tilt_C1 * (tilt_C1 + tilt_C2)) + fidir * (tilt_C2 + 2 * tilt_C1) / cos_e;
        else
                return tetaref - tetadir + delrot - tan_e * (pow(fidir, 2) + pow(tilt_C2, 2) + 4 * tilt_C1 * tilt_C2 - 4 * cos2_e * pow(tilt_C1, 2)) + fidir * (tilt_C2 - 2 * cos2_e * tilt_C1) / cos_e;

}


#endif /* SECOND_CRYSTAL_ANGLE_APROXIMATION_HH_ */

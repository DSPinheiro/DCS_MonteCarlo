#pragma once

/*
 * Second_crystal_angle.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef SECOND_CRYSTAL_ANGLE_HH_
#define SECOND_CRYSTAL_ANGLE_HH_

#include <math.h>

static double getFirstApproxAngle2(double tetaref, double tetadir, double delrot, double sin_fi, double cos_fi, double squa_tilt2, double cosdel, double cosdel_othe, double cosdel_teta, double cosdel_teta_othe, double sin_teref_tedi, bool Parallel){

    double temp_sin, sinte;

    if(Parallel){
        temp_sin = sin(tetadir + tetaref - delrot) * cos_fi;
        sinte = temp_sin * (1 - squa_tilt2) + cosdel * sin_fi - cosdel_othe * cos_fi * sin_teref_tedi;
        return asin(sinte);
    }else{
        temp_sin = sin(- tetadir + tetaref + delrot) * cos_fi;
        sinte = temp_sin * (1 - squa_tilt2) + cosdel * sin_fi - cosdel_teta_othe * cos_fi * sin_teref_tedi;
        return asin(sinte);
    }

}



#endif /* SECOND_CRYSTAL_ANGLE_HH_ */

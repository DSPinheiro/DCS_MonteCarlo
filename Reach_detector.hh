/*
 * Reach_detector.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef REACH_DETECTOR_HH_
#define REACH_DETECTOR_HH_

#include <math.h>


static bool Reached(double z, double y, double tetadir_temp, double fidir_temp, double L_temp, double z_max, double z_min, double y_max, double y_min){
        double z_temp, y_temp;

        z_temp = z + tan(fidir_temp) * L_temp;
        y_temp = y + tan(tetadir_temp) * L_temp;

        if(z_temp < z_max and z_temp > z_min and y_temp < y_max and y_temp > y_min)
                return true;
        else
                return false;
}


#endif /* REACH_DETECTOR_HH_ */

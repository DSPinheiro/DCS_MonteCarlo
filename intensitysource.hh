/*
 * Intensity_source.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef INTENSITY_SOURCE_HH_
#define INTENSITY_SOURCE_HH_

#include <iostream>
#include <fstream>

#include "simuGlobals.hh"
#include "splint.hh"
#include "Boxmuller.hh"

class IntensitySource{
    public:
        static double getEnergy(double a_lamds_uni, double db_lamds_uni, double tw_d);
};

#endif /* INTENSITY_SOURCE_HH_ */

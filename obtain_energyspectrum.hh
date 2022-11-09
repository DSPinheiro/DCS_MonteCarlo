/*
 * Obtain_EnergySpectrum.hh
 *
 *  Created on: Oct 29, 2020
 *      Author: danip
 */

#ifndef OBTAIN_ENERGYSPECTRUM_HH_
#define OBTAIN_ENERGYSPECTRUM_HH_

#include <iostream>
#include <fstream>
#include <string.h>
#include <sys/stat.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <algorithm>

#include "simuGlobals.hh"
#include "spline.hh"

class Obtain_EnergySpectrum{
    public:
        static void Read_EnergySpectrum();
};


#endif /* OBTAIN_ENERGYSPECTRUM_HH_ */

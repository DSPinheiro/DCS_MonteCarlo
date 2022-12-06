#pragma once

/*
 * Obtain_data_fitting.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef OBTAIN_DATA_FITTING_HH_
#define OBTAIN_DATA_FITTING_HH_

#include "simuGlobals.hh"

class Obtain_data_fitting{
    public:
        static void FitData(
            int numbins,
            double angle_para,
            int toint_para,
            double angle_anti,
            int toint_anti);
};

#endif /* OBTAIN_DATA_FITTING_HH_ */

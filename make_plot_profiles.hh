#pragma once

/*
 * Make_plot_profiles.hh
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */

#ifndef MAKE_PLOT_PROFILES_HH_
#define MAKE_PLOT_PROFILES_HH_

#include <iostream>

#include "simuGlobals.hh"
#include "simulationmain.h"


class Make_plot_profiles{
    public:
        static void plotProfiles(
            double energy_para,
            double angle_para,
            int toint_para,
            double energy_anti,
            double angle_anti,
            int toint_anti,
            int n_plot, 
            SimulationMain* w);
};


#endif /* MAKE_PLOT_PROFILES_HH_ */

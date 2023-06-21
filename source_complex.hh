#pragma once

/*
 * Source_complex.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef SOURCE_COMPLEX_HH_
#define SOURCE_COMPLEX_HH_


#include <iostream>
#include <stdlib.h>
#include <algorithm>

#include "simulationmain.h"

#include "simuGlobals.hh"
#include "obtain_time.hh"
#include "make_plot_profiles.hh"


class Source_complex{

public:
    static void run_Source(SimulationMain* w);
};


#endif /* SOURCE_COMPLEX_HH_ */

#pragma once

/*
 * Source_simple.hh
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */

#ifndef SOURCE_SIMPLE_HH_
#define SOURCE_SIMPLE_HH_

#include <iostream>
#include <stdlib.h>

#include "simuGlobals.hh"
#include "Obtain_d_Lattice_temp.hh"
#include "obtain_misalign.hh"
#include "limits_horizontal_diverge.hh"
#include "obtain_time.hh"
#include "Reach_detector.hh"
#include "First_crystal_angle_Full.hh"
#include "First_crystal_angle.hh"
#include "First_crystal_angle_1_Approximation.hh"
#include "intensitysource.hh"
#include "gaussi_rockin.hh"
#include "Second_crystal_angle_Full.hh"
#include "Second_crystal_angle.hh"
#include "Second_crystal_angle_aproximation.hh"
#include "obtain_data_fitting.hh"

class Source_simple{
    public:
        static void run_Source();
};


#endif /* SOURCE_SIMPLE_HH_ */

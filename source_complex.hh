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

#include "FindLoc.hh"
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
#include "obtainvertical.hh"
#include "get_new_temp_for_bin.hh"
#include "Obtain_ypzp.hh"
#include "obtain_curved_hor_corr.hh"
#include "make_pointcryst.hh"
#include "make_graph_imageplates.hh"
#include "make_plot_profiles.hh"

class Source_complex{

public:
    static void run_Source(SimulationMain *w = nullptr);
};


#endif /* SOURCE_COMPLEX_HH_ */

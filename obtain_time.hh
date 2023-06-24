#pragma once

/*
 * Obtain_time.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef OBTAIN_TIME_HH_
#define OBTAIN_TIME_HH_

#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

#include "simuGlobals.hh"
#include "simulation_interface.h"

class Obtain_time{
public:
    static std::vector<int> simuTime(
        int First_call,
        int process_remain,
        int int_time,
        int int_time_mili,
        SimulationInterface* w);
};



#endif /* OBTAIN_TIME_HH_ */

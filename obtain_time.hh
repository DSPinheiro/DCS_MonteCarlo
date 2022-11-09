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
#include <string>
#include <fstream>

#include "simuGlobals.hh"

#include "simulationmain.h"

class Obtain_time{
public:
    static std::vector<int> simuTime(int First_call, int process_remain, int int_time, int int_time_mili, SimulationMain *w = nullptr);
};



#endif /* OBTAIN_TIME_HH_ */

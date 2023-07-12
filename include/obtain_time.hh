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
    Obtain_time(SimulationInterface *w);
    ~Obtain_time();

    void simuTime(bool finished, double pctDone, SimulationInterface* w);

private:
    std::chrono::system_clock::time_point startingTime;
    std::chrono::duration<float,std::nano> updateTotalDuration;
};



#endif /* OBTAIN_TIME_HH_ */

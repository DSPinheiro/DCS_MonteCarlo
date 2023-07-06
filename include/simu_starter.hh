#pragma once

/*
 * Double_Crystal_diffraction.hh
 *
 *  Created on: Oct 30, 2020
 *      Author: danip
 */

#ifndef SIMU_STARTER_HH_
#define SIMU_STARTER_HH_

#include <iostream>
#include <sys/stat.h>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#include "simuGlobals.hh"
#include "source_simple.hh"
#include "source_complex.hh"
#include "mask_test_c2.hh"

class Simu_Starter{
    public:
        static void Make_Simu(SimulationInterface* w);

        inline static std::string Out_angle(int n){
            if(n == 0)
                return "Glancing angle evaluated without approximation ";
            else if(n == 1)
                return "Glancing angle evaluated with partial approximation ";
            else
                return "Glancing angle evaluated with full approximation ";
        };
};


#endif /* SIMU_STARTER_HH_ */

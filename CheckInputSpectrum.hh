#pragma once

/*
 * CheckInputSpectrum.hh
 *
 *  Created on: Oct 29, 2020
 *      Author: danip
 */

#ifndef CHECKINPUTSPECTRUM_HH_
#define CHECKINPUTSPECTRUM_HH_

#include <stdexcept>
#include <iostream>

#include "simuGlobals.hh"
#include "Obtain_d_Lattice_temp.hh"
#include "obtainvertical.hh"

class CheckInputSpectrum{
    public:
        static bool CheckSpectrum(std::string unit);

};


#endif /* CHECKINPUTSPECTRUM_HH_ */

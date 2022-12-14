#pragma once

/*
 * Limits_horizontal_diverge.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef LIMITS_HORIZONTAL_DIVERGE_HH_
#define LIMITS_HORIZONTAL_DIVERGE_HH_



#include <algorithm>

#include "simuGlobals.hh"

class Limits_horizontal_diverge{
    public:
        static std::vector<double> getLims(double tetaref, double delrot_min, double delrot_max, double fi_max, double teta_max, double teta_min);
};


#endif /* LIMITS_HORIZONTAL_DIVERGE_HH_ */

#pragma once

/*
 * Obtain_curve_responce.hh
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */

#ifndef OBTAIN_CURVE_RESPONCE_HH_
#define OBTAIN_CURVE_RESPONCE_HH_


#include <iostream>
#include <sys/stat.h>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#include "simuGlobals.hh"
#include "spline.hh"

class Obtain_curve_responce{
    public:
        static void Read_CurveResponce();
};



#endif /* OBTAIN_CURVE_RESPONCE_HH_ */

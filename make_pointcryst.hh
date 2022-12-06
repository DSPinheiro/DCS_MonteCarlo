#pragma once

/*
 * Make_pointcryst.hh
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */

#ifndef MAKE_POINTCRYST_HH_
#define MAKE_POINTCRYST_HH_


#include <iostream>

#include "simuGlobals.hh"

class Make_pointcryst{
    public:
        static void Make(int crystal, double y, double z);
};


#endif /* MAKE_POINTCRYST_HH_ */

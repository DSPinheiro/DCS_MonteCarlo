/*
 * MakefitVoigt.hh
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */

#ifndef MAKEFITVOIGT_HH_
#define MAKEFITVOIGT_HH_

#include <stdio.h>
#include <iostream>
#include <fstream>


#include "simuGlobals.hh"
#include "TrueVoig.hh"
#include "mrqmin.hh"
#include "PseuVoigt.hh"


class MakefitVoigt{
    public:
        static void fit(bool Parallel);
};


#endif /* MAKEFITVOIGT_HH_ */

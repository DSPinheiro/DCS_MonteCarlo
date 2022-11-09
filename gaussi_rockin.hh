/*
 * Gaussi_rockin.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef GAUSSI_ROCKIN_HH_
#define GAUSSI_ROCKIN_HH_

#include "simuGlobals.hh"
#include "splint.hh"


class Gaussi_rockin{
    public:
        static bool getReflection(double angle, double tetabra1, bool type_crystal, bool poli_p);
};



#endif /* GAUSSI_ROCKIN_HH_ */

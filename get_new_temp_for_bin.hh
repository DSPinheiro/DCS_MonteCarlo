/*
 * Get_new_temp_for_bin.hh
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */

#ifndef GET_NEW_TEMP_FOR_BIN_HH_
#define GET_NEW_TEMP_FOR_BIN_HH_

#include <iostream>


#include "simuGlobals.hh"

class Get_new_temp_for_bins{
    public:
        static double getNewTemp(int bin_tem, int bin_fas, double pha_tem);
};


#endif /* GET_NEW_TEMP_FOR_BIN_HH_ */

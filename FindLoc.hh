#pragma once

/*
 * FindLoc.hh
 *
 *  Created on: Oct 30, 2020
 *      Author: danip
 */

#ifndef FINDLOC_HH_
#define FINDLOC_HH_

#include <vector>
#include <stdexcept>

static int FindLoc(std::vector<double> array, double value){
        for(unsigned int i = 0; i < array.size(); i++){
                if(array[i] == value){
                        return i;
                }
        }

        throw std::runtime_error("Value not found in array.");

}



#endif /* FINDLOC_HH_ */

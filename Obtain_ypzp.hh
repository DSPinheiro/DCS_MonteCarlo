#pragma once

/*
 * Obtain_ypzp.hh
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */

#ifndef OBTAIN_YPZP_HH_
#define OBTAIN_YPZP_HH_

#include <vector>
#include <math.h>

static std::vector<double> getYZ(double r_temp, double sin_tetap_temp, double cos_tetap_temp, double tan_tetadir_temp, double tan_fidir_temp, double L_temp){
        std::vector<double> res;

        res.push_back(r_temp * cos_tetap_temp + tan_tetadir_temp * L_temp);
        res.push_back(r_temp * sin_tetap_temp + tan_fidir_temp * L_temp);

        return res;

        //this was in the original code but isnt used (?)
        /*if(yp_temp == 0){

                cos_tetap_temp = 0;
                if(zp_temp > 0)
                        sin_tetap_temp = 1;
                else
                        sin_tetap_temp = -1;
        }else{
                temp = zp_temp / yp_temp;

                if(yp_temp > 0){
                        sin_tetap_temp = temp / sqrt(1 + pow(temp, 2));
                        cos_tetap_temp = 1 / sqrt(1 + pow(temp, 2));
                }else{
                        sin_tetap_temp = -temp / sqrt(1 + pow(temp, 2));
                        cos_tetap_temp = -1 / sqrt(1 + pow(temp, 2));
                }
        }*/


}


#endif /* OBTAIN_YPZP_HH_ */

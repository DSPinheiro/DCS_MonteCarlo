#pragma once

/*
 * covsrt.hh
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */

#ifndef COVSRT_HH_
#define COVSRT_HH_

#include <vector>
#include <iostream>

static void cov_srt(std::vector<std::vector<double>> covar, int npc, int ma, int* ia, int mfit){

    double swap;

    int k;

    for(int i = mfit - 1; i < ma; i++){
        for(int j = 0; j <= i; j++){
            covar.at(i).at(j) = 0;
            covar.at(j).at(i) = 0;
        }
    }

    k = mfit;

    for(int j = ma - 1; j >= 0; j--){
        if(ia[j] != 0){
            k--;
            for(int i = 0; i < ma; i++){
                swap = covar.at(i).at(k);
                covar.at(i).at(k) = covar.at(i).at(j);
                covar.at(i).at(j) = swap;
            }

            for(int i = 0; i < ma; i++){
                swap = covar.at(k).at(i);
                covar.at(k).at(i) = covar.at(j).at(i);
                covar.at(j).at(i) = swap;
            }
        }
    }



}



#endif /* COVSRT_HH_ */

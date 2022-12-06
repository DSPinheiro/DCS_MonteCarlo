/*
 * Obtain_Curved_Hor_Corr.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */

#include "obtain_curved_hor_corr.hh"

using namespace std;



vector<double> Obtain_Curved_Hor_Corr::horCorr(
    double y_pro_C1,
    double y_max_C1,
    double z_pro_C1,
    double z_max_C1,
    bool type_c){

    double R_cur_crys_t;

    if(type_c)
        R_cur_crys_t = CurvedCrystalInput.R_cur_crys_1;
    else
        R_cur_crys_t = CurvedCrystalInput.R_cur_crys_2;

    vector<double> res;

    res.push_back(cos(2 * R_cur_crys_t * y_pro_C1 * (1 - pow((z_pro_C1 / z_max_C1), 2)) / pow(y_max_C1, 2)));
    res.push_back(sin(2 * R_cur_crys_t * y_pro_C1 * (1 - pow((z_pro_C1 / z_max_C1), 2)) / pow(y_max_C1, 2)));

    return res;

}

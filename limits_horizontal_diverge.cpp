/*
 * Limits_horizontal_diverge.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "limits_horizontal_diverge.hh"


using namespace std;

extern GeoParameters GeoParameters;

extern double min_angle_resp, max_angle_resp;

vector<double> Limits_horizontal_diverge::getLims(double tetaref, double delrot_min, double delrot_max, double fi_max, double teta_max, double teta_min){

    double tan_e, cos_e, teta_min_temp, corre_term1, corre_term2, teta_max_temp;

    double teta_max_out, teta_min_out;

    tan_e = tan(tetaref) / 2;
    cos_e = cos(tetaref);

    corre_term1 = tan_e * (pow(fi_max, 2) + pow(GeoParameters.tilt_C1, 2));
    corre_term2 = fi_max * abs(GeoParameters.tilt_C1) / cos_e;

    teta_min_temp = min_angle_resp + delrot_min + corre_term1 - corre_term2;

    if(teta_min_temp > teta_min)
        teta_min_out = teta_min_temp;

    teta_max_temp = max_angle_resp + delrot_max + corre_term1 + corre_term2;

    if(teta_max_temp < teta_max)
        teta_max_out = teta_max_temp;

    vector<double> res;
    res.push_back(teta_min_out);
    res.push_back(teta_max_out);

    return res;
}

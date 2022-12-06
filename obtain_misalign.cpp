/*
 * Obtain_misalign.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "obtain_misalign.hh"


using namespace std;

vector<double> Obtain_misalign::misalign(double Dis_total){

    double dev, term_arc, Geo_para_matr[4][4], len, hei, tetaref, div, fi_temp, fi_max, fi_min, z_max, z_min;

    fi_max = 10;
    fi_min = -10;

    z_max = GeolengthelementsInput.zdetc / 2;
    z_min = -GeolengthelementsInput.zdetc / 2;
    tetaref = M_PI / 2 - teta_crys1;

    Geo_para_matr[0][0] = GeolengthelementsInput.S_aper / 2;
    Geo_para_matr[0][1] = -GeolengthelementsInput.S_aper / 2;
    Geo_para_matr[0][2] = GeoParapathlengthsInput.dist_T_Cr1;
    Geo_para_matr[0][3] = GeoParametersInput.xsi;

    Geo_para_matr[1][0] = GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[1][1] = -GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[1][2] = GeoParapathlengthsInput.dist_T_Cr1 / 2;
    Geo_para_matr[1][3] = GeoParametersInput.xsi - 2 * GeoParametersInput.tilt_C1 * sin(tetaref) * GeoParapathlengthsInput.dist_Cr1_Cr2 / (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2);

    Geo_para_matr[2][0] = GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[2][1] = -GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[2][2] = GeoParapathlengthsInput.dist_Cr2_Det;
    Geo_para_matr[2][3] = GeoParametersInput.xsi - 2 * GeoParametersInput.tilt_C1 * sin(tetaref) * (GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det) /
            (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det) - 2 * GeoParametersInput.tilt_C2 * sin(tetaref) *
        GeoParapathlengthsInput.dist_Cr2_Det / (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det);

    Geo_para_matr[3][0] = z_max;
    Geo_para_matr[3][1] = - z_max;

    if(GeoParapathlengthsInput.type_source == "P")
        term_arc = z_max;
    else
        term_arc = (GeolengthelementsInput.S_aper + GeolengthelementsInput.zdetc) / 2;


    fi_max = atan(term_arc / Dis_total) + GeoParametersInput.xsi;
    fi_min = - atan(term_arc / Dis_total) + GeoParametersInput.xsi;

    dev = term_arc + Dis_total * tan(GeoParametersInput.xsi - atan(term_arc / Dis_total));

    z_max = GeolengthelementsInput.zdetc / 2 + dev;
    z_min = -GeolengthelementsInput.zdetc / 2 + dev;

    vector<double> res;
    res.push_back(fi_max);
    res.push_back(fi_min);
    res.push_back(z_max);
    res.push_back(z_min);

    return res;
}

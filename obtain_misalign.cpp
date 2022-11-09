/*
 * Obtain_misalign.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "obtain_misalign.hh"

extern Geolengthelements Geolengthelements;
extern GeoParapathlengths GeoParapathlengths;
extern GeoParameters GeoParameters;

extern double teta_crys1;

using namespace std;

vector<double> Obtain_misalign::misalign(double Dis_total){

    double dev, term_arc, Geo_para_matr[4][4], len, hei, tetaref, div, fi_temp, fi_max, fi_min, z_max, z_min;

    fi_max = 10;
    fi_min = -10;

    z_max = Geolengthelements.zdetc / 2;
    z_min = - Geolengthelements.zdetc / 2;
    tetaref = M_PI / 2 - teta_crys1;

    Geo_para_matr[0][0] = Geolengthelements.S_aper / 2;
    Geo_para_matr[0][1] = - Geolengthelements.S_aper / 2;
    Geo_para_matr[0][2] = GeoParapathlengths.dist_T_Cr1;
    Geo_para_matr[0][3] = GeoParameters.xsi;

    Geo_para_matr[1][0] = Geolengthelements.z_first_crys / 2;
    Geo_para_matr[1][1] = - Geolengthelements.z_first_crys / 2;
    Geo_para_matr[1][2] = GeoParapathlengths.dist_T_Cr1 / 2;
    Geo_para_matr[1][3] = GeoParameters.xsi - 2 * GeoParameters.tilt_C1 * sin(tetaref) * GeoParapathlengths.dist_Cr1_Cr2 / (GeoParapathlengths.dist_T_Cr1 + GeoParapathlengths.dist_Cr1_Cr2);

    Geo_para_matr[2][0] = Geolengthelements.z_first_crys / 2;
    Geo_para_matr[2][1] = - Geolengthelements.z_first_crys / 2;
    Geo_para_matr[2][2] = GeoParapathlengths.dist_Cr2_Det;
    Geo_para_matr[2][3] = GeoParameters.xsi - 2 * GeoParameters.tilt_C1 * sin(tetaref) * (GeoParapathlengths.dist_Cr1_Cr2 + GeoParapathlengths.dist_Cr2_Det) /
            (GeoParapathlengths.dist_T_Cr1 + GeoParapathlengths.dist_Cr1_Cr2 + GeoParapathlengths.dist_Cr2_Det) - 2 * GeoParameters.tilt_C2 * sin(tetaref) *
            GeoParapathlengths.dist_Cr2_Det / (GeoParapathlengths.dist_T_Cr1 + GeoParapathlengths.dist_Cr1_Cr2 + GeoParapathlengths.dist_Cr2_Det);

    Geo_para_matr[3][0] = z_max;
    Geo_para_matr[3][1] = - z_max;

    if(GeoParapathlengths.type_source == "P")
        term_arc = z_max;
    else
        term_arc = (Geolengthelements.S_aper + Geolengthelements.zdetc) / 2;


    fi_max = atan(term_arc / Dis_total) + GeoParameters.xsi;
    fi_min = - atan(term_arc / Dis_total) + GeoParameters.xsi;

    dev = term_arc + Dis_total * tan(GeoParameters.xsi - atan(term_arc / Dis_total));

    z_max = Geolengthelements.zdetc / 2 + dev;
    z_min = - Geolengthelements.zdetc / 2 + dev;

    vector<double> res;
    res.push_back(fi_max);
    res.push_back(fi_min);
    res.push_back(z_max);
    res.push_back(z_min);

    return res;
}

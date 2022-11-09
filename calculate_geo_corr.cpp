/*
 * calculate_geo_corre.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */

#include "calculate_geo_corr.hh"

extern double teta_crys1;

extern GeoParapathlengths GeoParapathlengths;
extern UserSettings UserSettings;
extern Geolengthelements Geolengthelements;

double vert_div_corr;

void calculate_geo_corre::geo_corre(){
    double Dis_total, teta_ref, tan_e, con_deg, con_rad;

    con_deg = 180 / M_PI;
    con_rad = M_PI / 180;

    Dis_total = GeoParapathlengths.LT_aper + GeoParapathlengths.dist_T_Cr1 + GeoParapathlengths.dist_Cr1_Cr2 + GeoParapathlengths.dist_Cr2_Det;
    teta_ref = M_PI / 2 - teta_crys1 * con_rad;
    tan_e = tan(teta_ref);

    if(UserSettings.Make_Vertical){
        if(GeoParapathlengths.type_source == "UC")
            vert_div_corr = tan_e * (pow(Geolengthelements.S_aper, 2) + pow(Geolengthelements.zdetc, 2) / (24 * pow(Dis_total, 2)));
        else if(UserSettings.Make_Vertical)
            vert_div_corr = tan_e * (pow(Geolengthelements.z_aper, 2) + pow(Geolengthelements.zdetc, 2) / (24 * pow(Dis_total, 2)));

        vert_div_corr = vert_div_corr * con_deg;

    }else
        vert_div_corr = 0;

}

/*
 * calculate_geo_corre.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */

#include "calculate_geo_corr.hh"


void calculate_geo_corre::geo_corre(){
    
    double Dis_total, teta_ref, tan_e, con_deg, con_rad;

    con_deg = 180 / M_PI;
    con_rad = M_PI / 180;

    Dis_total = GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det;
    teta_ref = M_PI / 2 - teta_crys1 * con_rad;
    tan_e = tan(teta_ref);

    if(UserSettingsInput.Make_Vertical){
        if(GeoParapathlengthsInput.type_source == "UC")
            vert_div_corr = tan_e * (pow(GeolengthelementsInput.S_aper, 2) + pow(GeolengthelementsInput.zdetc, 2) / (24 * pow(Dis_total, 2)));
        else if(UserSettingsInput.Make_Vertical)
            vert_div_corr = tan_e * (pow(GeolengthelementsInput.z_aper, 2) + pow(GeolengthelementsInput.zdetc, 2) / (24 * pow(Dis_total, 2)));

        vert_div_corr *= con_deg;

    }else
        vert_div_corr = 0;

}

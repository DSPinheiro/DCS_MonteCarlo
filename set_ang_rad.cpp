/*
 * Set_ang_rad.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */


#include "set_ang_rad.hh"

extern GeoParameters GeoParameters;

extern bool never_set_angle;

extern double teta_crys1;

void Set_ang_rad::Set_angs(){
    if(never_set_angle){
        //tetaref *= convrad;
        GeoParameters.xsi *= convrad;
        teta_crys1 *= convrad;
        GeoParameters.tilt_C1 *= convrad;
        GeoParameters.tilt_C2 *= convrad;

        //Old vars
        //stan_V_C1 *= convrad;
        //stan_H_C1 *= convrad;
        //stan_V_C2 *= convrad;
        //stan_H_C2 *= convrad;

        GeoParameters.OffsetRotCry1 *= convrad;

        GeoParameters.teta_table *= convrad;
        GeoParameters.teta_detec_para *= convrad;
        GeoParameters.teta_detec_anti *= convrad;

        //Old vars
        //y_first_crys_2 /= 2;
        //z_first_crys_2 /= 2;
        //step_x_hist_y_crys /= 5;
        //step_x_hist_z_crys /= 5;
        //S_aper_D_2 *= 2;

        never_set_angle = false;
    }
}

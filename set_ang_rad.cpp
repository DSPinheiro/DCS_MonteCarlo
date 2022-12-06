/*
 * Set_ang_rad.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */


#include "set_ang_rad.hh"



void Set_ang_rad::Set_angs(){
    
    if(never_set_angle){
        //tetaref *= convrad;
        GeoParametersInput.xsi *= convrad;
        teta_crys1 *= convrad;
        GeoParametersInput.tilt_C1 *= convrad;
        GeoParametersInput.tilt_C2 *= convrad;

        //Old vars
        //stan_V_C1 *= convrad;
        //stan_H_C1 *= convrad;
        //stan_V_C2 *= convrad;
        //stan_H_C2 *= convrad;

        GeoParametersInput.OffsetRotCry1 *= convrad;

        GeoParametersInput.teta_table *= convrad;
        GeoParametersInput.teta_detec_para *= convrad;
        GeoParametersInput.teta_detec_anti *= convrad;

        //Old vars
        //y_first_crys_2 /= 2;
        //z_first_crys_2 /= 2;
        //step_x_hist_y_crys /= 5;
        //step_x_hist_z_crys /= 5;
        //S_aper_D_2 *= 2;

        never_set_angle = false;
    }
}

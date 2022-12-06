/*
 * Make_graph_imageplate.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "make_graph_imageplates.hh"


using namespace std;



void Make_graph_imageplate::initPlates(){

    int int_plot[6];

    double ratio_xy[6];
    
    double y_detc_2, z_detc_2;

    y_detc_2 = GeolengthelementsInput.ydetc / 2;
    z_detc_2 = GeolengthelementsInput.zdetc / 2;


    int_plot[0] = NumberRaysInput.nbeams - 1;
    int_plot[1] = NumberRaysInput.nbeams + 1;
    int_plot[2] = (int)(NumberRaysInput.nbeams / 100) + 1;
    int_plot[3] = 10;
    int_plot[4] = (int)(NumberRaysInput.nbeams / 100) + 1;
    int_plot[5] = 10;

    if(GraphOptionsInput.make_imageC1_After_refle){
        int_plot[1] = 500;
        int_plot[2] = 1;
        int_plot[3] = 10;
        int_plot[4] = 1;
        int_plot[5] = 10;
    }

    if(GeoParapathlengthsInput.type_source == "UR"){
        max_plot_x[0] = GeolengthelementsInput.y_aper;
        max_plot_x[1] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[2] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[3] = y_detc_2;
        max_plot_x[4] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[5] = y_detc_2;

        max_plot_y[0] = 3 * GeolengthelementsInput.z_aper;
        max_plot_y[1] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[2] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[3] = z_detc_2;
        max_plot_y[4] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[5] = z_detc_2;
    }else if(GeoParapathlengthsInput.type_source == "UC"){
        max_plot_x[0] = GeolengthelementsInput.S_aper;
        max_plot_x[1] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[2] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[3] = y_detc_2;
        max_plot_x[4] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[5] = y_detc_2;

        max_plot_y[0] = GeolengthelementsInput.S_aper;
        max_plot_y[1] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[2] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[3] = z_detc_2;
        max_plot_y[4] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[5] = z_detc_2;
    }


    ratio_xy[0] = max_plot_x[0] / max_plot_y[0];
    ratio_xy[1] = max_plot_x[1] / max_plot_y[1];
    ratio_xy[2] = max_plot_x[2] / max_plot_y[2];
    ratio_xy[3] = max_plot_x[3] / max_plot_y[3];
    ratio_xy[4] = max_plot_x[4] / max_plot_y[4];
    ratio_xy[5] = max_plot_x[5] / max_plot_y[5];


    //TODO
    //more gui stuff


}

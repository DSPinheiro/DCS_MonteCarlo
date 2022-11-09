/*
 * Make_graph_imageplate.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "make_graph_imageplates.hh"


using namespace std;

extern Geolengthelements Geolengthelements;
extern numberrays numberrays;
extern Graph_options Graph_options;
extern GeoParapathlengths GeoParapathlengths;


int int_plot[6];
double max_plot_x[6], max_plot_y[6], ratio_xy[6];

void Make_graph_imageplate::initPlates(){

    double y_detc_2, z_detc_2;

    y_detc_2 = Geolengthelements.ydetc / 2;
    z_detc_2 = Geolengthelements.zdetc / 2;


    int_plot[0] = numberrays.nbeams - 1;
    int_plot[1] = numberrays.nbeams + 1;
    int_plot[2] = (int)(numberrays.nbeams / 100) + 1;
    int_plot[3] = 10;
    int_plot[4] = (int)(numberrays.nbeams / 100) + 1;
    int_plot[5] = 10;

    if(Graph_options.make_imageC1_After_refle){
        int_plot[1] = 500;
        int_plot[2] = 1;
        int_plot[3] = 10;
        int_plot[4] = 1;
        int_plot[5] = 10;
    }

    if(GeoParapathlengths.type_source == "UR"){
        max_plot_x[0] = Geolengthelements.y_aper;
        max_plot_x[1] = Geolengthelements.y_first_crys / 2;
        max_plot_x[2] = Geolengthelements.y_first_crys / 2;
        max_plot_x[3] = y_detc_2;
        max_plot_x[4] = Geolengthelements.y_first_crys / 2;
        max_plot_x[5] = y_detc_2;

        max_plot_y[0] = 3 * Geolengthelements.z_aper;
        max_plot_y[1] = Geolengthelements.z_first_crys / 2;
        max_plot_y[2] = Geolengthelements.z_first_crys / 2;
        max_plot_y[3] = z_detc_2;
        max_plot_y[4] = Geolengthelements.z_first_crys / 2;
        max_plot_y[5] = z_detc_2;
    }else if(GeoParapathlengths.type_source == "UC"){
        max_plot_x[0] = Geolengthelements.S_aper;
        max_plot_x[1] = Geolengthelements.y_first_crys / 2;
        max_plot_x[2] = Geolengthelements.y_first_crys / 2;
        max_plot_x[3] = y_detc_2;
        max_plot_x[4] = Geolengthelements.y_first_crys / 2;
        max_plot_x[5] = y_detc_2;

        max_plot_y[0] = Geolengthelements.S_aper;
        max_plot_y[1] = Geolengthelements.z_first_crys / 2;
        max_plot_y[2] = Geolengthelements.z_first_crys / 2;
        max_plot_y[3] = z_detc_2;
        max_plot_y[4] = Geolengthelements.z_first_crys / 2;
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

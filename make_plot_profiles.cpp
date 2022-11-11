/*
 * Make_plot_profiles.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "make_plot_profiles.hh"

using namespace std;


extern Geometry Geometry;
extern UserSettings UserSettings;
extern Graph_options Graph_options;
extern plotparameters plotparameters;

bool transmi_2crys = false;

extern double max_plot_x[6], max_plot_y[6], max_hist[6], max_plot[2][2], min_plot[2][2];

extern double hist_image_plate_source[n_his_ima][n_his_ima], hist_image_plate_crystal1[n_his_ima][n_his_ima], hist_image_plate_crystal2_para[n_his_ima][n_his_ima], hist_image_plate_crystal2_anti[n_his_ima][n_his_ima], hist_image_plate_detc_para[n_his_ima][n_his_ima], hist_image_plate_detc_anti[n_his_ima][n_his_ima];

extern double counts_sour, counts_C1, counts_C2_para, counts_detc_para, counts_C2_anti, counts_detc_anti;

vector<plot> plot_para, plot_anti;

void Make_plot_profiles::plotProfiles(double angle_para, int toint_para, double angle_anti, int toint_anti, int n_plot, SimulationMain *w){

    if(plot_para.size() < (unsigned int)plotparameters.nubins)
        plot_para.resize(plotparameters.nubins);
    if(plot_anti.size() < (unsigned int)plotparameters.nubins)
        plot_anti.resize(plotparameters.nubins);

    double hist_image[n_his_ima][n_his_ima];

    string X_plots[6] = {"Entrance of spectrometer		",
                         "first crystal					",
                         "second crystal parallel		",
                         "detector parallel				",
                         "second crystal antiparallel	",
                         "detector antiparallel			"};

    string X_leg, legen_counts_temp[6];

    int counts, col1_x, col1_y, col1_x_va[3] = {180, 1000, 2050};

    double temp, step_x, step_y, max_plot_x_temp, max_plot_y_temp, max_plot_z_temp, step_x_hist, step_y_hist, step_z_hist;


    if(Geometry.mode_bragg_geo)
        copy(begin(legen_counts), end(legen_counts), begin(legen_counts_temp));
    else{
        if(transmi_2crys)
            copy(begin(legen_counts), end(legen_counts), begin(legen_counts_temp));
        else
            copy(begin(legen_counts_1C), end(legen_counts_1C), begin(legen_counts_temp));
    }


    plot_para.at(n_plot - 1).x = angle_para;
    plot_anti.at(n_plot - 1).x = angle_anti;
    plot_para.at(n_plot - 1).y = (double)toint_para;
    plot_anti.at(n_plot - 1).y = (double)toint_anti;
    plot_para.at(n_plot - 1).error = sqrt((double)toint_para);
    plot_anti.at(n_plot - 1).error = sqrt((double)toint_anti);


    //TODO
    //gui stuff

    for(int i = 0; i < 2; i++){
        if(i == 0)
            temp = plot_para.at(n_plot - 1).y + plot_para.at(n_plot - 1).error;
        else if(i == 1)
            temp = plot_anti.at(n_plot - 1).y + plot_anti.at(n_plot - 1).error;

        if(max_plot[i][1] < temp + 0.00001)
            max_plot[i][1] = temp + 0.00001;

        if(max_plot[i][1] > temp - 0.00001)
            max_plot[i][1] = temp - 0.00001;

        step_x = (-min_plot[i][0] + max_plot[i][0]) / 5;
        step_y = (-min_plot[i][1] + max_plot[i][1]) / 5;

        if(not (n_plot == 1)){

            if(n_plot % 6 == 0){
                emit w->changePlots(plot_para, plot_anti);
            }

        }

    }


    if(n_plot % 6 == 0){

        if(not UserSettings.Simple_simu and Graph_options.make_image_plates){
            //TODO
            //gui stuff

            for(int crystal = 0; crystal < 6; crystal++){
                max_plot_x_temp = max_plot_x[crystal];
                max_plot_y_temp = max_plot_y[crystal];


                X_leg = X_plots[crystal];

                if(crystal == 0){
                    copy(&hist_image_plate_source[0][0], &hist_image_plate_source[0][0]+n_his_ima*n_his_ima,&hist_image[0][0]);
                    counts = counts_sour;
                    col1_y = 400 + crystal * 600;
                    col1_x = col1_x_va[0];
                }else if(crystal == 1){
                    copy(&hist_image_plate_crystal1[0][0], &hist_image_plate_crystal1[0][0]+n_his_ima*n_his_ima,&hist_image[0][0]);
                    counts = counts_C1;
                    col1_y = 450 + crystal * 500;
                    col1_x = col1_x_va[0] - 40;
                }else if(crystal == 2){
                    copy(&hist_image_plate_crystal2_para[0][0], &hist_image_plate_crystal2_para[0][0]+n_his_ima*n_his_ima,&hist_image[0][0]);
                    counts = counts_C2_para;
                    col1_y = 450 + crystal * 600;
                    col1_x = col1_x_va[1] - 20;
                }else if(crystal == 3){
                    copy(&hist_image_plate_detc_para[0][0], &hist_image_plate_detc_para[0][0]+n_his_ima*n_his_ima,&hist_image[0][0]);
                    counts = counts_detc_para;
                    col1_y = 400 + crystal * 500;
                    col1_x = col1_x_va[1] + 120;
                }else if(crystal == 4){
                    copy(&hist_image_plate_crystal2_anti[0][0], &hist_image_plate_crystal2_anti[0][0]+n_his_ima*n_his_ima,&hist_image[0][0]);
                    counts = counts_C2_anti;
                    col1_y = 450 + crystal * 600;
                    col1_x = col1_x_va[2] - 40;
                }else if(crystal == 5){
                    copy(&hist_image_plate_detc_anti[0][0], &hist_image_plate_detc_anti[0][0]+n_his_ima*n_his_ima,&hist_image[0][0]);
                    counts = counts_detc_anti;
                    col1_y = 400 + crystal * 500;
                    col1_x = col1_x_va[2] + 85;
                }



                //TODO
                //gui reports


                if(max_hist[crystal] == 0)
                    max_plot_z_temp = 1;
                else
                    max_plot_z_temp = max_hist[crystal];


                step_x_hist = max_plot_x_temp * 0.4;
                step_y_hist = max_plot_y_temp * 0.4;
                step_z_hist = max_plot_z_temp / 5;

                //TODO
                //gui plotting
                emit w->changePlates(hist_image, max_plot_z_temp, crystal);

            }

            //TODO
            //if(UserSettings.print_scan)
                //gui

        }
    }

}

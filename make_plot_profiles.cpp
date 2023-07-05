/*
 * Make_plot_profiles.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "make_plot_profiles.hh"

using namespace std;


void Make_plot_profiles::plotProfiles(
    double energy_para,
    double angle_para,
    int toint_para,
    double energy_anti,
    double angle_anti,
    int toint_anti,
    int n_plot,
    int &counts_sour,
    int &counts_C1,
    int &counts_C2_para,
    int &counts_detc_para,
    int &counts_C2_anti,
    int &counts_detc_anti,
    SimulationInterface *w){

    bool transmi_2crys = false;

    plot emptyPlot = {0.0,0.0,0.0,0.0};
    if(plot_para.size() < (unsigned int)PlotParametersInput.nubins)
    {
        plot_para.resize(PlotParametersInput.nubins, emptyPlot);
    }
    if(plot_anti.size() < (unsigned int)PlotParametersInput.nubins)
    {
        plot_anti.resize(PlotParametersInput.nubins, emptyPlot);
    }

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


    if(GeometryInput.mode_bragg_geo)
    {
        legen_counts_temp[0] = legen_counts_1;
        legen_counts_temp[1] = legen_counts_2;
        legen_counts_temp[2] = legen_counts_3;
        legen_counts_temp[3] = legen_counts_4;
        legen_counts_temp[4] = legen_counts_5;
        legen_counts_temp[5] = legen_counts_6;
    }
    else
    {
        if(transmi_2crys)
        {
            legen_counts_temp[0] = legen_counts_1;
            legen_counts_temp[1] = legen_counts_2;
            legen_counts_temp[2] = legen_counts_3;
            legen_counts_temp[3] = legen_counts_4;
            legen_counts_temp[4] = legen_counts_5;
            legen_counts_temp[5] = legen_counts_6;
        }
        else
        {
            legen_counts_temp[0] = legen_counts_1C_1;
            legen_counts_temp[1] = legen_counts_1C_2;
            legen_counts_temp[2] = legen_counts_1C_3;
            legen_counts_temp[3] = legen_counts_1C_4;
            legen_counts_temp[4] = legen_counts_1C_5;
            legen_counts_temp[5] = legen_counts_1C_6;
        }
    }

    
    plot_para.at(n_plot - 1).x = angle_para;
    plot_para.at(n_plot - 1).energy = energy_para;
    plot_anti.at(n_plot - 1).x = angle_anti;
    plot_anti.at(n_plot - 1).energy = energy_anti;
    plot_para.at(n_plot - 1).y = (double)toint_para;
    plot_anti.at(n_plot - 1).y = (double)toint_anti;
    plot_para.at(n_plot - 1).error = sqrt((double)toint_para);
    plot_anti.at(n_plot - 1).error = sqrt((double)toint_anti);


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

        if(! (n_plot == 1)){

            if(n_plot % 6 == 0){
                SimulationInterface::Plots plots = { plot_para, plot_anti };
                emit w->changePlotsSignal(plots);
            }

        }

    }


    if(n_plot % 6 == 0){

        if(!UserSettingsInput.Simple_simu && GraphOptionsInput.make_image_plates){
            for(int crystal = 0; crystal < 6; crystal++){
                max_plot_x_temp = max_plot_x[crystal];
                max_plot_y_temp = max_plot_y[crystal];


                X_leg = X_plots[crystal];

                if(crystal == 0){
                    copy(&hist_image_plate_source[0][0], &hist_image_plate_source[0][0]+ n_his_ima* n_his_ima,&hist_image[0][0]);
                    counts = counts_sour;
                    col1_y = 400 + crystal * 600;
                    col1_x = col1_x_va[0];
                }else if(crystal == 1){
                    copy(&hist_image_plate_crystal1[0][0], &hist_image_plate_crystal1[0][0]+ n_his_ima* n_his_ima,&hist_image[0][0]);
                    counts = counts_C1;
                    col1_y = 450 + crystal * 500;
                    col1_x = col1_x_va[0] - 40;
                }else if(crystal == 2){
                    copy(&hist_image_plate_crystal2_para[0][0], &hist_image_plate_crystal2_para[0][0]+ n_his_ima* n_his_ima,&hist_image[0][0]);
                    counts = counts_C2_para;
                    col1_y = 450 + crystal * 600;
                    col1_x = col1_x_va[1] - 20;
                }else if(crystal == 3){
                    copy(&hist_image_plate_detc_para[0][0], &hist_image_plate_detc_para[0][0]+ n_his_ima* n_his_ima,&hist_image[0][0]);
                    counts = counts_detc_para;
                    col1_y = 400 + crystal * 500;
                    col1_x = col1_x_va[1] + 120;
                }else if(crystal == 4){
                    copy(&hist_image_plate_crystal2_anti[0][0], &hist_image_plate_crystal2_anti[0][0]+ n_his_ima* n_his_ima,&hist_image[0][0]);
                    counts = counts_C2_anti;
                    col1_y = 450 + crystal * 600;
                    col1_x = col1_x_va[2] - 40;
                }else if(crystal == 5){
                    copy(&hist_image_plate_detc_anti[0][0], &hist_image_plate_detc_anti[0][0]+ n_his_ima* n_his_ima,&hist_image[0][0]);
                    counts = counts_detc_anti;
                    col1_y = 400 + crystal * 500;
                    col1_x = col1_x_va[2] + 85;
                }


                if(max_hist[crystal] == 0)
                    max_plot_z_temp = 1;
                else
                    max_plot_z_temp = max_hist[crystal];


                step_x_hist = max_plot_x_temp * 0.4;
                step_y_hist = max_plot_y_temp * 0.4;
                step_z_hist = max_plot_z_temp / 5;

                SimulationInterface::Plates plates;
                plates.crystal = crystal;
                // Not ideal (but at least does not crash)
                // cgodinho 15/05/2023 (there is not much tought into this anyways)
                memcpy(&plates.hist_image[0][0], &hist_image[0][0], n_his_ima * n_his_ima * sizeof(double));
                plates.max_z = max_plot_z_temp;
                emit w->changePlatesSignal(plates);

            }
        }
    }

}

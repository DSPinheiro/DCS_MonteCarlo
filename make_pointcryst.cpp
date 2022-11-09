/*
 * Make_pointcryst.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "make_pointcryst.hh"


using namespace std;

extern double max_plot_x[6], max_plot_y[6], max_hist[6];

extern double hist_image_plate_source[n_his_ima][n_his_ima], hist_image_plate_crystal1[n_his_ima][n_his_ima], hist_image_plate_crystal2_para[n_his_ima][n_his_ima], hist_image_plate_crystal2_anti[n_his_ima][n_his_ima], hist_image_plate_detc_para[n_his_ima][n_his_ima], hist_image_plate_detc_anti[n_his_ima][n_his_ima];

extern double counts_sour, counts_C1, counts_C2_para, counts_detc_para, counts_C2_anti, counts_detc_anti;

double hist_image[n_his_ima][n_his_ima];


void Make_pointcryst::Make(int crystal, double y, double z){

    double step_z_hist, step_y_hist, max_plot_x_temp, max_plot_y_temp;
    int col2_x = 1400, nx, ny, counts;


    max_plot_x_temp = max_plot_x[crystal - 1];
    max_plot_y_temp = max_plot_y[crystal - 1];

    nx = (int)(nm2 * y / max_plot_x_temp + np2);
    ny = (int)(nm2 * z / max_plot_y_temp + np2);


    if(crystal == 1){

        if(not (nx > n_his_ima or ny > n_his_ima or nx <= 0 or ny <= 0)){
            if(ny == 0){
                //TODO
                //implement report on gui
            }

            hist_image_plate_source[nx - 1][ny - 1]++;
            counts_sour++;

            if(max_hist[crystal - 1] < hist_image_plate_source[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_source[nx - 1][ny - 1];
        }
    }else if(crystal == 2){

        if(not (nx > n_his_ima or ny > n_his_ima or nx <= 0 or ny <= 0)){
            hist_image_plate_crystal1[nx - 1][ny - 1]++;
            counts_C1++;

            if(max_hist[crystal - 1] < hist_image_plate_crystal1[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_crystal1[nx - 1][ny - 1];
        }
    }else if(crystal == 3){

        if(not (nx > n_his_ima or ny > n_his_ima or nx <= 0 or ny <= 0)){
            hist_image_plate_crystal2_para[nx - 1][ny - 1]++;
            counts_C2_para++;

            if(max_hist[crystal - 1] < hist_image_plate_crystal2_para[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_crystal2_para[nx - 1][ny - 1];
        }
    }else if(crystal == 4){

        if(not (nx > n_his_ima or ny > n_his_ima or nx <= 0 or ny <= 0)){
            hist_image_plate_detc_para[nx - 1][ny - 1]++;
            counts_detc_para++;

            if(max_hist[crystal - 1] < hist_image_plate_detc_para[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_detc_para[nx - 1][ny - 1];
        }
    }else if(crystal == 5){

        if(not (nx > n_his_ima or ny > n_his_ima or nx <= 0 or ny <= 0)){
            hist_image_plate_crystal2_anti[nx - 1][ny - 1]++;
            counts_C2_anti++;

            if(max_hist[crystal - 1] < hist_image_plate_crystal2_anti[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_crystal2_anti[nx - 1][ny - 1];
        }
    }else if(crystal == 6){

        if(not (nx > n_his_ima or ny > n_his_ima or nx <= 0 or ny <= 0)){
            hist_image_plate_crystal2_anti[nx - 1][ny - 1]++;
            counts_detc_anti++;

            if(max_hist[crystal - 1] < hist_image_plate_detc_anti[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_detc_anti[nx - 1][ny - 1];
        }
    }


}

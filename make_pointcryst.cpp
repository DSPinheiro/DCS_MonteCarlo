/*
 * Make_pointcryst.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "make_pointcryst.hh"


using namespace std;


double hist_image[n_his_ima][n_his_ima];


void Make_pointcryst::Make(int crystal, double y, double z){

    double step_z_hist, step_y_hist, max_plot_x_temp, max_plot_y_temp;
    int col2_x = 1400, nx, ny, counts;


    max_plot_x_temp = max_plot_x[crystal - 1];
    max_plot_y_temp = max_plot_y[crystal - 1];

    nx = (int)(nm2 * y / max_plot_x_temp + np2);
    ny = (int)(nm2 * z / max_plot_y_temp + np2);


    if(crystal == 1){

        if(! (nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)){
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

        if(! (nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)){
            hist_image_plate_crystal1[nx - 1][ny - 1]++;
            counts_C1++;

            if(max_hist[crystal - 1] < hist_image_plate_crystal1[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_crystal1[nx - 1][ny - 1];
        }
    }else if(crystal == 3){

        if(! (nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)){
            hist_image_plate_crystal2_para[nx - 1][ny - 1]++;
            counts_C2_para++;

            if(max_hist[crystal - 1] < hist_image_plate_crystal2_para[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_crystal2_para[nx - 1][ny - 1];
        }
    }else if(crystal == 4){

        if(! (nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)){
            hist_image_plate_detc_para[nx - 1][ny - 1]++;
            counts_detc_para++;

            if(max_hist[crystal - 1] < hist_image_plate_detc_para[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_detc_para[nx - 1][ny - 1];
        }
    }else if(crystal == 5){

        if(! (nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)){
            hist_image_plate_crystal2_anti[nx - 1][ny - 1]++;
            counts_C2_anti++;

            if(max_hist[crystal - 1] < hist_image_plate_crystal2_anti[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_crystal2_anti[nx - 1][ny - 1];
        }
    }else if(crystal == 6){

        if(! (nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)){
            hist_image_plate_detc_anti[nx - 1][ny - 1]++;
            counts_detc_anti++;

            if(max_hist[crystal - 1] < hist_image_plate_detc_anti[nx - 1][ny - 1])
                max_hist[crystal - 1] = hist_image_plate_detc_anti[nx - 1][ny - 1];
        }
    }


}

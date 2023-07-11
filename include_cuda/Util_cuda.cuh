#pragma once

#ifndef UTIL_CUDA_CUH_
#define UTIL_CUDA_CUH_


#include "../include/simuConsts.h"


namespace Util_CUDA {
    struct MakeParameters
    {
        double *max_plot_x;
        double *max_plot_y;
        float *hist_image_plate_source;
        float *hist_image_plate_crystal1;
        float *hist_image_plate_crystal2_para;
        float *hist_image_plate_crystal2_anti;
        float *hist_image_plate_detc_para;
        float *hist_image_plate_detc_anti;
        float *max_hist;

        int &counts_sour;
        int &counts_C1;
        int &counts_C2_para;
        int &counts_detc_para;
        int &counts_C2_anti;
        int &counts_detc_anti;

        __device__ MakeParameters(double *max_plot_x,
                        double *max_plot_y,
                        float *hist_image_plate_source,
                        float *hist_image_plate_crystal1,
                        float *hist_image_plate_crystal2_para,
                        float *hist_image_plate_crystal2_anti,
                        float *hist_image_plate_detc_para,
                        float *hist_image_plate_detc_anti,
                        float *max_hist,
                        int &counts_sour,
                        int &counts_C1,
                        int &counts_C2_para,
                        int &counts_detc_para,
                        int &counts_C2_anti,
                        int &counts_detc_anti) : 
                        max_plot_x (max_plot_x),
                        max_plot_y (max_plot_y),
                        hist_image_plate_source (hist_image_plate_source),
                        hist_image_plate_crystal1 (hist_image_plate_crystal1),
                        hist_image_plate_crystal2_para (hist_image_plate_crystal2_para),
                        hist_image_plate_crystal2_anti (hist_image_plate_crystal2_anti),
                        hist_image_plate_detc_para (hist_image_plate_detc_para),
                        hist_image_plate_detc_anti (hist_image_plate_detc_anti),
                        max_hist (max_hist),
                        counts_sour (counts_sour),
                        counts_C1 (counts_C1),
                        counts_C2_para (counts_C2_para),
                        counts_detc_para (counts_detc_para),
                        counts_C2_anti (counts_C2_anti),
                        counts_detc_anti (counts_detc_anti)
                        {}
    };

    struct EnergyParameters
    {
        int make_more_lines;
        bool Do_background;
        double p1_ener;
        double p2_ener;
        double p3_ener;
        energy_vecs_CUDA *Energy_spectrum_vectors;
        pick picks[5];
        double gauss_Doop;
    };

    struct ReflectionParameters
    {
        double *available_energies;
        double *min_angle_resp;
        double *max_angle_resp;
        bool mka_poli;
        plotresponc_vecs_CUDA **Crystal_Responces;
        int64_t size;
    };

    __device__ void setupRand(curandState *state, unsigned int seed, int block, int thread);
    __device__ double GaussianBox(curandState *state, double sta_dev, double mean, bool box_muller = false);
    __device__ double2 getYZ(double r_temp, double sin_tetap_temp, double cos_tetap_temp, double tan_tetadir_temp, double tan_fidir_temp, double L_temp);
    __device__ void Make(int crystal, double y, double z, MakeParameters *pars);
    __device__ double2 horCorr(double y_pro_C1, double y_max_C1, double z_pro_C1, double z_max_C1, bool type_c, double R_cur_crys_1, double R_cur_crys_2);
    __device__ double4 getFullAngle(double r1x, double r1y, double r1z, double n1x, double n1y, double n1z);
    __device__ double getEnergy(curandState *state, double a_lamds_uni, double db_lamds_uni, double tw_d, EnergyParameters pars);
    __device__ double splint_te(int64_t size, double *xa, double *ya, double *y2a, double x);
    __device__ bool getReflection(curandState *state, double angle, double tetabra, double lamda, bool type_crystal, ReflectionParameters pars, bool poli_p = false);
    __device__ double4 getFullAngle2(double r2x, double r2y, double r2z, double n2x, double n2y, double n2z);
}


#endif /* UTIL_CUDA_CUH_ */
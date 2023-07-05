#pragma once

#ifndef PARALLEL_BIN_CUH_
#define PARALLEL_BIN_CUH_

#include <curand.h>
#include <curand_kernel.h>

#include "simuConsts.h"

class ParallelBin {

public:
    struct BinParameters
    {
        unsigned int seed;
        double teta_table_thread;
        double p;
        double tetadir;
        double sin_tetadir;
        double cos_tetadir;
        double tan_tetadir;
        double cos_tetadirCry1;
        double fidir;
        double sin_fidir;
        double cos_fidir;
        double tan_fidir;
        double cos_fidirtilt;
        double z;
        double y;
        double r;
        double tetap;
        double sin_tetap;
        double cos_tetap;
        double var_temp;
        double y_pro_C1;
        double z_pro_C1;
        double rx;
        double ry;
        double rz;
        double Costeta_CHC;
        double Sinteta_CHC;
        double n1x;
        double n1y;
        double angle;
        double r2x;
        double r2y;
        double r2z;
        double lamda;
        double tetabra1;
        bool first_crystal_reach;
        double rx_rot;
        double ry_rot;
        double cos_fidirtilt2_para;
        double cos_fidirtilt2_anti;
        double corr_dis;
        double sin_tetatab_del_dir;
        double tetabra2;
        double n2x_para;
        double n2y_para;
        double r3x;
        double r3y;
        double r3z;
        bool sec_crystal_Parallel_reach;
        double rx_rot_sec;
        double ry_rot_sec;
        double tetadir_det;
        double tan_tetadir_det;
        double cos_tetadir_det;
        double fidir_det;
        double tan_fidir_det;
        double cos_fidir_det;
        double corr_dis_d_pa;
        double y_pro_C1_d_pa;
        double cos_tetap_det;
        double sin_tetap_det;
        double r_det;
        double y_det;
        double z_det;
        double n2x_anti;
        double n2y_anti;
        bool sec_crystal_Antiparallel_reach;
        int total_current_bins;

        int curr3DEventCount_para;
        int curr3DEventCount_anti;

        BinParameters() : 
        total_current_bins (0), sec_crystal_Antiparallel_reach (false), sec_crystal_Parallel_reach (false), curr3DEventCount_para(0), curr3DEventCount_anti(0)
        {}

    };

    struct SetupParameters
    {
        double teta_table;
        double table_resolution;
        bool make_table_noise;
        bool Make_Horizontal;
        bool Make_Vertical;
        double xsi;
        char type_source[3];
        double S_shi_ver_B;
        double S_shi_hor_B;
        double S_sour;
        double z_sour;
        double y_sour;
        double S_shi_ver_A;
        double S_shi_hor_A;
        bool make_image_plates;
        bool make_imageC1_After_refle;
        bool Curve_crystall;
        bool mka_poli;
        bool see_para;
        bool make_imageC2_After_refle;
        int number_events;
        bool see_anti;
        double del_teta_L;
        double teta_min_L;
        double del_fi_L;
        double fi_min_L;
        double tilt_C1_temp;
        double S_sour_2;
        double S_aper_R_2;
        double S_aper_var_2;
        double z_sour_2;
        double y_sour_2;
        double S_aper_sqr;
        double LT_aper_Db;
        double y_max_C1;
        double y_min_C1;
        double z_max_C1;
        double z_min_C1;
        double n1x_temp;
        double n1y_temp;
        double n1z;
        double a_lamds_uni;
        double b_lamds_uni;
        double tw_d1_para;
        double relationP_S;
        double cos_tetartab;
        double sin_tetartab;
        double tilt_C2_para_temp;
        double tilt_C2_anti_temp;
        double cos_difteC1_Ta;
        double sin_difteC1_Ta;
        double dist_Cr1_Cr2_Db;
        double tetaref;
        double y_max_C2;
        double y_min_C2;
        double z_max_C2;
        double z_min_C2;
        double tw_d2_para;
        double n2x_para_temp;
        double n2y_para_temp;
        double n2z_para;
        double cos_tetartabdete_para;
        double sin_tetartabdete_para;
        double cos_difteC2_det_para;
        double sin_difteC2_det_para;
        double dist_Cr2_det_Db;
        double ydetc_2_max;
        double ydetc_2_min;
        double zdetc_2_max;
        double zdetc_2_min;
        double tw_d2_anti;
        double n2x_anti_temp;
        double n2y_anti_temp;
        double n2z_anti;
        double cos_tetartabdete_anti;
        double sin_tetartabdete_anti;
        int64_t total_expexted_bins;
        int numbins;
        double delrot;
        double R_cur_crys_1;
        double R_cur_crys_2;
        int make_more_lines;
        bool Do_background;
        double p1_ener;
        double p2_ener;
        double p3_ener;
        energy_vecs_CUDA Energy_spectrum_vectors;
        pick picks[5];
        double gauss_Doop;
        double *available_energies;
        double *min_angle_resp;
        double *max_angle_resp;
        plotresponc_vecs_CUDA *Crystal_Responces;


        SetupParameters(double tetaref, double tw_d1_para, double tw_d2_para, double tw_d2_anti, double S_aper_R_2, double S_aper_var_2, double S_aper_sqr,
                        double S_sour_2, double z_sour_2, double y_sour_2, double zdetc_2_max, double zdetc_2_min, double ydetc_2_max, double ydetc_2_min,
                        double delrot, int64_t total_expexted_bins) :
                        tetaref (tetaref), tw_d1_para (tw_d1_para), tw_d2_para (tw_d2_para), tw_d2_anti (tw_d2_anti), S_aper_R_2 (S_aper_R_2), S_aper_var_2 (S_aper_var_2),
                        S_aper_sqr (S_aper_sqr), S_sour_2 (S_sour_2), z_sour_2 (z_sour_2), y_sour_2 (y_sour_2), zdetc_2_max (zdetc_2_max), zdetc_2_min (zdetc_2_min),
                        ydetc_2_max (ydetc_2_max), ydetc_2_min (ydetc_2_min), delrot (delrot), total_expexted_bins (total_expexted_bins), numbins (0)
        {}
    };

    struct ReductionVars
    {
        #if defined(OPENMP) || defined(CUDA)
        double energy_sum_para_thread;
        double energy_sum_anti_thread;
        #else
        std::vector<double> *energy_sum_para;
        std::vector<double> *energy_sum_anti;
        #endif
        int toint_para;
        int toint_anti;
        int counts_sour;
        int counts_C1;
        int counts_C2_para;
        int counts_C2_anti;
        int counts_detc_para;
        int counts_detc_anti;

        double max_plot_x[6];
        double max_plot_y[6];
        double hist_image_plate_source[n_his_ima][n_his_ima];
        double hist_image_plate_crystal1[n_his_ima][n_his_ima];
        double hist_image_plate_crystal2_para[n_his_ima][n_his_ima];
        double hist_image_plate_crystal2_anti[n_his_ima][n_his_ima];
        double hist_image_plate_detc_para[n_his_ima][n_his_ima];
        double hist_image_plate_detc_anti[n_his_ima][n_his_ima];
        double max_hist[6];

        double **eventsToTrace_paraCUDA;
        double **eventsToTrace_antiCUDA;

        int Events3D;

        ReductionVars(int Events3D) :
        #if defined(OPENMP) || defined(CUDA)
        Events3D(Events3D), energy_sum_para_thread (0), energy_sum_anti_thread (0), 
        #endif
        toint_para (0), toint_anti (0), counts_sour (0), counts_C1 (0), counts_C2_para (0), counts_C2_anti (0), counts_detc_para (0), counts_detc_anti (0)
        {
            eventsToTrace_paraCUDA = new double*[Events3D];
            eventsToTrace_antiCUDA = new double*[Events3D];

            for(int i = 0; i < Events3D; i++)
            {
                eventsToTrace_paraCUDA[i] = new double[12];
                eventsToTrace_antiCUDA[i] = new double[12];
                eventsToTrace_paraCUDA = {NULL};
                eventsToTrace_antiCUDA = {NULL};
            }
        }

        __host__ __device__ ReductionVars(
            #if defined(OPENMP) || defined(CUDA)
            double energy_sum_para_thread, double energy_sum_anti_thread,
            #endif
            int toint_para, int toint_anti, int counts_sour, int counts_C1, int counts_C2_para, int counts_C2_anti, int counts_detc_para,
            int counts_detc_anti, double max_plot_x[6], double max_plot_y[6], double hist_image_plate_source[n_his_ima][n_his_ima],
            double hist_image_plate_crystal1[n_his_ima][n_his_ima], double hist_image_plate_crystal2_para[n_his_ima][n_his_ima],
            double hist_image_plate_crystal2_anti[n_his_ima][n_his_ima], double hist_image_plate_detc_para[n_his_ima][n_his_ima],
            double hist_image_plate_detc_anti[n_his_ima][n_his_ima], double max_hist[6],
            double **eventsToTrace_paraCUDA, double **eventsToTrace_antiCUDA) :
        #ifdef OPENMP
        energy_sum_para_thread (energy_sum_para_thread), energy_sum_anti_thread (energy_sum_anti_thread),
        #endif
        toint_para (toint_para), toint_anti (toint_anti), counts_sour (counts_sour), counts_C1 (counts_C1), counts_C2_para (counts_C2_para),
        counts_C2_anti (counts_C2_anti), counts_detc_para (counts_detc_para), counts_detc_anti (counts_detc_anti), eventsToTrace_paraCUDA (eventsToTrace_paraCUDA),
        eventsToTrace_antiCUDA (eventsToTrace_antiCUDA)
        {}


        __host__ __device__
        ReductionVars operator+(ReductionVars a){
            double new_max_x[6];
            new_max_x[0] = (a.max_plot_x[0] > max_plot_x[0]) ? a.max_plot_x[0] : max_plot_x[0];
            new_max_x[1] = (a.max_plot_x[1] > max_plot_x[1]) ? a.max_plot_x[1] : max_plot_x[1];
            new_max_x[2] = (a.max_plot_x[2] > max_plot_x[2]) ? a.max_plot_x[2] : max_plot_x[2];
            new_max_x[3] = (a.max_plot_x[3] > max_plot_x[3]) ? a.max_plot_x[3] : max_plot_x[3];
            new_max_x[4] = (a.max_plot_x[4] > max_plot_x[4]) ? a.max_plot_x[4] : max_plot_x[4];
            new_max_x[5] = (a.max_plot_x[5] > max_plot_x[5]) ? a.max_plot_x[5] : max_plot_x[5];
            
            double new_max_y[6];
            new_max_y[0] = (a.max_plot_y[0] > max_plot_y[0]) ? a.max_plot_y[0] : max_plot_y[0];
            new_max_y[1] = (a.max_plot_y[1] > max_plot_y[1]) ? a.max_plot_y[1] : max_plot_y[1];
            new_max_y[2] = (a.max_plot_y[2] > max_plot_y[2]) ? a.max_plot_y[2] : max_plot_y[2];
            new_max_y[3] = (a.max_plot_y[3] > max_plot_y[3]) ? a.max_plot_y[3] : max_plot_y[3];
            new_max_y[4] = (a.max_plot_y[4] > max_plot_y[4]) ? a.max_plot_y[4] : max_plot_y[4];
            new_max_y[5] = (a.max_plot_y[5] > max_plot_y[5]) ? a.max_plot_y[5] : max_plot_y[5];
            
            double new_max[6];
            new_max[0] = (a.max_hist[0] > max_hist[0]) ? a.max_hist[0] : max_hist[0];
            new_max[1] = (a.max_hist[1] > max_hist[1]) ? a.max_hist[1] : max_hist[1];
            new_max[2] = (a.max_hist[2] > max_hist[2]) ? a.max_hist[2] : max_hist[2];
            new_max[3] = (a.max_hist[3] > max_hist[3]) ? a.max_hist[3] : max_hist[3];
            new_max[4] = (a.max_hist[4] > max_hist[4]) ? a.max_hist[4] : max_hist[4];
            new_max[5] = (a.max_hist[5] > max_hist[5]) ? a.max_hist[5] : max_hist[5];
            
            double new_hist_image_plate_source[n_his_ima][n_his_ima];
            double new_hist_image_plate_crystal1[n_his_ima][n_his_ima];
            double new_hist_image_plate_crystal2_para[n_his_ima][n_his_ima];
            double new_hist_image_plate_crystal2_anti[n_his_ima][n_his_ima];
            double new_hist_image_plate_detc_para[n_his_ima][n_his_ima];
            double new_hist_image_plate_detc_anti[n_his_ima][n_his_ima];
            
            for(int i = 0; i < n_his_ima * n_his_ima; i++)
            {
                int x = i / n_his_ima;
                int y = i % n_his_ima;

                new_hist_image_plate_source[x][y] = a.hist_image_plate_source[x][y] + hist_image_plate_source[x][y];
                new_hist_image_plate_crystal1[x][y] = a.hist_image_plate_crystal1[x][y] + hist_image_plate_crystal1[x][y];
                new_hist_image_plate_crystal2_para[x][y] = a.hist_image_plate_crystal2_para[x][y] + hist_image_plate_crystal2_para[x][y];
                new_hist_image_plate_crystal2_anti[x][y] = a.hist_image_plate_crystal2_anti[x][y] + hist_image_plate_crystal2_anti[x][y];
                new_hist_image_plate_detc_para[x][y] = a.hist_image_plate_detc_para[x][y] + hist_image_plate_detc_para[x][y];
                new_hist_image_plate_detc_anti[x][y] = a.hist_image_plate_detc_anti[x][y] + hist_image_plate_detc_anti[x][y];
            }

            
            double **new_events_para = new double*[Events3D];
            double **new_events_anti = new double*[Events3D];

            for(int i = 0; i < Events3D; i++)
            {
                new_events_para[i] = new double[12];
                new_events_anti[i] = new double[12];

                new_events_para[i] = {NULL};
                new_events_anti[i] = {NULL};
                

                for(int j = 0; j < 12; j++)
                {
                    new_events_para[i][j] = (a.eventsToTrace_paraCUDA[i][j] == NULL) ? eventsToTrace_paraCUDA[i][j] : a.eventsToTrace_paraCUDA[i][j];
                    new_events_anti[i][j] = (a.eventsToTrace_antiCUDA[i][j] == NULL) ? eventsToTrace_antiCUDA[i][j] : a.eventsToTrace_antiCUDA[i][j];
                }
            }

            return ParallelBin::ReductionVars{
                #if defined(OPENMP) || defined(CUDA)
                a.energy_sum_para_thread + energy_sum_para_thread,
                a.energy_sum_anti_thread + energy_sum_anti_thread,
                #endif
                a.toint_para + toint_para,
                a.toint_anti + toint_anti,
                a.counts_sour + counts_sour,
                a.counts_C1 + counts_C1,
                a.counts_C2_para + counts_C2_para,
                a.counts_C2_anti + counts_C2_anti,
                a.counts_detc_para + counts_detc_para,
                a.counts_detc_anti + counts_detc_anti,

                new_max_x,
                new_max_y,
                new_hist_image_plate_source,
                new_hist_image_plate_crystal1,
                new_hist_image_plate_crystal2_para,
                new_hist_image_plate_crystal2_anti,
                new_hist_image_plate_detc_para,
                new_hist_image_plate_detc_anti,
                new_max,

                new_events_para,
                new_events_anti
                };
        }
    };


    static void makeBin(int totalBinEventCount, const int Events3D, BinParameters *bin, SetupParameters *setup, ReductionVars *reduce);

private:
    static const int warpSize = 32;
};

__global__ static void makeBinKernel(int totalBinEventCount, curandState *randState, 
                            ParallelBin::BinParameters *bin, ParallelBin::SetupParameters *setup, ParallelBin::ReductionVars *reduce);


#endif /* PARALLEL_BIN_CUH_ */
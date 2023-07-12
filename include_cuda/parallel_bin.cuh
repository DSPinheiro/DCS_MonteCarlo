#pragma once

#ifndef PARALLEL_BIN_CUH_
#define PARALLEL_BIN_CUH_

#include <curand.h>
#include <curand_kernel.h>

#include <algorithm>

#include "../include/simuConsts.h"

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
        int numbins;
        double tw_d1_para;
        double tw_d2_para;
        double tw_d2_anti;
        double delrot;
        double tilt_C2_para_temp;
        double tilt_C2_anti_temp;
        double n2z_para;
        double n2z_anti;
        double n2x_para_temp;
        double n2y_para_temp;
        double n2x_anti_temp;
        double n2y_anti_temp;
        double cos_difteC2_det_para;
        double sin_difteC2_det_para;

        int curr3DEventCount_para;
        int curr3DEventCount_anti;

        BinParameters(double tw_d1_para, double tw_d2_para, double tw_d2_anti, double delrot) :
        tw_d1_para (tw_d1_para), tw_d2_para (tw_d2_para), tw_d2_anti (tw_d2_anti), delrot (delrot), total_current_bins (0), first_crystal_reach (false),
        sec_crystal_Antiparallel_reach (false), sec_crystal_Parallel_reach (false), curr3DEventCount_para(0), curr3DEventCount_anti(0), numbins (0)
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
        double relationP_S;
        double cos_tetartab;
        double sin_tetartab;
        double cos_difteC1_Ta;
        double sin_difteC1_Ta;
        double dist_Cr1_Cr2_Db;
        double tetaref;
        double y_max_C2;
        double y_min_C2;
        double z_max_C2;
        double z_min_C2;
        double cos_tetartabdete_para;
        double sin_tetartabdete_para;
        double dist_Cr2_det_Db;
        double ydetc_2_max;
        double ydetc_2_min;
        double zdetc_2_max;
        double zdetc_2_min;
        double cos_tetartabdete_anti;
        double sin_tetartabdete_anti;
        size_t total_expexted_bins;
        double R_cur_crys_1;
        double R_cur_crys_2;
        int make_more_lines;
        bool Do_background;
        double p1_ener;
        double p2_ener;
        double p3_ener;
        energy_vecs_CUDA *Energy_spectrum_vectors;
        pick picks[5];
        double gauss_Doop;
        double *available_energies;
        int available_energies_size;
        double *min_angle_resp;
        double *max_angle_resp;
        plotresponc_vecs_CUDA **Crystal_Responces;
        double *max_plot_x;
        double *max_plot_y;


        SetupParameters(double teta_table, double table_resolution, bool make_table_noise, bool Make_Horizontal, bool Make_Vertical,  double xsi, char type_source[3],
                        double S_shi_ver_B, double S_shi_hor_B, double S_sour, double z_sour, double y_sour, double S_shi_ver_A, double S_shi_hor_A, bool make_image_plates,
                        bool make_imageC1_After_refle, bool Curve_crystall, bool mka_poli, bool see_para, bool make_imageC2_After_refle, int number_events, bool see_anti,
                        double relationP_S, int make_more_lines, bool Do_background, double p1_ener, double p2_ener, double p3_ener, double gauss_Doop,
                        energy_vecs_CUDA *Energy_spectrum_vectors, pick picks[5], double *available_energies, int available_energies_size, double *min_angle_resp,
                        double *max_angle_resp, plotresponc_vecs_CUDA **Crystal_Responces, double tetaref, double S_aper_R_2, double S_aper_var_2, double S_aper_sqr,
                        double S_sour_2, double z_sour_2, double y_sour_2, double zdetc_2_max, double zdetc_2_min, double ydetc_2_max, double ydetc_2_min,
                        size_t total_expexted_bins) :
                        teta_table (teta_table), table_resolution (table_resolution), make_table_noise (make_table_noise), Make_Horizontal (Make_Horizontal),
                        Make_Vertical (Make_Vertical), xsi (xsi), S_shi_ver_B (S_shi_ver_B), S_shi_hor_B (S_shi_hor_B), S_sour (S_sour),
                        z_sour (z_sour), y_sour (y_sour), S_shi_ver_A (S_shi_ver_A), S_shi_hor_A (S_shi_hor_A), make_image_plates (make_image_plates),
                        make_imageC1_After_refle (make_imageC1_After_refle), Curve_crystall (Curve_crystall), mka_poli (mka_poli), see_para (see_para),
                        make_imageC2_After_refle (make_imageC2_After_refle), number_events (number_events), see_anti (see_anti), relationP_S (relationP_S),
                        make_more_lines (make_more_lines), Do_background (Do_background), p1_ener (p1_ener), p2_ener (p2_ener), p3_ener (p3_ener), gauss_Doop (gauss_Doop),
                        Energy_spectrum_vectors (Energy_spectrum_vectors), available_energies (available_energies), available_energies_size (available_energies_size),
                        min_angle_resp (min_angle_resp), max_angle_resp (max_angle_resp), Crystal_Responces (Crystal_Responces),
                        tetaref (tetaref), S_aper_R_2 (S_aper_R_2), S_aper_var_2 (S_aper_var_2),
                        S_aper_sqr (S_aper_sqr), S_sour_2 (S_sour_2), z_sour_2 (z_sour_2), y_sour_2 (y_sour_2), zdetc_2_max (zdetc_2_max), zdetc_2_min (zdetc_2_min),
                        ydetc_2_max (ydetc_2_max), ydetc_2_min (ydetc_2_min), total_expexted_bins (total_expexted_bins)
        {
            this->type_source[0] = type_source[0];
            this->type_source[1] = type_source[1];
            this->type_source[2] = type_source[2];

            this->picks[0] = picks[0];
            this->picks[1] = picks[1];
            this->picks[2] = picks[2];
            this->picks[3] = picks[3];
            this->picks[4] = picks[4];
        }
    };

    struct ReductionVars
    {
        double energy_sum_para_thread;
        double energy_sum_anti_thread;
        int toint_para;
        int toint_anti;
        int counts_sour;
        int counts_C1;
        int counts_C2_para;
        int counts_C2_anti;
        int counts_detc_para;
        int counts_detc_anti;

        float *hist_image_plate_source;
        float *hist_image_plate_crystal1;
        float *hist_image_plate_crystal2_para;
        float *hist_image_plate_crystal2_anti;
        float *hist_image_plate_detc_para;
        float *hist_image_plate_detc_anti;
        float *max_hist;

        float *eventsToTrace_paraCUDA;
        float *eventsToTrace_antiCUDA;

        int Events3D;

        ReductionVars(int Events3D) :
        Events3D (Events3D), energy_sum_para_thread (0), energy_sum_anti_thread (0),
        toint_para (0), toint_anti (0), counts_sour (0), counts_C1 (0), counts_C2_para (0), counts_C2_anti (0), counts_detc_para (0), counts_detc_anti (0)
        {
            eventsToTrace_paraCUDA = new float[Events3D * 12];
            eventsToTrace_antiCUDA = new float[Events3D * 12];

            for(int i = 0; i < Events3D * 12; i++)
            {
                eventsToTrace_paraCUDA[i] = invalidEventCoord;
                eventsToTrace_antiCUDA[i] = invalidEventCoord;
            }

            hist_image_plate_source = new float[n_his_ima * n_his_ima];
            hist_image_plate_crystal1 = new float[n_his_ima * n_his_ima];
            hist_image_plate_crystal2_para = new float[n_his_ima * n_his_ima];
            hist_image_plate_crystal2_anti = new float[n_his_ima * n_his_ima];
            hist_image_plate_detc_para = new float[n_his_ima * n_his_ima];
            hist_image_plate_detc_anti = new float[n_his_ima * n_his_ima];

            for(int i = 0; i < n_his_ima * n_his_ima; i++)
            {
                hist_image_plate_source[i] = 0;
                hist_image_plate_crystal1[i] = 0;
                hist_image_plate_crystal2_para[i] = 0;
                hist_image_plate_crystal2_anti[i] = 0;
                hist_image_plate_detc_para[i] = 0;
                hist_image_plate_detc_anti[i] = 0;
            }

            max_hist = new float[6];

            for(int i = 0; i < 6; i++)
            {
                max_hist[i] = 0;
            }
        }

        __host__ __device__ ReductionVars(
            double energy_sum_para_thread, double energy_sum_anti_thread,
            int toint_para, int toint_anti, int counts_sour, int counts_C1, int counts_C2_para, int counts_C2_anti, int counts_detc_para,
            int counts_detc_anti, float hist_image_plate_source[n_his_ima * n_his_ima],
            float hist_image_plate_crystal1[n_his_ima * n_his_ima], float hist_image_plate_crystal2_para[n_his_ima * n_his_ima],
            float hist_image_plate_crystal2_anti[n_his_ima * n_his_ima], float hist_image_plate_detc_para[n_his_ima * n_his_ima],
            float hist_image_plate_detc_anti[n_his_ima * n_his_ima], float max_hist[6],
            float *eventsToTrace_paraCUDA, float *eventsToTrace_antiCUDA) :
        energy_sum_para_thread (energy_sum_para_thread), energy_sum_anti_thread (energy_sum_anti_thread),
        toint_para (toint_para), toint_anti (toint_anti), counts_sour (counts_sour), counts_C1 (counts_C1), counts_C2_para (counts_C2_para),
        counts_C2_anti (counts_C2_anti), counts_detc_para (counts_detc_para), counts_detc_anti (counts_detc_anti), 
        eventsToTrace_paraCUDA (eventsToTrace_paraCUDA), eventsToTrace_antiCUDA (eventsToTrace_antiCUDA)
        {
            std::copy(&(this->hist_image_plate_source)[0], &(this->hist_image_plate_source)[0] + n_his_ima * n_his_ima, &hist_image_plate_source[0]);
            std::copy(&(this->hist_image_plate_crystal1)[0], &(this->hist_image_plate_crystal1)[0] + n_his_ima * n_his_ima, &hist_image_plate_crystal1[0]);
            std::copy(&(this->hist_image_plate_crystal2_para)[0], &(this->hist_image_plate_crystal2_para)[0] + n_his_ima * n_his_ima, &hist_image_plate_crystal2_para[0]);
            std::copy(&(this->hist_image_plate_crystal2_anti)[0], &(this->hist_image_plate_crystal2_anti)[0] + n_his_ima * n_his_ima, &hist_image_plate_crystal2_anti[0]);
            std::copy(&(this->hist_image_plate_detc_para)[0], &(this->hist_image_plate_detc_para)[0] + n_his_ima * n_his_ima, &hist_image_plate_detc_para[0]);
            std::copy(&(this->hist_image_plate_detc_anti)[0], &(this->hist_image_plate_detc_anti)[0] + n_his_ima * n_his_ima, &hist_image_plate_detc_anti[0]);
            std::copy(&(this->max_hist)[0], &(this->max_hist)[0] + 6, &max_hist[0]);
        }


        __host__ __device__
        ReductionVars operator+(ReductionVars a){
            float new_max[6];
            new_max[0] = (a.max_hist[0] > max_hist[0]) ? a.max_hist[0] : max_hist[0];
            new_max[1] = (a.max_hist[1] > max_hist[1]) ? a.max_hist[1] : max_hist[1];
            new_max[2] = (a.max_hist[2] > max_hist[2]) ? a.max_hist[2] : max_hist[2];
            new_max[3] = (a.max_hist[3] > max_hist[3]) ? a.max_hist[3] : max_hist[3];
            new_max[4] = (a.max_hist[4] > max_hist[4]) ? a.max_hist[4] : max_hist[4];
            new_max[5] = (a.max_hist[5] > max_hist[5]) ? a.max_hist[5] : max_hist[5];
            
            float new_hist_image_plate_source[n_his_ima * n_his_ima];
            float new_hist_image_plate_crystal1[n_his_ima * n_his_ima];
            float new_hist_image_plate_crystal2_para[n_his_ima * n_his_ima];
            float new_hist_image_plate_crystal2_anti[n_his_ima * n_his_ima];
            float new_hist_image_plate_detc_para[n_his_ima * n_his_ima];
            float new_hist_image_plate_detc_anti[n_his_ima * n_his_ima];
            
            for(int i = 0; i < n_his_ima * n_his_ima; i++)
            {
                new_hist_image_plate_source[i] = a.hist_image_plate_source[i] + hist_image_plate_source[i];
                new_hist_image_plate_crystal1[i] = a.hist_image_plate_crystal1[i] + hist_image_plate_crystal1[i];
                new_hist_image_plate_crystal2_para[i] = a.hist_image_plate_crystal2_para[i] + hist_image_plate_crystal2_para[i];
                new_hist_image_plate_crystal2_anti[i] = a.hist_image_plate_crystal2_anti[i] + hist_image_plate_crystal2_anti[i];
                new_hist_image_plate_detc_para[i] = a.hist_image_plate_detc_para[i] + hist_image_plate_detc_para[i];
                new_hist_image_plate_detc_anti[i] = a.hist_image_plate_detc_anti[i] + hist_image_plate_detc_anti[i];
            }

            
            float *new_events_para = new float[Events3D * 12];
            float *new_events_anti = new float[Events3D * 12];

            for(int i = 0; i < Events3D * 12; i++)
            {
                new_events_para[i] = (a.eventsToTrace_paraCUDA[i] == invalidEventCoord) ? eventsToTrace_paraCUDA[i] : a.eventsToTrace_paraCUDA[i];
                new_events_anti[i] = (a.eventsToTrace_antiCUDA[i] == invalidEventCoord) ? eventsToTrace_antiCUDA[i] : a.eventsToTrace_antiCUDA[i];
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


    static void loadConstantParameters(SetupParameters *setup);
    static void makeBin(int totalBinEventCount, BinParameters *bin, ReductionVars *reduce);

private:
    static const int warpSize = 32;

    inline static SetupParameters *setupCUDA;
    inline static BinParameters *binCUDA;
    inline static ReductionVars *reduceCUDA;

    //Declare the arrays for the reduction variable
    inline static float *hist_image_plate_source_CUDA;
    inline static float *hist_image_plate_crystal1_CUDA;
    inline static float *hist_image_plate_crystal2_para_CUDA;
    inline static float *hist_image_plate_crystal2_anti_CUDA;
    inline static float *hist_image_plate_detc_para_CUDA;
    inline static float *hist_image_plate_detc_anti_CUDA;
    inline static float *max_hist_CUDA;
    inline static float* eventsToTrace_paraCUDA;
    inline static float* eventsToTrace_antiCUDA;



    static void loadBinParameters(BinParameters *bin, ReductionVars *reduce);
    static void unloadBinParameters(BinParameters *bin, ReductionVars *reduce);
};

__global__ static void makeBinKernel(int totalBinEventCount, ParallelBin::BinParameters *bin, ParallelBin::SetupParameters *setup, ParallelBin::ReductionVars *reduce);


#endif /* PARALLEL_BIN_CUH_ */
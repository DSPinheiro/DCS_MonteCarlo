#pragma once

/*
 * Source_complex.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef SOURCE_COMPLEX_HH_
#define SOURCE_COMPLEX_HH_


#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <random>

#include "simulation_interface.h"

#include "simuGlobals.hh"
#include "obtain_time.hh"
#include "make_plot_profiles.hh"


class Source_complex{

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

        BinParameters() : 
        total_current_bins (0), sec_crystal_Antiparallel_reach (false), sec_crystal_Parallel_reach (false)
        {}

    };
    
    struct SetupParameters
    {
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
        double LT_aper_Db;
        double S_aper_sqr;
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
        double cos_tetartab;
        double sin_tetartab;
        double tilt_C2_para_temp;
        double tilt_C2_anti_temp;
        double cos_difteC1_Ta;
        double sin_difteC1_Ta;
        double dist_Cr1_Cr2_Db;
        double tetaref;
        double delrot;
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
        #ifdef OPENMP
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

        ReductionVars() :
        #ifdef OPENMP
        energy_sum_para_thread (0), energy_sum_anti_thread (0), 
        #endif
        toint_para (0), toint_anti (0), counts_sour (0), counts_C1 (0), counts_C2_para (0), counts_C2_anti (0), counts_detc_para (0), counts_detc_anti (0)
        {}
    };


    static bool run_Source(SimulationInterface* w);
    static void makeBin(SimulationInterface *w, \
                        SetupParameters *setup, BinParameters *bin, ReductionVars *reduce, \
                        std::vector<std::vector<double>> *eventsToTrace_para, std::vector<std::vector<double>> *eventsToTrace_anti);
};


#endif /* SOURCE_COMPLEX_HH_ */

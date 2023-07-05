/*
 * Source_complex.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#include "source_complex.hh"
#include "Util.h"

#ifdef OPENMP
#include <omp.h>
#endif

#ifdef CUDA
#include "parallel_bin.cuh"
#endif

using namespace std;



bool Source_complex::run_Source(SimulationInterface *w){

    #ifdef OPENMP
    omp_set_num_threads(ParallelSettingsInput.OMP_threads);
    #endif

    stringstream logString;
    
    //setup the variables that will be used during bin count MC simulation
    #ifdef CUDA
    ParallelBin::BinParameters *bin_CUDA = nullptr;
    BinParameters *bin = nullptr;
    if(ParallelSettingsInput.Make_GPU)
    {
        bin_CUDA = new ParallelBin::BinParameters();
    }
    else
    {
        bin = new BinParameters();
    }
    #else
    //setup the variables that will be used during bin count MC simulation
    BinParameters *bin = new BinParameters();
    #endif
    
    //setup the lattice constant for the temperature
    double d_lat1_para, d_lat2_para, d_lat1_anti, d_lat2_anti;

    d_lat1_para = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_1_para);
    d_lat1_anti = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_1_anti);
    d_lat2_para = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_para);
    d_lat2_anti = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_anti);
    
    //2 * first crystal lattice spacing (fixed throughout the simulation)
    double tw_d1_anti = 2 * d_lat1_anti;


    //setup the second crystal rotation limits
    double delrot_min, delrot_max, delrot_inc;

    delrot_min = Mini_angl * M_PI / 180;
    delrot_max = Maxi_angl * M_PI / 180;
    delrot_inc = (delrot_max - delrot_min) / (double)PlotParametersInput.nubins;


    //setup the variables that are constant throughout the simulation, or per bin.
    //some of them are initialized in the constructor, other are conditionaly initialized.
    
    #ifdef CUDA
    ParallelBin::SetupParameters *setup_CUDA = nullptr;
    SetupParameters *setup = nullptr;
    if(ParallelSettingsInput.Make_GPU)
    {
        setup_CUDA = new ParallelBin::SetupParameters(M_PI / 2 - (teta_crys1 * M_PI / 180), 2 * d_lat1_para, 2 * d_lat2_para, 2 * d_lat2_anti, \
                                                    GeolengthelementsInput.S_aper / 2, GeolengthelementsInput.S_aper_var / 2, pow(GeolengthelementsInput.S_aper / 2, 2), \
                                                    GeolengthelementsInput.S_sour / 2, GeolengthelementsInput.z_sour / 2, GeolengthelementsInput.y_sour / 2, \
                                                    GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                    -GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                    GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                    -GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                    delrot_max, NumberRaysInput.number_rotati * NumberRaysInput.nbeams * PlotParametersInput.nubins);
            
    }
    else
    {
        setup = new SetupParameters(M_PI / 2 - (teta_crys1 * M_PI / 180), 2 * d_lat1_para, 2 * d_lat2_para, 2 * d_lat2_anti, \
                                                GeolengthelementsInput.S_aper / 2, GeolengthelementsInput.S_aper_var / 2, pow(GeolengthelementsInput.S_aper / 2, 2), \
                                                GeolengthelementsInput.S_sour / 2, GeolengthelementsInput.z_sour / 2, GeolengthelementsInput.y_sour / 2, \
                                                GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                -GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                -GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                delrot_max, NumberRaysInput.number_rotati * NumberRaysInput.nbeams * PlotParametersInput.nubins);
    }
    #else
    SetupParameters *setup = new SetupParameters(M_PI / 2 - (teta_crys1 * M_PI / 180), 2 * d_lat1_para, 2 * d_lat2_para, 2 * d_lat2_anti, \
                                                GeolengthelementsInput.S_aper / 2, GeolengthelementsInput.S_aper_var / 2, pow(GeolengthelementsInput.S_aper / 2, 2), \
                                                GeolengthelementsInput.S_sour / 2, GeolengthelementsInput.z_sour / 2, GeolengthelementsInput.y_sour / 2, \
                                                GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                -GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                -GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver, \
                                                delrot_max, NumberRaysInput.number_rotati * NumberRaysInput.nbeams * PlotParametersInput.nubins);
    #endif
    

    //setup the variables that are going to be reduced in the MC loops to calculate the counts on the final spectrum
    #ifdef CUDA
    ParallelBin::ReductionVars *reduce_CUDA = nullptr;
    ReductionVars *reduce = nullptr;
    if(ParallelSettingsInput.Make_GPU)
    {
        reduce_CUDA = new ParallelBin::ReductionVars(NumberRaysInput.number_events);
    }
    else
    {
        reduce = new ReductionVars();
    }
    #else
    ReductionVars *reduce = new ReductionVars();
    #endif

    double counts_C2_para_t, counts_detc_para_t, counts_C2_anti_t, counts_detc_anti_t;
    


    double max_valu_para, max_valu_anti, tetabrag_ref;
    double cos_e, sin_e, cos_tilt_C1, sin_tilt_C1;
    double auxBragg;
    double teta_max_L, inc_tem;
    double angle_para, angle_anti, pha_temp[4], fi_max_L, tetartab;
    double dist_T_Cr1_Db, y_min_aper, y_max_aper, z_min_aper, z_max_aper;
    


    vector<int> int_time_out;
    int int_time_out_begg, int_time_mili_out_begg, total_para, total_anti, bin_tem = 1, bin_fas = 1;
    
    int max_para, I;
    
    
    int* toint_para_total = new int[PlotParametersInput.nubins];
    int* toint_anti_total = new int[PlotParametersInput.nubins];

    fill(toint_para_total, toint_para_total + PlotParametersInput.nubins, 0);
    fill(toint_anti_total, toint_anti_total + PlotParametersInput.nubins, 0);
    
    
    bool make_G_function, cond_rotation, poliP;

    max_valu_para = 0;
    max_valu_anti = 0;


    make_G_function = false;


    pha_temp[0] = ((double)rand() / RAND_MAX) * 2 * M_PI;
    pha_temp[1] = ((double)rand() / RAND_MAX) * 2 * M_PI;
    pha_temp[2] = ((double)rand() / RAND_MAX) * 2 * M_PI;
    pha_temp[3] = ((double)rand() / RAND_MAX) * 2 * M_PI;

    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        cos_e = cos(setup_CUDA->tetaref);
        sin_e = sin(setup_CUDA->tetaref);
    }
    else
    {
    #endif
    cos_e = cos(setup->tetaref);
    sin_e = sin(setup->tetaref);
    #ifdef CUDA
    }
    #endif

    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        if(CurveVerticalTiltInput.make_CurveTilt)
            setup_CUDA->tilt_C1_temp = Util::ObtainVert(1, 0);
        else
            setup_CUDA->tilt_C1_temp = GeoParametersInput.tilt_C1 * convrad;
    }
    else
    {
    #endif
    if(CurveVerticalTiltInput.make_CurveTilt)
        setup->tilt_C1_temp = Util::ObtainVert(1, 0);
    else
        setup->tilt_C1_temp = GeoParametersInput.tilt_C1 * convrad;
    #ifdef CUDA
    }
    #endif


    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        cos_tilt_C1 = cos(setup_CUDA->tilt_C1_temp);
        sin_tilt_C1 = sin(setup_CUDA->tilt_C1_temp);
    }
    else
    {
    #endif
    cos_tilt_C1 = cos(setup->tilt_C1_temp);
    sin_tilt_C1 = sin(setup->tilt_C1_temp);
    #ifdef CUDA
    }
    #endif


    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        bin_CUDA->n1x = -cos_tilt_C1 * sin_e;
        bin_CUDA->n1y = cos_tilt_C1 * cos_e;
    }
    else
    {
    #endif
    bin->n1x = -cos_tilt_C1 * sin_e;
    bin->n1y = cos_tilt_C1 * cos_e;
    #ifdef CUDA
    }
    #endif

    
    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        setup_CUDA->n1z = sin_tilt_C1;
    }
    else
    {
    #endif
    setup->n1z = sin_tilt_C1;
    #ifdef CUDA
    }
    #endif


    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        tetabrag_ref = asin(-bin_CUDA->n1x);
    }
    else
    {
    #endif
    tetabrag_ref = asin(-bin->n1x);
    #ifdef CUDA
    }
    #endif


    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        if(CurvedCrystalInput.Curve_crystall){
            setup_CUDA->n1x_temp = bin_CUDA->n1x;
            setup_CUDA->n1y_temp = bin_CUDA->n1y;
        }
    }
    else
    {
    #endif
    if(CurvedCrystalInput.Curve_crystall){
        setup->n1x_temp = bin->n1x;
        setup->n1y_temp = bin->n1y;
    }
    #ifdef CUDA
    }
    #endif

    if(GeoParapathlengthsInput.type_source == "UC"){
        teta_max_L = atan((GeolengthelementsInput.S_aper - GeolengthelementsInput.S_shi_hor_B + GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper);
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {    
            setup_CUDA->teta_min_L = - atan((GeolengthelementsInput.S_aper + GeolengthelementsInput.S_shi_hor_B - GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper);
        }
        else
        {
        #endif
        setup->teta_min_L = - atan((GeolengthelementsInput.S_aper + GeolengthelementsInput.S_shi_hor_B - GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper);
        #ifdef CUDA
        }
        #endif
    }else{
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {    
            teta_max_L = atan(min((GeolengthelementsInput.y_aper / 2 + setup_CUDA->y_sour_2 + GeolengthelementsInput.S_shi_hor_B - GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.y_first_crys / 2 + setup_CUDA->y_sour_2 - GeolengthelementsInput.S_shi_hor_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
            setup_CUDA->teta_min_L = - atan(min((GeolengthelementsInput.y_aper / 2 + setup_CUDA->y_sour_2 - GeolengthelementsInput.S_shi_hor_B + GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.y_first_crys / 2 + setup_CUDA->y_sour_2 + GeolengthelementsInput.S_shi_hor_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        }
        else
        {
        #endif
        teta_max_L = atan(min((GeolengthelementsInput.y_aper / 2 + setup->y_sour_2 + GeolengthelementsInput.S_shi_hor_B - GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.y_first_crys / 2 + setup->y_sour_2 - GeolengthelementsInput.S_shi_hor_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        setup->teta_min_L = - atan(min((GeolengthelementsInput.y_aper / 2 + setup->y_sour_2 - GeolengthelementsInput.S_shi_hor_B + GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.y_first_crys / 2 + setup->y_sour_2 + GeolengthelementsInput.S_shi_hor_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        #ifdef CUDA
        }
        #endif
    }


    if(FullEnergySpectrumInput.make_more_lines == 0){
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            auxBragg = asin(picks[1].lamda / setup_CUDA->tw_d1_para);
        }
        else
        {
        #endif
        auxBragg = asin(picks[1].lamda / setup->tw_d1_para);
        #ifdef CUDA
        }
        #endif
        
        if(GeoParametersInput.make_table_noise)
        {
            teta_max_L = min(M_PI / 2 + (GeoParametersInput.teta_table + GeoParametersInput.table_resolution) * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg + limitReflec, teta_max_L);
            
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                setup_CUDA->teta_min_L = max(M_PI / 2 + (GeoParametersInput.teta_table - GeoParametersInput.table_resolution) * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg - limitReflec, setup_CUDA->teta_min_L);
            }
            else
            {
            #endif
            setup->teta_min_L = max(M_PI / 2 + (GeoParametersInput.teta_table - GeoParametersInput.table_resolution) * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg - limitReflec, setup->teta_min_L);
            #ifdef CUDA
            }
            #endif
        }
        else
        {
            teta_max_L = min(M_PI / 2 + GeoParametersInput.teta_table * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg + limitReflec, teta_max_L);
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                setup_CUDA->teta_min_L = max(M_PI / 2 + GeoParametersInput.teta_table * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg - limitReflec, setup_CUDA->teta_min_L);
            }
            else
            {
            #endif
            setup->teta_min_L = max(M_PI / 2 + GeoParametersInput.teta_table * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg - limitReflec, setup->teta_min_L);
            #ifdef CUDA
            }
            #endif
        }
    }

    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        setup_CUDA->del_teta_L = teta_max_L - setup_CUDA->teta_min_L;
    }
    else
    {
    #endif
    setup->del_teta_L = teta_max_L - setup->teta_min_L;
    #ifdef CUDA
    }
    #endif

    if(GeoParapathlengthsInput.type_source == "UC"){
        fi_max_L = atan((GeolengthelementsInput.S_aper - GeolengthelementsInput.S_shi_ver_B + GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper);
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->fi_min_L = - atan((GeolengthelementsInput.S_aper + GeolengthelementsInput.S_shi_ver_B - GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper);
        }
        else
        {
        #endif
        setup->fi_min_L = - atan((GeolengthelementsInput.S_aper + GeolengthelementsInput.S_shi_ver_B - GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper);
        #ifdef CUDA
        }
        #endif
    }else{
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            fi_max_L = atan(min((GeolengthelementsInput.z_aper / 2 + setup_CUDA->z_sour_2 + GeolengthelementsInput.S_shi_ver_B - GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.z_first_crys / 2 + setup_CUDA->z_sour_2 - GeolengthelementsInput.S_shi_ver_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
            setup_CUDA->fi_min_L = - atan(min((GeolengthelementsInput.z_aper / 2 + setup_CUDA->z_sour_2 - GeolengthelementsInput.S_shi_ver_B + GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.z_first_crys / 2 + setup_CUDA->z_sour_2 + GeolengthelementsInput.S_shi_ver_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        }
        else
        {
        #endif
        fi_max_L = atan(min((GeolengthelementsInput.z_aper / 2 + setup->z_sour_2 + GeolengthelementsInput.S_shi_ver_B - GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.z_first_crys / 2 + setup->z_sour_2 - GeolengthelementsInput.S_shi_ver_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        setup->fi_min_L = - atan(min((GeolengthelementsInput.z_aper / 2 + setup->z_sour_2 - GeolengthelementsInput.S_shi_ver_B + GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.z_first_crys / 2 + setup->z_sour_2 + GeolengthelementsInput.S_shi_ver_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        #ifdef CUDA
        }
        #endif
    }

    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        setup_CUDA->del_fi_L = fi_max_L - setup_CUDA->fi_min_L;
    }
    else
    {
    #endif
    setup->del_fi_L = fi_max_L - setup->fi_min_L;
    #ifdef CUDA
    }
    #endif

    tetartab = GeoParametersInput.teta_table * convrad / 2;
    
    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        setup_CUDA->cos_tetartab = cos(GeoParametersInput.teta_table * convrad);
        setup_CUDA->sin_tetartab = sin(GeoParametersInput.teta_table * convrad);
        setup_CUDA->cos_difteC1_Ta = cos(GeoParametersInput.teta_table * convrad - setup_CUDA->tetaref);
        setup_CUDA->sin_difteC1_Ta = sin(GeoParametersInput.teta_table * convrad - setup_CUDA->tetaref);

        setup_CUDA->cos_tetartabdete_para = cos(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
        setup_CUDA->sin_tetartabdete_para = sin(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);

        setup_CUDA->cos_tetartabdete_anti = cos(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_anti * convrad);
        setup_CUDA->sin_tetartabdete_anti = sin(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_anti * convrad);
    }
    else
    {
    #endif
    setup->cos_tetartab = cos(GeoParametersInput.teta_table * convrad);
    setup->sin_tetartab = sin(GeoParametersInput.teta_table * convrad);
    setup->cos_difteC1_Ta = cos(GeoParametersInput.teta_table * convrad - setup->tetaref);
    setup->sin_difteC1_Ta = sin(GeoParametersInput.teta_table * convrad - setup->tetaref);

    setup->cos_tetartabdete_para = cos(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
    setup->sin_tetartabdete_para = sin(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);

    setup->cos_tetartabdete_anti = cos(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_anti * convrad);
    setup->sin_tetartabdete_anti = sin(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_anti * convrad);
    #ifdef CUDA
    }
    #endif

    max_hist[0] = 0;
    max_hist[1] = 0;


    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        if(tw_d1_anti < setup_CUDA->tw_d2_anti){
            setup_CUDA->a_lamds_uni = tw_d1_anti * sin(setup_CUDA->tetaref + delrot_min - 0.2);
            setup_CUDA->b_lamds_uni = setup_CUDA->tw_d2_anti * sin(setup_CUDA->tetaref + delrot_max + 0.2);
        }else{
            setup_CUDA->a_lamds_uni = setup_CUDA->tw_d2_anti * sin(setup_CUDA->tetaref + delrot_min - 0.2);
            setup_CUDA->b_lamds_uni = tw_d1_anti * sin(setup_CUDA->tetaref + delrot_max + 0.2);
        }

        setup_CUDA->b_lamds_uni = setup_CUDA->b_lamds_uni - setup_CUDA->a_lamds_uni;
    }
    else
    {
    #endif
    if(tw_d1_anti < setup->tw_d2_anti){
        setup->a_lamds_uni = tw_d1_anti * sin(setup->tetaref + delrot_min - 0.2);
        setup->b_lamds_uni = setup->tw_d2_anti * sin(setup->tetaref + delrot_max + 0.2);
    }else{
        setup->a_lamds_uni = setup->tw_d2_anti * sin(setup->tetaref + delrot_min - 0.2);
        setup->b_lamds_uni = tw_d1_anti * sin(setup->tetaref + delrot_max + 0.2);
    }

    setup->b_lamds_uni = setup->b_lamds_uni - setup->a_lamds_uni;
    #ifdef CUDA
    }
    #endif

    counts_C2_para_t = 0;
    counts_detc_para_t = 0;
    counts_C2_anti_t = 0;
    counts_detc_anti_t = 0;

    min_plot[0][0] = Mini_angl + teta_crys1;
    min_plot[1][0] = Mini_angl - teta_crys1;
    max_plot[0][0] = Maxi_angl + teta_crys1;
    max_plot[1][0] = Maxi_angl - teta_crys1;


    dist_T_Cr1_Db = (double)GeoParapathlengthsInput.dist_T_Cr1;
    
    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        setup_CUDA->LT_aper_Db = (double)GeoParapathlengthsInput.LT_aper;
        setup_CUDA->dist_Cr1_Cr2_Db = (double)GeoParapathlengthsInput.dist_Cr1_Cr2;
        setup_CUDA->dist_Cr2_det_Db = (double)GeoParapathlengthsInput.dist_Cr2_Det;
    }
    else
    {
    #endif
    setup->LT_aper_Db = (double)GeoParapathlengthsInput.LT_aper;
    setup->dist_Cr1_Cr2_Db = (double)GeoParapathlengthsInput.dist_Cr1_Cr2;
    setup->dist_Cr2_det_Db = (double)GeoParapathlengthsInput.dist_Cr2_Det;
    #ifdef CUDA
    }
    #endif

    y_min_aper = GeolengthelementsInput.S_shi_hor_A - GeolengthelementsInput.y_aper / 2;
    y_max_aper = GeolengthelementsInput.S_shi_hor_A + GeolengthelementsInput.y_aper / 2;
    z_min_aper = GeolengthelementsInput.S_shi_ver_A - GeolengthelementsInput.z_aper / 2;
    z_max_aper = GeolengthelementsInput.S_shi_ver_A + GeolengthelementsInput.z_aper / 2;

    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        setup_CUDA->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2;
        setup_CUDA->y_max_C1 = GeolengthelementsInput.y_first_crys / 2;
        setup_CUDA->y_min_C2 = setup_CUDA->y_min_C1;
        setup_CUDA->y_max_C2 = setup_CUDA->y_max_C1;
    }
    else
    {
    #endif
    setup->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2;
    setup->y_max_C1 = GeolengthelementsInput.y_first_crys / 2;
    setup->y_min_C2 = setup->y_min_C1;
    setup->y_max_C2 = setup->y_max_C1;
    #ifdef CUDA
    }
    #endif

    if(UserSettingsInput.center_Mask){
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->z_max_C1 = 0.6;
            setup_CUDA->z_min_C1 = - 0.6;

            setup_CUDA->y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
            setup_CUDA->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        }
        else
        {
        #endif
        setup->z_max_C1 = 0.6;
        setup->z_min_C1 = - 0.6;

        setup->y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
        setup->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        #ifdef CUDA
        }
        #endif
    }else{
        if(UserSettingsInput.mask_C1 == 0){
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                setup_CUDA->z_max_C1 = GeolengthelementsInput.z_first_crys / 2;
                setup_CUDA->z_min_C1 = -GeolengthelementsInput.z_first_crys / 2;
            }
            else
            {
            #endif
            setup->z_max_C1 = GeolengthelementsInput.z_first_crys / 2;
            setup->z_min_C1 = -GeolengthelementsInput.z_first_crys / 2;
            #ifdef CUDA
            }
            #endif
        }else if(UserSettingsInput.mask_C1 == 1){
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                setup_CUDA->z_max_C1 = GeolengthelementsInput.z_first_crys / 2 - 0.2;
                setup_CUDA->z_min_C1 = 0;
                setup_CUDA->y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
                setup_CUDA->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
            }
            else
            {
            #endif
            setup->z_max_C1 = GeolengthelementsInput.z_first_crys / 2 - 0.2;
            setup->z_min_C1 = 0;
            setup->y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
            setup->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
            #ifdef CUDA
            }
            #endif
        }else if(UserSettingsInput.mask_C1 == 2){
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                setup_CUDA->z_max_C1 = 0;
                setup_CUDA->z_min_C1 = -GeolengthelementsInput.z_first_crys / 2 + 0.2;
                setup_CUDA->y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
                setup_CUDA->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
            }
            else
            {
            #endif
            setup->z_max_C1 = 0;
            setup->z_min_C1 = -GeolengthelementsInput.z_first_crys / 2 + 0.2;
            setup->y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
            setup->y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
            #ifdef CUDA
            }
            #endif
        }
    }


    if(UserSettingsInput.mask_C2 == 0){
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->z_max_C2 = GeolengthelementsInput.z_first_crys / 2;
            setup_CUDA->z_min_C2 = -GeolengthelementsInput.z_first_crys / 2;
        }
        else
        {
        #endif
        setup->z_max_C2 = GeolengthelementsInput.z_first_crys / 2;
        setup->z_min_C2 = -GeolengthelementsInput.z_first_crys / 2;
        #ifdef CUDA
        }
        #endif
    }else if(UserSettingsInput.mask_C2 == 1){
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->z_max_C2 = GeolengthelementsInput.z_first_crys / 2 - 0.2;
            setup_CUDA->z_min_C2 = 0;
            setup_CUDA->y_max_C2 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
            setup_CUDA->y_min_C2 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        }
        else
        {
        #endif
        setup->z_max_C2 = GeolengthelementsInput.z_first_crys / 2 - 0.2;
        setup->z_min_C2 = 0;
        setup->y_max_C2 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
        setup->y_min_C2 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        #ifdef CUDA
        }
        #endif
    }else if(UserSettingsInput.mask_C2 == 2){
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->z_max_C2 = 0;
            setup_CUDA->z_min_C2 = -GeolengthelementsInput.z_first_crys / 2 + 0.2;
            setup_CUDA->y_max_C2 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
            setup_CUDA->y_min_C2 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        }
        else
        {
        #endif
        setup->z_max_C2 = 0;
        setup->z_min_C2 = -GeolengthelementsInput.z_first_crys / 2 + 0.2;
        setup->y_max_C2 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
        setup->y_min_C2 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        #ifdef CUDA
        }
        #endif
    }


    if(GraphOptionsInput.make_image_plates){
        Util::initPlates();
    }

    int_time_out.push_back(0);
    int_time_out.push_back(0);
    int_time_out = Obtain_time::simuTime(0, 0, int_time_out[0], int_time_out[1], w);
    int_time_out_begg = int_time_out[0];
    int_time_mili_out_begg = int_time_out[1];

    
    #if ! defined(OPENMP) && ! defined(CUDA)
    reduce->energy_sum_para = new vector<double>();
    reduce->energy_sum_anti = new vector<double>();
    
    reduce->energy_sum_para->resize(PlotParametersInput.nubins);
    reduce->energy_sum_anti->resize(PlotParametersInput.nubins);
    #else
    vector<double> energy_sum_para;
    vector<double> energy_sum_anti;

    energy_sum_para.resize(PlotParametersInput.nubins);
    energy_sum_anti.resize(PlotParametersInput.nubins);
    #endif
    
    #ifdef CUDA
    if(ParallelSettingsInput.Make_GPU)
    {
        bin_CUDA->teta_table_thread = GeoParametersInput.teta_table;
    
        #ifdef QT_EXISTS
        emit w->setTetaTableSignal(bin_CUDA->teta_table_thread);
        #endif
        
    }
    else
    {
    #endif
    bin->teta_table_thread = GeoParametersInput.teta_table;
    
    #ifdef QT_EXISTS
    emit w->setTetaTableSignal(bin->teta_table_thread);
    #endif
    
    #ifdef CUDA
    }
    #endif


    //wierd hack but seems to work
    while(
        #ifdef CUDA
        ((ParallelSettingsInput.Make_GPU) ? setup_CUDA->numbins : setup->numbins)
        #else
        setup->numbins
        #endif
        < PlotParametersInput.nubins)
    {
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->numbins++;
        }
        else
        {
        #endif
        setup->numbins++;
        #ifdef CUDA
        }
        #endif

        if(TemperatureParametersInput.mk_temp_bin){
            inc_tem = Util::getNewTemp(
                bin_tem,
                bin_fas,
                pha_temp[2]);

            logString << "inc_te: " << inc_tem << endl;

            d_lat2_para = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_para + inc_tem);

            inc_tem = Util::getNewTemp(
                bin_tem,
                bin_fas,
                pha_temp[3]);

            d_lat2_anti = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_anti + inc_tem);

            bin_tem++;

            tw_d1_anti = 2 * d_lat1_anti;
            
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                setup_CUDA->tw_d1_para = 2 * d_lat1_para;
                setup_CUDA->tw_d2_para = 2 * d_lat2_para;
                setup_CUDA->tw_d2_anti = 2 * d_lat2_anti;
            }
            else
            {
            #endif
            setup->tw_d1_para = 2 * d_lat1_para;
            setup->tw_d2_para = 2 * d_lat2_para;
            setup->tw_d2_anti = 2 * d_lat2_anti;
            #ifdef CUDA
            }
            #endif


            logString << "tw_1: " << d_lat1_para << endl;
            logString << "tw_2: " << d_lat2_para << endl;
            
            #ifdef QT_EXISTS
                emit w->LogLineSignal(logString.str());
            #else
                cout << logString.str();
            #endif

        }

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            angle_para = setup_CUDA->delrot * 180 / M_PI + teta_crys1;
            angle_anti = setup_CUDA->delrot * 180 / M_PI - teta_crys1;
        }
        else
        {
        #endif
        angle_para = setup->delrot * 180 / M_PI + teta_crys1;
        angle_anti = setup->delrot * 180 / M_PI - teta_crys1;
        #ifdef CUDA
        }
        #endif

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            if(CurveVerticalTiltInput.make_CurveTilt){
                setup_CUDA->tilt_C2_para_temp = Util::ObtainVert(2, angle_para);
                setup_CUDA->tilt_C2_anti_temp = Util::ObtainVert(2, angle_anti);
            }else{
                setup_CUDA->tilt_C2_para_temp = GeoParametersInput.tilt_C2 * convrad;
                setup_CUDA->tilt_C2_anti_temp = GeoParametersInput.tilt_C2 * convrad;
            }
        }
        else
        {
        #endif
        if(CurveVerticalTiltInput.make_CurveTilt){
            setup->tilt_C2_para_temp = Util::ObtainVert(2, angle_para);
            setup->tilt_C2_anti_temp = Util::ObtainVert(2, angle_anti);
        }else{
            setup->tilt_C2_para_temp = GeoParametersInput.tilt_C2 * convrad;
            setup->tilt_C2_anti_temp = GeoParametersInput.tilt_C2 * convrad;
        }
        #ifdef CUDA
        }
        #endif


        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            reduce_CUDA->counts_C2_para = 0;
            reduce_CUDA->counts_C2_anti = 0;
        }
        else
        {
        #endif
        reduce->counts_C2_para = 0;
        reduce->counts_C2_anti = 0;
        #ifdef CUDA
        }
        #endif

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            bin_CUDA->n2x_para = cos(setup_CUDA->tilt_C2_para_temp) * sin(GeoParametersInput.teta_table * convrad - setup_CUDA->tetaref + setup_CUDA->delrot);
            bin_CUDA->n2y_para = - cos(setup_CUDA->tilt_C2_para_temp) * cos(GeoParametersInput.teta_table * convrad - setup_CUDA->tetaref + setup_CUDA->delrot);
            setup_CUDA->n2z_para = sin(setup_CUDA->tilt_C2_para_temp);
            bin_CUDA->n2x_anti = - cos(setup_CUDA->tilt_C2_anti_temp) * sin(GeoParametersInput.teta_table * convrad + setup_CUDA->tetaref + setup_CUDA->delrot);
            bin_CUDA->n2y_anti = cos(setup_CUDA->tilt_C2_anti_temp) * cos(GeoParametersInput.teta_table * convrad + setup_CUDA->tetaref + setup_CUDA->delrot);
            setup_CUDA->n2z_anti = sin(setup_CUDA->tilt_C2_anti_temp);
        }
        else
        {
        #endif
        bin->n2x_para = cos(setup->tilt_C2_para_temp) * sin(GeoParametersInput.teta_table * convrad - setup->tetaref + setup->delrot);
        bin->n2y_para = - cos(setup->tilt_C2_para_temp) * cos(GeoParametersInput.teta_table * convrad - setup->tetaref + setup->delrot);
        setup->n2z_para = sin(setup->tilt_C2_para_temp);
        bin->n2x_anti = - cos(setup->tilt_C2_anti_temp) * sin(GeoParametersInput.teta_table * convrad + setup->tetaref + setup->delrot);
        bin->n2y_anti = cos(setup->tilt_C2_anti_temp) * cos(GeoParametersInput.teta_table * convrad + setup->tetaref + setup->delrot);
        setup->n2z_anti = sin(setup->tilt_C2_anti_temp);
        #ifdef CUDA
        }
        #endif

        if(CurvedCrystalInput.Curve_crystall){
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                setup_CUDA->n2x_para_temp = bin_CUDA->n2x_para;
                setup_CUDA->n2y_para_temp = bin_CUDA->n2y_para;
                setup_CUDA->n2x_anti_temp = bin_CUDA->n2x_anti;
                setup_CUDA->n2y_anti_temp = bin_CUDA->n2y_anti;
            }
            else
            {
            #endif
            setup->n2x_para_temp = bin->n2x_para;
            setup->n2y_para_temp = bin->n2y_para;
            setup->n2x_anti_temp = bin->n2x_anti;
            setup->n2y_anti_temp = bin->n2y_anti;
            #ifdef CUDA
            }
            #endif
        }

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->cos_difteC2_det_para = cos(setup_CUDA->tetaref - setup_CUDA->delrot + GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
            setup_CUDA->sin_difteC2_det_para = sin(setup_CUDA->tetaref - setup_CUDA->delrot + GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
        }
        else
        {
        #endif
        setup->cos_difteC2_det_para = cos(setup->tetaref - setup->delrot + GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
        setup->sin_difteC2_det_para = sin(setup->tetaref - setup->delrot + GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
        #ifdef CUDA
        }
        #endif

        I = 1;


        vector<vector<double>> eventsToTrace_para;
        vector<vector<double>> eventsToTrace_anti;

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            reduce_CUDA->energy_sum_para_thread = 0;
            reduce_CUDA->energy_sum_anti_thread = 0;
        }
        #endif

        #ifdef OPENMP
            reduce->energy_sum_para_thread = 0;
            reduce->energy_sum_anti_thread = 0;
        #endif

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            ParallelBin::makeBin(NumberRaysInput.nbeams * NumberRaysInput.number_rotati, NumberRaysInput.number_events,
                                bin_CUDA, setup_CUDA, reduce_CUDA);
        }
        else
        {
        #endif
        makeBin(w, setup, bin, reduce, &eventsToTrace_para, &eventsToTrace_anti);
        #ifdef CUDA
        }
        #endif
        
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            eventsToTrace_para.resize(bin_CUDA->curr3DEventCount_para);
            eventsToTrace_anti.resize(bin_CUDA->curr3DEventCount_anti);

            for(int i = 0; i < bin_CUDA->curr3DEventCount_para; i++)
            {
                std::vector<double> tmpEvent(12);
                tmpEvent.insert(tmpEvent.end(), &reduce_CUDA->eventsToTrace_paraCUDA[i][0], &reduce_CUDA->eventsToTrace_paraCUDA[i][12]);
                eventsToTrace_para.push_back(tmpEvent);
            }
            
            for(int i = 0; i < bin_CUDA->curr3DEventCount_anti; i++)
            {
                std::vector<double> tmpEvent(12);
                tmpEvent.insert(tmpEvent.end(), &reduce_CUDA->eventsToTrace_antiCUDA[i][0], &reduce_CUDA->eventsToTrace_antiCUDA[i][12]);
                eventsToTrace_anti.push_back(tmpEvent);
            }
        }
        #endif

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            energy_sum_para[setup_CUDA->numbins - 1] = reduce_CUDA->energy_sum_para_thread;
            energy_sum_anti[setup_CUDA->numbins - 1] = reduce_CUDA->energy_sum_anti_thread;    
        }
        else
        {
        #endif
        
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            energy_sum_para[setup->numbins - 1] = reduce_CUDA->energy_sum_para_thread;
            energy_sum_anti[setup->numbins - 1] = reduce_CUDA->energy_sum_anti_thread;
        }
        #endif

        #ifdef OPENMP
        energy_sum_para[setup->numbins - 1] = reduce->energy_sum_para_thread;
        energy_sum_anti[setup->numbins - 1] = reduce->energy_sum_anti_thread;
        #endif

        #ifdef CUDA
        }
        #endif

        #ifdef QT_EXISTS
            if (!w->isOpen())
            {
                gener_out << "********************************" << endl;
                gener_out << endl;
                gener_out << " SIMULATION TERMINATED BEFORE FINISH!!!" << endl;
                gener_out << endl;
                gener_out << " RESULTS ARE GARBAGE!!!" << endl;
                gener_out << endl;
                gener_out << "********************************" << endl;
                gener_out << endl;

                w->setPctDone(1.0f);

                return false;
            }


            if(GraphOptionsInput.make_image_plates){
            emit w->changeStatsSignal(
                SimulationInterface::Stats
                {
                    #ifdef CUDA
                    (ParallelSettingsInput.Make_GPU) ? reduce_CUDA->counts_sour : reduce->counts_sour ,
                    (ParallelSettingsInput.Make_GPU) ? reduce_CUDA->counts_C1 : reduce->counts_C1 ,
                    (ParallelSettingsInput.Make_GPU) ? reduce_CUDA->counts_C2_para : reduce->counts_C2_para ,
                    (ParallelSettingsInput.Make_GPU) ? reduce_CUDA->counts_C2_anti : reduce->counts_C2_anti ,
                    (ParallelSettingsInput.Make_GPU) ? reduce_CUDA->counts_detc_para : reduce->counts_detc_para ,
                    (ParallelSettingsInput.Make_GPU) ? reduce_CUDA->counts_detc_anti : reduce->counts_detc_anti ,
                    (ParallelSettingsInput.Make_GPU) ? setup_CUDA->delrot : setup->delrot,
                    #else
                    reduce->counts_sour,
                    reduce->counts_C1,
                    reduce->counts_C2_para,
                    reduce->counts_C2_anti,
                    reduce->counts_detc_para,
                    reduce->counts_detc_anti,
                    setup->delrot,
                    #endif
                    eventsToTrace_para,
                    eventsToTrace_anti
                }
            );
            }
        #endif

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            toint_para_total[setup_CUDA->numbins - 1] += reduce_CUDA->toint_para;
            total_para = toint_para_total[setup_CUDA->numbins - 1];

            toint_anti_total[setup_CUDA->numbins - 1] += reduce_CUDA->toint_anti;
            total_anti = toint_anti_total[setup_CUDA->numbins - 1];


            angle_para = setup_CUDA->delrot * 180 / M_PI + teta_crys1;
            angle_anti = setup_CUDA->delrot * 180 / M_PI - teta_crys1;
        }
        else
        {
        #endif
        toint_para_total[setup->numbins - 1] += reduce->toint_para;
        total_para = toint_para_total[setup->numbins - 1];

        toint_anti_total[setup->numbins - 1] += reduce->toint_anti;
        total_anti = toint_anti_total[setup->numbins - 1];


        angle_para = setup->delrot * 180 / M_PI + teta_crys1;
        angle_anti = setup->delrot * 180 / M_PI - teta_crys1;
        #ifdef CUDA
        }
        #endif


        if(export_prof){
            #ifdef CUDA
            if(ParallelSettingsInput.Make_GPU)
            {
                hist_para << - angle_para << "\t" << reduce_CUDA->toint_para << "\t" << sqrt((double)reduce_CUDA->toint_para) << endl;
                hist_anti << - angle_anti << "\t" << reduce_CUDA->toint_anti << "\t" << sqrt((double)reduce_CUDA->toint_anti) << endl;
            }
            else
            {
            #endif
                hist_para << - angle_para << "\t" << reduce->toint_para << "\t" << sqrt((double)reduce->toint_para) << endl;
                hist_anti << - angle_anti << "\t" << reduce->toint_anti << "\t" << sqrt((double)reduce->toint_anti) << endl;
            #ifdef CUDA
            }
            #endif
        }


        #ifdef QT_EXISTS
            if(GraphOptionsInput.make_graph_profile){
                #ifdef CUDA
                if(ParallelSettingsInput.Make_GPU)
                {
                    Make_plot_profiles::plotProfiles(
                        energy_sum_para[setup_CUDA->numbins - 1] / total_para,
                        angle_para,
                        total_para,
                        energy_sum_anti[setup_CUDA->numbins - 1] / total_anti,
                        angle_anti,
                        total_anti,
                        setup_CUDA->numbins,
                        reduce_CUDA->counts_sour,
                        reduce_CUDA->counts_C1,
                        reduce_CUDA->counts_C2_para,
                        reduce_CUDA->counts_C2_anti,
                        reduce_CUDA->counts_detc_para,
                        reduce_CUDA->counts_detc_anti,
                        w);
                }
                else
                {
                #endif
                Make_plot_profiles::plotProfiles(
                    #ifndef OPENMP
                    reduce->energy_sum_para->at(setup->numbins - 1) / total_para,
                    #else
                    energy_sum_para[setup->numbins - 1] / total_para,
                    #endif
                    angle_para,
                    total_para,
                    #ifndef OPENMP
                    reduce->energy_sum_anti->at(setup->numbins - 1) / total_anti,
                    #else
                    energy_sum_anti[setup->numbins - 1] / total_anti,
                    #endif
                    angle_anti,
                    total_anti,
                    setup->numbins,
                    reduce->counts_sour,
                    reduce->counts_C1,
                    reduce->counts_C2_para,
                    reduce->counts_C2_anti,
                    reduce->counts_detc_para,
                    reduce->counts_detc_anti,
                    w);
                #ifdef CUDA
                }
                #endif
            }
        #endif


        if(
            #ifdef CUDA
            (ParallelSettingsInput.Make_GPU) ? setup_CUDA->numbins : setup->numbins
            #else
            setup->numbins
            #endif
            == PlotParametersInput.nubins
        )
        {
            if(UserSettingsInput.fitting){
                Util::FitData(
                    #ifdef CUDA
                    (ParallelSettingsInput.Make_GPU) ? setup_CUDA->numbins : setup->numbins ,
                    #else
                    setup->numbins,
                    #endif
                    angle_para,
                    total_para,
                    angle_anti,
                    total_anti);

                if(max_valu_para < total_para){
                    max_valu_para = total_para;
                    peak_posi_para = angle_para;
                    amplitu_con_para = (double)total_para;
                }


                if(max_valu_anti < total_anti){
                    max_valu_anti = total_anti;
                    peak_posi_anti = angle_anti;
                    amplitu_con_anti = (double)total_anti;
                }
            }
        }


        if (
            #ifdef CUDA
            (ParallelSettingsInput.Make_GPU) ? setup_CUDA->numbins : setup->numbins
            #else
            setup->numbins
            #endif
            % 5 == 0
        )
        {
            int_time_out = Obtain_time::simuTime(
                1,
                #ifdef CUDA
                (ParallelSettingsInput.Make_GPU) ? (int)((PlotParametersInput.nubins - setup_CUDA->numbins) / 5) : (int)((PlotParametersInput.nubins - setup->numbins) / 5) ,
                #else
                (int)((PlotParametersInput.nubins - setup->numbins) / 5),
                #endif    
                int_time_out[0],
                int_time_out[1],
                w);
        }

        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            counts_C2_para_t += reduce_CUDA->counts_C2_para;
            counts_detc_para_t += reduce_CUDA->counts_detc_para;
            counts_C2_anti_t += reduce_CUDA->counts_C2_anti;
            counts_detc_anti_t += reduce_CUDA->counts_detc_anti;

            reduce_CUDA->counts_C2_para = 0;
            reduce_CUDA->counts_detc_para = 0;
            reduce_CUDA->counts_detc_anti = 0;
            reduce_CUDA->counts_C2_anti = 0;

            reduce_CUDA->toint_para = 0;
            reduce_CUDA->toint_anti = 0;
        }
        else
        {
        #endif
        counts_C2_para_t += reduce->counts_C2_para;
        counts_detc_para_t += reduce->counts_detc_para;
        counts_C2_anti_t += reduce->counts_C2_anti;
        counts_detc_anti_t += reduce->counts_detc_anti;

        reduce->counts_C2_para = 0;
        reduce->counts_detc_para = 0;
        reduce->counts_detc_anti = 0;
        reduce->counts_C2_anti = 0;

        reduce->toint_para = 0;
        reduce->toint_anti = 0;
        #ifdef CUDA
        }
        #endif

        
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            fill(&reduce_CUDA->hist_image_plate_crystal2_para[0][0], &reduce_CUDA->hist_image_plate_crystal2_para[0][0] + n_his_ima * n_his_ima, 0);
            fill(&reduce_CUDA->hist_image_plate_crystal2_anti[0][0], &reduce_CUDA->hist_image_plate_crystal2_anti[0][0] + n_his_ima * n_his_ima, 0);
            fill(&reduce_CUDA->hist_image_plate_detc_para[0][0], &reduce_CUDA->hist_image_plate_detc_para[0][0] + n_his_ima * n_his_ima, 0);
            fill(&reduce_CUDA->hist_image_plate_detc_anti[0][0], &reduce_CUDA->hist_image_plate_detc_anti[0][0] + n_his_ima * n_his_ima, 0);

            reduce_CUDA->max_hist[2] = 0;
            reduce_CUDA->max_hist[3] = 0;
            reduce_CUDA->max_hist[4] = 0;
            reduce_CUDA->max_hist[5] = 0;    
        }
        else
        {
        #endif
        fill(&hist_image_plate_crystal2_para[0][0], &hist_image_plate_crystal2_para[0][0] + n_his_ima * n_his_ima, 0);
        fill(&hist_image_plate_crystal2_anti[0][0], &hist_image_plate_crystal2_anti[0][0] + n_his_ima * n_his_ima, 0);
        fill(&hist_image_plate_detc_para[0][0], &hist_image_plate_detc_para[0][0] + n_his_ima * n_his_ima, 0);
        fill(&hist_image_plate_detc_anti[0][0], &hist_image_plate_detc_anti[0][0] + n_his_ima * n_his_ima, 0);
        
        max_hist[2] = 0;
        max_hist[3] = 0;
        max_hist[4] = 0;
        max_hist[5] = 0;    
        #ifdef CUDA
        }
        #endif
        
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            setup_CUDA->delrot = setup_CUDA->delrot - delrot_inc;
        }
        else
        {
        #endif
        setup->delrot = setup->delrot - delrot_inc;
        #ifdef CUDA
        }
        #endif
    }


    if(export_prof){
        gener_out << "********************************" << endl;
        gener_out << endl;
        gener_out << " Number of counts in the several geometric elements" << endl;
        gener_out << endl;
        
        #ifdef CUDA
        if(ParallelSettingsInput.Make_GPU)
        {
            gener_out << legen_counts_1 << "\t" << reduce_CUDA->counts_sour << endl;
            gener_out << legen_counts_2 << "\t" << reduce_CUDA->counts_C1 << endl;
        }
        else
        {
        #endif
        gener_out << legen_counts_1 << "\t" << reduce->counts_sour << endl;
        gener_out << legen_counts_2 << "\t" << reduce->counts_C1 << endl;
        #ifdef CUDA
        }
        #endif

        gener_out << legen_counts_3 << "\t" << counts_C2_para_t << endl;
        gener_out << legen_counts_4 << "\t" << counts_detc_para_t << endl;
        gener_out << legen_counts_5 << "\t" << counts_C2_anti_t << endl;
        gener_out << legen_counts_6 << "\t" << counts_detc_anti_t << endl;
    }

    return true;

}


void Source_complex::makeBin(SimulationInterface *w, SetupParameters *setup, BinParameters *bin, ReductionVars *reduce, \
                            vector<vector<double>> *eventsToTrace_para, vector<vector<double>> *eventsToTrace_anti)
{
    int I = 1;

    #ifdef OPENMP
    double energy_sum_para_thread = reduce->energy_sum_para_thread; double energy_sum_anti_thread = reduce->energy_sum_anti_thread;
    #endif
    int toint_para = reduce->toint_para; int toint_anti = reduce->toint_anti; int counts_sour = reduce->counts_sour; int counts_C1 = reduce->counts_C1;
    int counts_C2_para = reduce->counts_C2_para; int counts_C2_anti = reduce->counts_C2_anti; int counts_detc_para = reduce->counts_detc_para;
    int counts_detc_anti = reduce->counts_detc_anti;


    unsigned int seed = bin->seed; double teta_table_thread = bin->teta_table_thread; double p = bin->p; double tetadir = bin->tetadir;
    double sin_tetadir = bin->sin_tetadir; double cos_tetadir = bin->cos_tetadir; double tan_tetadir = bin->tan_tetadir;
    double cos_tetadirCry1 = bin->cos_tetadirCry1; double fidir = bin->fidir; double sin_fidir = bin->sin_fidir; double cos_fidir = bin->cos_fidir;
    double tan_fidir = bin->tan_fidir; double cos_fidirtilt = bin->cos_fidirtilt; double z = bin->z; double y = bin->y; double r = bin->r;
    double tetap = bin->tetap; double sin_tetap = bin->sin_tetap; double cos_tetap = bin->cos_tetap; double var_temp = bin->var_temp; double y_pro_C1 = bin->y_pro_C1;
    double z_pro_C1 = bin->z_pro_C1; double rx = bin->rx; double ry = bin->ry; double rz = bin->rz; double Costeta_CHC = bin->Costeta_CHC;
    double Sinteta_CHC = bin->Sinteta_CHC; double n1x = bin->n1x; double n1y = bin->n1y; double angle = bin->angle; double r2x = bin->r2x; double r2y = bin->r2y;
    double r2z = bin->r2z; double lamda = bin->lamda; double tetabra1 = bin->tetabra1; bool first_crystal_reach = bin->first_crystal_reach; double rx_rot = bin->rx_rot;
    double ry_rot = bin->ry_rot; double cos_fidirtilt2_para = bin->cos_fidirtilt2_para; double cos_fidirtilt2_anti = bin->cos_fidirtilt2_anti; double corr_dis = bin->corr_dis;
    double sin_tetatab_del_dir = bin->sin_tetatab_del_dir; double tetabra2 = bin->tetabra2; double n2x_para = bin->n2x_para; double n2y_para = bin->n2y_para;
    double r3x = bin->r3x; double r3y = bin->r3y; double r3z = bin->r3z; bool sec_crystal_Parallel_reach = bin->sec_crystal_Parallel_reach;
    double rx_rot_sec = bin->rx_rot_sec; double ry_rot_sec = bin->ry_rot_sec; double tetadir_det = bin->tetadir_det; double tan_tetadir_det = bin->tan_tetadir_det;
    double cos_tetadir_det = bin->cos_tetadir_det; double fidir_det = bin->fidir_det; double tan_fidir_det = bin->tan_fidir_det; double cos_fidir_det = bin->cos_fidir_det;
    double corr_dis_d_pa = bin->corr_dis_d_pa; double y_pro_C1_d_pa = bin->y_pro_C1_d_pa; double cos_tetap_det = bin->cos_tetap_det; double sin_tetap_det = bin->sin_tetap_det;
    double r_det = bin->r_det; double y_det = bin->y_det; double z_det = bin->z_det; double n2x_anti = bin->n2x_anti; double n2y_anti = bin->n2y_anti;
    bool sec_crystal_Antiparallel_reach = bin->sec_crystal_Antiparallel_reach; int total_current_bins = bin->total_current_bins;

    #ifdef OPENMP
    #pragma omp parallel firstprivate(seed, teta_table_thread, p, tetadir, sin_tetadir, cos_tetadir, tan_tetadir, cos_tetadirCry1, fidir, sin_fidir, cos_fidir, \
                                    tan_fidir, cos_fidirtilt, z, y, r, tetap, sin_tetap, cos_tetap, var_temp, y_pro_C1, z_pro_C1, rx, ry, rz, Costeta_CHC, Sinteta_CHC, \
                                    n1x, n1y, angle, r2x, r2y, r2z, lamda, tetabra1, first_crystal_reach, rx_rot, ry_rot, cos_fidirtilt2_para, cos_fidirtilt2_anti, \
                                    corr_dis, sin_tetatab_del_dir, tetabra2, n2x_para, n2y_para, r3x, r3y, r3z, sec_crystal_Parallel_reach, rx_rot_sec, ry_rot_sec, \
                                    tetadir_det, tan_tetadir_det, cos_tetadir_det, fidir_det, tan_fidir_det, cos_fidir_det, corr_dis_d_pa, y_pro_C1_d_pa, cos_tetap_det, \
                                    sin_tetap_det, r_det, y_det, z_det, n2x_anti, n2y_anti, sec_crystal_Antiparallel_reach) \
                            reduction(+:energy_sum_para_thread, energy_sum_anti_thread, toint_para, toint_anti, counts_sour, counts_C1, counts_C2_para, \
                                        counts_C2_anti, counts_detc_para, counts_detc_anti)
    #endif
    while(I <= NumberRaysInput.nbeams * NumberRaysInput.number_rotati){
        std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        #ifdef OPENMP
        // random seed for this event if we have openMP enabled
        seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() + 17 * omp_get_thread_num();
        #else
        seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        #endif

        static thread_local std::mt19937 generator;
        uniform_int_distribution<uint32_t> uniform(0, RAND_MAX);
        
        // Density function -> https://cplusplus.com/reference/random/normal_distribution/
        normal_distribution<double> table_noise(0, (GeoParametersInput.table_resolution / 2.355) / 3.0);


        generator.seed(seed);
        
        //Temporary event to show in the 3D view
        //If we have less than maxEventNum we just append otherwise we see
        //this temporary event stored reaches the exit before appending.
        //Each triple of values is 1 point and each event will have 4 points.
        //Source - Crystal1 - Crystal2 - Detector
        vector<double> tmpEvent_para;
        vector<double> tmpEvent_anti;

        if(UserSettingsInput.Make_Horizontal){
            p = setup->del_teta_L * ((double)uniform(generator) / RAND_MAX) + setup->teta_min_L;
            
            tetadir = p;
        }else
            tetadir = 0;


        sin_tetadir = sin(tetadir);
        cos_tetadir = cos(tetadir);
        tan_tetadir = sin_tetadir / cos_tetadir;

        if(GeoParametersInput.make_table_noise && GeoParametersInput.table_resolution >= 0.01)
        {
            teta_table_thread = GeoParametersInput.teta_table;
            teta_table_thread += table_noise(generator);
            
            #ifdef QT_EXISTS
            
            #ifdef OPENMP
            #pragma omp critical
            {
            #endif
                emit w->setTetaTableSignal(teta_table_thread);
            #ifdef OPENMP
            }
            #endif

            #endif
        }

        cos_tetadirCry1 = cos(tetadir + teta_table_thread * convrad);


        if(UserSettingsInput.Make_Vertical){
            p = setup->del_fi_L * ((double)uniform(generator) / RAND_MAX) + setup->fi_min_L;
            fidir = p;
        }else
            fidir = GeoParametersInput.xsi * convrad;


        sin_fidir = sin(fidir);
        cos_fidir = cos(fidir);
        tan_fidir = sin_fidir / cos_fidir;
        cos_fidirtilt = cos(fidir + setup->tilt_C1_temp);


        if(GeoParapathlengthsInput.type_source == "P"){
            z = GeolengthelementsInput.S_shi_ver_B;
            y = GeolengthelementsInput.S_shi_hor_B;
            r = sqrt(pow(z, 2) + pow(y, 2));

            if(GeolengthelementsInput.S_shi_ver_B == 0 && GeolengthelementsInput.S_shi_hor_B == 0){
                sin_tetap = 0;
                cos_tetap = 1;
            }else if(GeolengthelementsInput.S_shi_hor_B == 0){
                sin_tetap = 1;
                cos_tetap = 0;
            }else{
                var_temp = z / y;
                if(y > 0){
                    sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
                }else{
                    sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
                }
            }

        }else if(GeoParapathlengthsInput.type_source == "UC"){
            r = setup->S_sour_2 + 1;
            while(r > setup->S_sour_2){
                z = ((double)uniform(generator) / RAND_MAX) * GeolengthelementsInput.S_sour - setup->S_sour_2;
                y = ((double)uniform(generator) / RAND_MAX) * GeolengthelementsInput.S_sour - setup->S_sour_2;
                r = sqrt(pow(z, 2) + pow(y, 2));
            }
            
            z += GeolengthelementsInput.S_shi_ver_B;
            y += GeolengthelementsInput.S_shi_hor_B;
            
            
            if(y != 0)
                var_temp = z / y;
            else
                var_temp = 0;

            r = sqrt(pow(z, 2) + pow(y, 2));

            if(y > 0){
                sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
            }else{
                sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
            }
        }else if(GeoParapathlengthsInput.type_source == "G"){
            p = 2 * M_PI * ((double)uniform(generator) / RAND_MAX);
            tetap = p;
            sin_tetap = sin(tetap);
            cos_tetap = cos(tetap);
            r = setup->S_aper_R_2 + 1;

            while(r > setup->S_aper_R_2)
                r = Util::GaussianBox(setup->S_aper_var_2, 0);

            if(! (GeolengthelementsInput.S_shi_ver_B == 0 && GeolengthelementsInput.S_shi_hor_B == 0)){
                z = r * sin_tetap + GeolengthelementsInput.S_shi_ver_B;
                y = r * cos_tetap + GeolengthelementsInput.S_shi_hor_B;
                var_temp = z / y;
                r = sqrt(pow(z, 2) + pow(y, 2));

                if(y > 0){
                    sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
                }else{
                    sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
                }
            }

        }else if(GeoParapathlengthsInput.type_source == "UR"){
            z = ((double)uniform(generator) / RAND_MAX) * GeolengthelementsInput.z_sour - setup->z_sour_2;
            y = ((double)uniform(generator) / RAND_MAX) * GeolengthelementsInput.y_sour - setup->y_sour_2;

            z += GeolengthelementsInput.S_shi_ver_B;
            y += GeolengthelementsInput.S_shi_hor_B;
            var_temp = z / y;
            r = sqrt(pow(z, 2) + pow(y, 2));

            if(y > 0){
                sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
            }else{
                sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
            }
        }else{
            runtime_error("Bad input on the source type: type_source");
        }

        vector<double> yz = Util::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, setup->LT_aper_Db);

        y = yz[0];
        z = yz[1];

        var_temp = pow(y - GeolengthelementsInput.S_shi_hor_A, 2) + pow(z - GeolengthelementsInput.S_shi_ver_A, 2);

        if(var_temp < setup->S_aper_sqr){

            if(GraphOptionsInput.make_image_plates){
                Util::Make(1, y, z,
                            counts_sour,
                            counts_C1,
                            counts_C2_para,
                            counts_detc_para,
                            counts_C2_anti,
                            counts_detc_anti);

                //Event point at source
                tmpEvent_para.push_back(0); //X
                tmpEvent_para.push_back(y); //Y
                tmpEvent_para.push_back(z); //Z
                
                tmpEvent_anti.push_back(0); //X
                tmpEvent_anti.push_back(y); //Y
                tmpEvent_anti.push_back(z); //Z
            }


            r = sqrt(pow(y, 2) + pow(z, 2));

            vector<double> yz = Util::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, setup->LT_aper_Db);

            y = yz[0];
            z = yz[1];


            y_pro_C1 = y * cos_tetadir / cos_tetadirCry1;
            z_pro_C1 = z * cos_fidir / cos_fidirtilt;


            //logString.clear();
            //logString << y_pro_C1 << "\t" << y_max_C1 << "\t" << y_min_C1 << "\t;\t" << z_pro_C1 << "\t" << z_max_C1 << "\t" << z_min_C1 << endl;
            //emit w->LogLine(logString.str());

            if(y_pro_C1 < setup->y_max_C1 && y_pro_C1 > setup->y_min_C1 && z_pro_C1 < setup->z_max_C1 && z_pro_C1 > setup->z_min_C1){
                if(!GraphOptionsInput.make_imageC1_After_refle){
                    if(GraphOptionsInput.make_image_plates){
                        Util::Make(2, y_pro_C1, z_pro_C1,
                                    counts_sour,
                                    counts_C1,
                                    counts_C2_para,
                                    counts_detc_para,
                                    counts_C2_anti,
                                    counts_detc_anti);
                        
                        //Event point at the first crystal
                        tmpEvent_para.push_back(0); //X
                        tmpEvent_para.push_back(y_pro_C1); //Y
                        tmpEvent_para.push_back(z_pro_C1); //Z

                        tmpEvent_anti.push_back(0); //X
                        tmpEvent_anti.push_back(y_pro_C1); //Y
                        tmpEvent_anti.push_back(z_pro_C1); //Z
                    }
                }

                rx = cos_fidir * cos_tetadir;
                ry = cos_fidir * sin_tetadir;
                rz = sin_fidir;


                if(CurvedCrystalInput.Curve_crystall){
                    vector<double> corrRes = Util::horCorr(
                        y_pro_C1,
                        setup->y_max_C1,
                        z_pro_C1,
                        setup->z_max_C1,
                        true);

                    Costeta_CHC = corrRes[0];
                    Sinteta_CHC = corrRes[1];

                    n1x = setup->n1x_temp * Costeta_CHC + setup->n1y_temp * Sinteta_CHC;
                    n1y = setup->n1y_temp * Costeta_CHC - setup->n1x_temp * Sinteta_CHC;
                }

                vector<double> angleRes = Util::getFullAngle(rx, ry, rz, n1x, n1y, setup->n1z);
                angle = angleRes[0];
                r2x = angleRes[1];
                r2y = angleRes[2];
                r2z = angleRes[3];

                lamda = Util::getEnergy(setup->a_lamds_uni, setup->b_lamds_uni, setup->tw_d1_para);
                
                tetabra1 = asin(lamda / setup->tw_d1_para);

                //logString.clear();
                //logString << angle << "\t" << tetabra1 << endl;
                //emit w->LogLine(logString.str());

                bool poliP;

                if(PolarizationParametersInput.mka_poli)
                {
                    if(((double)uniform(generator) / RAND_MAX) < PolarizationParametersInput.relationP_S)
                        poliP = true;
                    else
                        poliP = false;
                }
                else
                {
                    poliP = false;
                }

                first_crystal_reach = Util::getReflection(
                        angle,
                        tetabra1,
                        lamda,
                        false,
                        poliP);

                if(first_crystal_reach){
                    if(GraphOptionsInput.make_imageC1_After_refle){
                        if(GraphOptionsInput.make_image_plates){
                            Util::Make(2, y_pro_C1, z_pro_C1,
                                        counts_sour,
                                        counts_C1,
                                        counts_C2_para,
                                        counts_detc_para,
                                        counts_C2_anti,
                                        counts_detc_anti);
                        }
                    }
                    
                    rx_rot = setup->cos_tetartab * r2x + setup->sin_tetartab * r2y;
                    ry_rot = -setup->sin_tetartab * r2x + setup->cos_tetartab * r2y;

                    //logString.clear();
                    //logString << ry_rot << "\t" << -sin_tetartab << "\t" << r2x << "\t" << cos_tetartab << "\t" << r2y << endl;
                    //emit w->LogLine(logString.str());

                    var_temp = ry_rot / rx_rot;

                    tetadir = atan(var_temp);
                    tan_tetadir = var_temp;
                    cos_tetadir = cos(tetadir);


                    var_temp = r2z / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                    fidir = atan(var_temp);
                    tan_fidir = var_temp;
                    cos_fidir = cos(fidir);
                    cos_fidirtilt2_para = cos(fidir + setup->tilt_C2_para_temp);
                    cos_fidirtilt2_anti = cos(fidir + setup->tilt_C2_anti_temp);

                    corr_dis = y_pro_C1 * setup->cos_difteC1_Ta;

                    y_pro_C1 = -y_pro_C1 * setup->sin_difteC1_Ta;


                    if(y_pro_C1 == 0){
                        cos_tetap = 0;
                        if(z_pro_C1 > 0)
                            sin_tetap = 1;
                        else
                            sin_tetap = -1;
                    }else{
                        var_temp = z_pro_C1 / y_pro_C1;
                        r = sqrt(pow(z_pro_C1, 2) + pow(y_pro_C1, 2));

                        if(y_pro_C1 > 0){
                            sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                            cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
                        }else{
                            sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                            cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
                        }

                    }

                    //logString.clear();
                    //logString << cos_tetap << "\t" << tan_tetadir << "\t" << dist_Cr1_Cr2_Db << "\t" << corr_dis << endl;
                    //emit w->LogLine(logString.str());

                    vector<double> yz = Util::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, setup->dist_Cr1_Cr2_Db - corr_dis);

                    y = yz[0];
                    z = yz[1];


                    if(UserSettingsInput.see_para){
                        sin_tetatab_del_dir = sin(setup->tetaref - setup->delrot + tetadir);
                        y_pro_C1 = y * cos_tetadir / sin_tetatab_del_dir;
                        z_pro_C1 = z * cos_fidir / cos_fidirtilt2_para;
                        
                        //logString.clear();
                        //logString << y_pro_C1 << "\t" << y_max_C2 << "\t" << y_min_C2 << "\t;\t" << z_pro_C1 << "\t" << z_max_C2 << "\t" << z_min_C2 << endl;
                        //emit w->LogLine(logString.str());

                        if(y_pro_C1 < setup->y_max_C2 && y_pro_C1 > setup->y_min_C2 && z_pro_C1 < setup->z_max_C2 && z_pro_C1 > setup->z_min_C2){

                            if(GraphOptionsInput.make_image_plates){
                                if(!GraphOptionsInput.make_imageC2_After_refle){
                                    Util::Make(3, y_pro_C1, z_pro_C1,
                                                    counts_sour,
                                                    counts_C1,
                                                    counts_C2_para,
                                                    counts_detc_para,
                                                    counts_C2_anti,
                                                    counts_detc_anti);
                                    
                                    //Event point at second crystal in parallel
                                    tmpEvent_para.push_back(0); //X
                                    tmpEvent_para.push_back(y_pro_C1); //Y
                                    tmpEvent_para.push_back(z_pro_C1); //Z
                                }
                            }

                            
                            tetabra2 = asin(lamda / setup->tw_d2_para);


                            if(CurvedCrystalInput.Curve_crystall){
                                vector<double> corrRes = Util::horCorr(
                                    y_pro_C1,
                                    setup->y_max_C2,
                                    z_pro_C1,
                                    setup->z_max_C2,
                                    false);

                                Costeta_CHC = corrRes[0];
                                Sinteta_CHC = corrRes[1];

                                n2x_para = setup->n2x_para_temp * Costeta_CHC + setup->n2y_para_temp * Sinteta_CHC;
                                n2y_para = setup->n2y_para_temp * Costeta_CHC - setup->n2x_para_temp * Sinteta_CHC;
                            }


                            vector<double> angleRes = Util::getFullAngle2(r2x, r2y, r2z, n2x_para, n2y_para, setup->n2z_para);
                            angle = angleRes[0];
                            r3x = angleRes[1];
                            r3y = angleRes[2];
                            r3z = angleRes[3];

                            sec_crystal_Parallel_reach = Util::getReflection(
                                angle,
                                tetabra2,
                                lamda,
                                true,
                                poliP);

                            if(sec_crystal_Parallel_reach){

                                if(GraphOptionsInput.make_image_plates){
                                    if(GraphOptionsInput.make_imageC2_After_refle){
                                        Util::Make(3, y_pro_C1, z_pro_C1,
                                                    counts_sour,
                                                    counts_C1,
                                                    counts_C2_para,
                                                    counts_detc_para,
                                                    counts_C2_anti,
                                                    counts_detc_anti);
                                    }
                                }


                                rx_rot_sec = setup->cos_tetartabdete_para * r3x + setup->sin_tetartabdete_para * r3y;
                                ry_rot_sec = -setup->sin_tetartabdete_para * r3x + setup->cos_tetartabdete_para * r3y;

                                var_temp = ry_rot_sec / rx_rot_sec;



                                tetadir_det = atan(var_temp);
                                tan_tetadir_det = var_temp;
                                cos_tetadir_det = cos(tetadir_det);


                                var_temp = rz / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                                fidir_det = atan(var_temp);
                                tan_fidir_det = var_temp;
                                cos_fidir_det = cos(fidir_det);
                                cos_fidirtilt2_para = cos(fidir + setup->tilt_C2_para_temp);

                                corr_dis_d_pa = -y_pro_C1 * setup->cos_difteC2_det_para;
                                y_pro_C1_d_pa = -y_pro_C1 * setup->sin_difteC2_det_para;


                                if(y_pro_C1_d_pa == 0){
                                    cos_tetap_det = 0;
                                    if(z_pro_C1 > 0)
                                        sin_tetap_det = 1;
                                    else
                                        sin_tetap_det = -1;
                                }else{
                                    var_temp = z_pro_C1 / y_pro_C1_d_pa;
                                    r_det = sqrt(pow(z_pro_C1, 2) + pow(y_pro_C1_d_pa, 2));

                                    if(y_pro_C1_d_pa > 0){
                                        sin_tetap_det = var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = 1 / sqrt(1 + pow(var_temp, 2));
                                    }else{
                                        sin_tetap_det = - var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = - 1 / sqrt(1 + pow(var_temp, 2));
                                    }
                                }

                                vector<double> yz = Util::getYZ(r_det, sin_tetap_det, cos_tetap_det, tan_tetadir_det, tan_fidir_det, setup->dist_Cr2_det_Db - corr_dis_d_pa);

                                y_det = yz[0];
                                z_det = yz[1];

                                if(y_det < setup->ydetc_2_max && y_det > setup->ydetc_2_min && z_det < setup->zdetc_2_max && z_det > setup->zdetc_2_min){

                                    if(GraphOptionsInput.make_image_plates){
                                        Util::Make(4, y_det, z_det,
                                                    counts_sour,
                                                    counts_C1,
                                                    counts_C2_para,
                                                    counts_detc_para,
                                                    counts_C2_anti,
                                                    counts_detc_anti);

                                        //Event point at detector in parallel
                                        tmpEvent_para.push_back(0); //X
                                        tmpEvent_para.push_back(y_det); //Y
                                        tmpEvent_para.push_back(z_det); //Z

                                        #ifdef OPENMP
                                        #pragma omp critical
                                        {
                                        #endif
                                        if (eventsToTrace_para->size() < NumberRaysInput.number_events) {
                                            eventsToTrace_para->push_back(tmpEvent_para);
                                        }
                                        else {
                                            eventsToTrace_para->erase(eventsToTrace_para->begin());
                                            eventsToTrace_para->push_back(tmpEvent_para);
                                        }
                                        #ifdef OPENMP
                                        }
                                        #endif
                                    }

                                    toint_para++;
                                    #ifdef OPENMP
                                    energy_sum_para_thread += Convert_Ag_minusone_eV / lamda;
                                    #else
                                    reduce->energy_sum_para->at(setup->numbins - 1) += Convert_Ag_minusone_eV / lamda;
                                    #endif
                                }
                                else {
                                    if (GraphOptionsInput.make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        #ifdef OPENMP
                                        #pragma omp critical
                                        {
                                        #endif
                                        if (eventsToTrace_para->size() < NumberRaysInput.number_events && tmpEvent_para.size() >= 6) {
                                            eventsToTrace_para->push_back(tmpEvent_para);
                                        }
                                        #ifdef OPENMP
                                        }
                                        #endif
                                    }
                                }
                            }
                            else {
                                if (GraphOptionsInput.make_image_plates) {
                                    //If the event does not reach the detector then only add when we have less than maxEventNum
                                    #ifdef OPENMP
                                    #pragma omp critical
                                    {
                                    #endif
                                    if (eventsToTrace_para->size() < NumberRaysInput.number_events && tmpEvent_para.size() >= 6) {
                                        eventsToTrace_para->push_back(tmpEvent_para);
                                    }
                                    #ifdef OPENMP
                                    }
                                    #endif
                                }
                            }
                        }
                        else {
                            if (GraphOptionsInput.make_image_plates) {
                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                #ifdef OPENMP
                                #pragma omp critical
                                {
                                #endif
                                if (eventsToTrace_para->size() < NumberRaysInput.number_events && tmpEvent_para.size() >= 6) {
                                    eventsToTrace_para->push_back(tmpEvent_para);
                                }
                                #ifdef OPENMP
                                }
                                #endif
                            }
                        }

                    }


                    if(UserSettingsInput.see_anti){


                        sin_tetatab_del_dir = sin(setup->tetaref + setup->delrot - tetadir);

                        y_pro_C1 = y * cos_tetadir / sin_tetatab_del_dir;
                        z_pro_C1 = z * cos_fidir / cos_fidirtilt2_anti;

                        
                        if(y_pro_C1 < setup->y_max_C2 && y_pro_C1 > setup->y_min_C2 && z_pro_C1 < setup->z_max_C2 && z_pro_C1 > setup->z_min_C2){


                            if(GraphOptionsInput.make_imageC2_After_refle){
                                if(GraphOptionsInput.make_image_plates){
                                    Util::Make(5, y_pro_C1, z_pro_C1,
                                                counts_sour,
                                                counts_C1,
                                                counts_C2_para,
                                                counts_detc_para,
                                                counts_C2_anti,
                                                counts_detc_anti);

                                    //Event point at second crystal in antiparallel
                                    tmpEvent_anti.push_back(0); //X
                                    tmpEvent_anti.push_back(y_pro_C1); //Y
                                    tmpEvent_anti.push_back(z_pro_C1); //Z
                                }
                            }

                            
                            tetabra2 = asin(lamda / setup->tw_d2_anti);

                            if(CurvedCrystalInput.Curve_crystall){
                                vector<double> corrRes = Util::horCorr(
                                    -y_pro_C1,
                                    setup->y_max_C2,
                                    z_pro_C1,
                                    setup->z_max_C2,
                                    false);

                                Costeta_CHC = corrRes[0];
                                Sinteta_CHC = corrRes[1];

                                n2x_anti = setup->n2x_anti_temp * Costeta_CHC + setup->n2y_anti_temp * Sinteta_CHC;
                                n2y_anti = setup->n2y_anti_temp * Costeta_CHC - setup->n2x_anti_temp * Sinteta_CHC;
                            }


                            vector<double> angleRes = Util::getFullAngle2(r2x, r2y, r2z, n2x_anti, n2y_anti, setup->n2z_anti);
                            angle = angleRes[0];
                            r3x = angleRes[1];
                            r3y = angleRes[2];
                            r3z = angleRes[3];


                            sec_crystal_Antiparallel_reach = Util::getReflection(
                                angle,
                                tetabra2,
                                lamda,
                                true,
                                poliP);

                            if(sec_crystal_Antiparallel_reach){

                                if(GraphOptionsInput.make_image_plates){
                                    if(GraphOptionsInput.make_imageC2_After_refle){
                                        Util::Make(5, y_pro_C1, z_pro_C1,
                                                    counts_sour,
                                                    counts_C1,
                                                    counts_C2_para,
                                                    counts_detc_para,
                                                    counts_C2_anti,
                                                    counts_detc_anti);
                                    }
                                }


                                rx_rot_sec = setup->cos_tetartabdete_anti * r3x + setup->sin_tetartabdete_anti * r3y;
                                ry_rot_sec = - setup->sin_tetartabdete_anti * r3x + setup->cos_tetartabdete_anti * r3y;


                                var_temp = - ry_rot_sec / rx_rot_sec;
                                tetadir_det = atan(var_temp);
                                tan_tetadir_det = var_temp;
                                cos_tetadir_det = cos(tetadir_det);


                                var_temp = rz / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                                fidir_det = atan(var_temp);
                                tan_fidir_det = var_temp;
                                cos_fidir_det = cos(fidir_det);
                                cos_fidirtilt2_anti = cos(fidir + setup->tilt_C2_anti_temp);

                                corr_dis_d_pa = y_pro_C1 * setup->cos_difteC2_det_para;
                                y_pro_C1_d_pa = y_pro_C1 * setup->sin_difteC2_det_para;


                                if(y_pro_C1_d_pa == 0){
                                    cos_tetap_det = 0;
                                    if(z_pro_C1 > 0)
                                        sin_tetap_det = 1;
                                    else
                                        sin_tetap_det = -1;
                                }else{
                                    var_temp = z_pro_C1 / y_pro_C1_d_pa;
                                    r_det = sqrt(pow(z_pro_C1, 2) + pow(y_pro_C1_d_pa, 2));

                                    if(y_pro_C1_d_pa > 0){
                                        sin_tetap_det = var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = 1 / sqrt(1 + pow(var_temp, 2));
                                    }else{
                                        sin_tetap_det = - var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = - 1 / sqrt(1 + pow(var_temp, 2));
                                    }
                                }


                                vector<double> yz = Util::getYZ(r_det, sin_tetap_det, cos_tetap_det, tan_tetadir_det, tan_fidir_det, setup->dist_Cr2_det_Db - corr_dis_d_pa);

                                y_det = yz[0];
                                z_det = yz[1];

                                if(y_det < setup->ydetc_2_max && y_det > setup->ydetc_2_min && z_det < setup->zdetc_2_max && z_det > setup->zdetc_2_min){
                                    if (GraphOptionsInput.make_image_plates) {
                                        Util::Make(6, y_det, z_det,
                                                counts_sour,
                                                counts_C1,
                                                counts_C2_para,
                                                counts_detc_para,
                                                counts_C2_anti,
                                                counts_detc_anti);

                                        
                                        //Event point at detector in antiparallel
                                        tmpEvent_anti.push_back(0); //X
                                        tmpEvent_anti.push_back(y_det); //Y
                                        tmpEvent_anti.push_back(z_det); //Z

                                        #ifdef OPENMP
                                        #pragma omp critical
                                        {
                                        #endif
                                        if (eventsToTrace_anti->size() < NumberRaysInput.number_events) {
                                            eventsToTrace_anti->push_back(tmpEvent_anti);
                                        }
                                        else {
                                            eventsToTrace_anti->erase(eventsToTrace_anti->begin());
                                            eventsToTrace_anti->push_back(tmpEvent_anti);
                                        }
                                        #ifdef OPENMP
                                        }
                                        #endif
                                    }

                                    toint_anti++;
                                    #ifdef OPENMP
                                    energy_sum_anti_thread += Convert_Ag_minusone_eV / lamda;
                                    #else
                                    reduce->energy_sum_anti->at(setup->numbins - 1) += Convert_Ag_minusone_eV / lamda;
                                    #endif
                                }
                                else {
                                    if (GraphOptionsInput.make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        #ifdef OPENMP
                                        #pragma omp critical
                                        {
                                        #endif
                                        if (eventsToTrace_anti->size() < NumberRaysInput.number_events && tmpEvent_anti.size() >= 6) {
                                            eventsToTrace_anti->push_back(tmpEvent_anti);
                                        }
                                        #ifdef OPENMP
                                        }
                                        #endif
                                    }
                                }

                            }
                            else {
                                if (GraphOptionsInput.make_image_plates) {
                                    //If the event does not reach the detector then only add when we have less than maxEventNum
                                    #ifdef OPENMP
                                    #pragma omp critical
                                    {
                                    #endif
                                    if (eventsToTrace_anti->size() < NumberRaysInput.number_events && tmpEvent_anti.size() >= 6) {
                                        eventsToTrace_anti->push_back(tmpEvent_anti);
                                    }
                                    #ifdef OPENMP
                                    }
                                    #endif
                                }
                            }

                        }
                        else {
                            if (GraphOptionsInput.make_image_plates) {
                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                #ifdef OPENMP
                                #pragma omp critical
                                {
                                #endif
                                if (eventsToTrace_anti->size() < NumberRaysInput.number_events && tmpEvent_anti.size() >= 6) {
                                    eventsToTrace_anti->push_back(tmpEvent_anti);
                                }
                                #ifdef OPENMP
                                }
                                #endif
                            }
                        }
                    }
                }
            }
        }

        #ifdef OPENMP
        #pragma omp atomic
        #endif
        I++;

        #ifdef QT_EXISTS
            #ifdef OPENMP
            #pragma omp atomic
            #endif
            total_current_bins += 1;
            w->setPctDone(static_cast<float>(total_current_bins) / setup->total_expexted_bins);
        #endif
    }


    #ifdef OPENMP
    reduce->energy_sum_para_thread = energy_sum_para_thread; reduce->energy_sum_anti_thread = energy_sum_anti_thread;
    #endif
    reduce->toint_para = toint_para; reduce->toint_anti = toint_anti; reduce->counts_sour = counts_sour; reduce->counts_C1 = counts_C1;
    reduce->counts_C2_para = counts_C2_para; reduce->counts_C2_anti = counts_C2_anti; reduce->counts_detc_para = counts_detc_para; reduce->counts_detc_anti = counts_detc_anti; 

    bin->seed = seed; bin->teta_table_thread = teta_table_thread; bin->p = p; bin->tetadir = tetadir; bin->sin_tetadir = sin_tetadir; bin->cos_tetadir = cos_tetadir;
    bin->tan_tetadir = tan_tetadir; bin->cos_tetadirCry1 = cos_tetadirCry1; bin->fidir = fidir; bin->sin_fidir = sin_fidir; bin->cos_fidir = cos_fidir;
    bin->tan_fidir = tan_fidir; bin->cos_fidirtilt = cos_fidirtilt; bin->z = z; bin->y = y; bin->r = r; bin->tetap = tetap; bin->sin_tetap = sin_tetap;
    bin->cos_tetap = cos_tetap; bin->var_temp = var_temp; bin->y_pro_C1 = y_pro_C1; bin->z_pro_C1 = z_pro_C1; bin->rx = rx; bin->ry = ry; bin->rz = rz;
    bin->Costeta_CHC = Costeta_CHC; bin->Sinteta_CHC = Sinteta_CHC; bin->n1x = n1x; bin->n1y = n1y; bin->angle = angle; bin->r2x = r2x; bin->r2y = r2y; bin->r2z = r2z;
    bin->lamda = lamda; bin->tetabra1 = tetabra1; bin->first_crystal_reach = first_crystal_reach; bin->rx_rot = rx_rot; bin->ry_rot = ry_rot;
    bin->cos_fidirtilt2_para = cos_fidirtilt2_para; bin->cos_fidirtilt2_anti = cos_fidirtilt2_anti; bin->corr_dis = corr_dis; bin->sin_tetatab_del_dir = sin_tetatab_del_dir;
    bin->tetabra2 = tetabra2; bin->n2x_para = n2x_para; bin->n2y_para = n2y_para; bin->r3x = r3x; bin->r3y = r3y; bin->r3z = r3z;
    bin->sec_crystal_Parallel_reach = sec_crystal_Parallel_reach; bin->rx_rot_sec = rx_rot_sec; bin->ry_rot_sec = ry_rot_sec; bin->tetadir_det = tetadir_det;
    bin->tan_tetadir_det = tan_tetadir_det; bin->cos_tetadir_det = cos_tetadir_det; bin->fidir_det = fidir_det; bin->tan_fidir_det = tan_fidir_det;
    bin->cos_fidir_det = cos_fidir_det; bin->corr_dis_d_pa = corr_dis_d_pa; bin->y_pro_C1_d_pa = y_pro_C1_d_pa; bin->cos_tetap_det = cos_tetap_det;
    bin->sin_tetap_det = sin_tetap_det; bin->r_det = r_det; bin->y_det = y_det; bin->z_det = z_det; bin->n2x_anti = n2x_anti; bin->n2y_anti = n2y_anti;
    bin->sec_crystal_Antiparallel_reach = sec_crystal_Antiparallel_reach; bin->total_current_bins = total_current_bins;
}
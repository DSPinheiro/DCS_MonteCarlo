/*
 * Source_complex.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#include "source_complex.hh"
#include "Util.h"


using namespace std;



bool Source_complex::run_Source(SimulationInterface *w){

    stringstream logString;
    
    double counts_C2_para_t, counts_detc_para_t, counts_C2_anti_t, counts_detc_anti_t;
    
    double max_valu_para, max_valu_anti, S_aper_R_2, S_aper_var_2, S_aper_sqr, S_sour_2, z_sour_2, y_sour_2, zdetc_2_max, zdetc_2_min, ydetc_2_max, ydetc_2_min, tetaref, tetabrag_ref;
    double cos_e, sin_e, n1x, n1y, n1z, delrot_min, delrot_max, delrot_inc, cos_tilt_C1, sin_tilt_C1;
    double delrot, d_lat1_para, d_lat2_para, d_lat1_anti, d_lat2_anti, tw_d1_para, tw_d2_para, tw_d1_anti, tw_d2_anti, a_lamds_uni, b_lamds_uni, auxBragg;
    double n2x_para, n2y_para, n2x_anti, n2y_anti, n2z_para, n2z_anti, p, tetadir, tilt_C1_temp, n1x_temp, n1y_temp;
    double fidir, z, y, r, tetap, sin_tetadir, cos_tetadir, rx, ry, rz, angle, lamda, tetabra1, tetabra2, teta_max_L, teta_min_L, del_teta_L, inc_tem;
    double angle_para, angle_anti, r2x, r2y, r2z, r3x, r3y, r3z, pha_temp[4], fi_max_L, fi_min_L, del_fi_L, tetartab;
    double cos_tetartab, sin_tetartab, cos_difteC1_Ta, sin_difteC1_Ta, cos_tetartabdete_para, sin_tetartabdete_para, cos_tetartabdete_anti, sin_tetartabdete_anti, z_max_C2, z_min_C2;
    double LT_aper_Db, dist_T_Cr1_Db, dist_Cr1_Cr2_Db, dist_Cr2_det_Db, y_min_aper, y_max_aper, z_min_aper, z_max_aper, y_min_C1, y_max_C1, y_min_C2, y_max_C2, z_max_C1, z_min_C1;
    double tilt_C2_para_temp, tilt_C2_anti_temp, n2x_para_temp, n2y_para_temp, n2x_anti_temp;
    double n2y_anti_temp, cos_difteC2_det_para, sin_difteC2_det_para, tan_tetadir, cos_tetadirCry1, sin_fidir, cos_fidir, tan_fidir, cos_fidirtilt, sin_tetap, cos_tetap, var_temp;
    double y_pro_C1, z_pro_C1, Costeta_CHC, Sinteta_CHC, rx_rot, ry_rot, cos_fidirtilt2_para, cos_fidirtilt2_anti, corr_dis, sin_tetatab_del_dir, rx_rot_sec, ry_rot_sec;
    double tetadir_det, tan_tetadir_det, cos_tetadir_det, fidir_det, tan_fidir_det, cos_fidir_det, corr_dis_d_pa, y_pro_C1_d_pa, cos_tetap_det, sin_tetap_det, z_det, y_det, r_det;

    //Old unused variables
    //double sin_tetadirCry1, teta_min, tan_e, del_fi, cos2_e, abs_fi_max, squa_tilt2, Dis_total
    //double teta_max, squa_tilt1, twtilt_C1, fi_max, cosdel_othe, cosdel_teta, cosdel_teta_othe
    //double sin_fi, cos_fi, toint_para_nor, toint_anti_nor, del_teta, cosdel, fi_min, sin_teref_tedi
    //bool reach
    
    vector<int> int_time_out;
    int int_time_out_begg, int_time_mili_out_begg, toint_para, toint_anti, total_para, total_anti, bin_tem = 1, bin_fas = 1;
    int counts_c2_para, counts_c2_anti;

    int numbins, max_para, I, n_rota;
    int64_t total_current_bins = 0, total_expexted_bins = NumberRaysInput.number_rotati * NumberRaysInput.nbeams * PlotParametersInput.nubins;

    int* toint_para_total = new int[PlotParametersInput.nubins];
    int* toint_anti_total = new int[PlotParametersInput.nubins];

    fill(toint_para_total, toint_para_total + PlotParametersInput.nubins, 0);
    fill(toint_anti_total, toint_anti_total + PlotParametersInput.nubins, 0);
    
    
    bool make_G_function, sec_crystal_Parallel_reach, sec_crystal_Antiparallel_reach, first_crystal_reach, cond_rotation, poliP;

    sec_crystal_Parallel_reach = false;
    sec_crystal_Antiparallel_reach = false;
    max_valu_para = 0;
    max_valu_anti = 0;


    make_G_function = false;


    pha_temp[0] = ((double)rand() / RAND_MAX) * 2 * M_PI;
    pha_temp[1] = ((double)rand() / RAND_MAX) * 2 * M_PI;
    pha_temp[2] = ((double)rand() / RAND_MAX) * 2 * M_PI;
    pha_temp[3] = ((double)rand() / RAND_MAX) * 2 * M_PI;


    d_lat1_para = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_1_para);
    d_lat1_anti = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_1_anti);
    d_lat2_para = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_para);
    d_lat2_anti = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_anti);


    tw_d1_para = 2 * d_lat1_para;
    tw_d1_anti = 2 * d_lat1_anti;
    tw_d2_para = 2 * d_lat2_para;
    tw_d2_anti = 2 * d_lat2_anti;


    S_aper_R_2 = GeolengthelementsInput.S_aper / 2;
    S_aper_var_2 = GeolengthelementsInput.S_aper_var / 2;
    S_aper_sqr = pow(S_aper_R_2, 2);
    S_sour_2 = GeolengthelementsInput.S_sour / 2;

    z_sour_2 = GeolengthelementsInput.z_sour / 2;
    y_sour_2 = GeolengthelementsInput.y_sour / 2;


    zdetc_2_max = GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver;
    zdetc_2_min = -GeolengthelementsInput.zdetc / 2 + GeolengthelementsInput.shift_det_ver;
    ydetc_2_max = GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver;
    ydetc_2_min = -GeolengthelementsInput.ydetc / 2 + GeolengthelementsInput.shift_det_ver;


    delrot_min = Mini_angl * M_PI / 180;
    delrot_max = Maxi_angl * M_PI / 180;
    delrot_inc = (delrot_max - delrot_min) / (double)PlotParametersInput.nubins;


    delrot = delrot_max;


    numbins = 0;


    tetaref = M_PI / 2 - (teta_crys1 * M_PI / 180);

    cos_e = cos(tetaref);
    sin_e = sin(tetaref);


    if(CurveVerticalTiltInput.make_CurveTilt)
        tilt_C1_temp = Util::ObtainVert(1, 0);
    else
        tilt_C1_temp = GeoParametersInput.tilt_C1 * convrad;


    cos_tilt_C1 = cos(tilt_C1_temp);
    sin_tilt_C1 = sin(tilt_C1_temp);

    n1x = -cos_tilt_C1 * sin_e;
    n1y = cos_tilt_C1 * cos_e;
    n1z = sin_tilt_C1;


    tetabrag_ref = asin(-n1x);

    if(CurvedCrystalInput.Curve_crystall){
        n1x_temp = n1x;
        n1y_temp = n1y;
    }


    if(GeoParapathlengthsInput.type_source == "UC"){
        teta_max_L = atan((GeolengthelementsInput.S_aper - GeolengthelementsInput.S_shi_hor_B + GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper);
        teta_min_L = - atan((GeolengthelementsInput.S_aper + GeolengthelementsInput.S_shi_hor_B - GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper);
    }else{
        teta_max_L = atan(min((GeolengthelementsInput.y_aper / 2 + y_sour_2 + GeolengthelementsInput.S_shi_hor_B - GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.y_first_crys / 2 + y_sour_2 - GeolengthelementsInput.S_shi_hor_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        teta_min_L = - atan(min((GeolengthelementsInput.y_aper / 2 + y_sour_2 - GeolengthelementsInput.S_shi_hor_B + GeolengthelementsInput.S_shi_hor_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.y_first_crys / 2 + y_sour_2 + GeolengthelementsInput.S_shi_hor_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
    }


    if(FullEnergySpectrumInput.make_more_lines == 0){
        auxBragg = asin(picks[1].lamda / tw_d1_para);
        teta_max_L = min(M_PI / 2 + GeoParametersInput.teta_table * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg + limitReflec, teta_max_L);
        teta_min_L = max(M_PI / 2 + GeoParametersInput.teta_table * M_PI / 180 + GeoParametersInput.Exp_crys1 * M_PI / 180 - auxBragg - limitReflec, teta_min_L);
    }


    del_teta_L = teta_max_L - teta_min_L;


    if(GeoParapathlengthsInput.type_source == "UC"){
        fi_max_L = atan((GeolengthelementsInput.S_aper - GeolengthelementsInput.S_shi_ver_B + GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper);
        fi_min_L = - atan((GeolengthelementsInput.S_aper + GeolengthelementsInput.S_shi_ver_B - GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper);
    }else{
        fi_max_L = atan(min((GeolengthelementsInput.z_aper / 2 + z_sour_2 + GeolengthelementsInput.S_shi_ver_B - GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.z_first_crys / 2 + z_sour_2 - GeolengthelementsInput.S_shi_ver_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
        fi_min_L = - atan(min((GeolengthelementsInput.z_aper / 2 + z_sour_2 - GeolengthelementsInput.S_shi_ver_B + GeolengthelementsInput.S_shi_ver_A) / GeoParapathlengthsInput.LT_aper, (GeolengthelementsInput.z_first_crys / 2 + z_sour_2 + GeolengthelementsInput.S_shi_ver_A) / (GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1)));
    }

    del_fi_L = fi_max_L - fi_min_L;

    tetartab = GeoParametersInput.teta_table * convrad / 2;
    cos_tetartab = cos(GeoParametersInput.teta_table * convrad);
    sin_tetartab = sin(GeoParametersInput.teta_table * convrad);
    cos_difteC1_Ta = cos(GeoParametersInput.teta_table * convrad - tetaref);
    sin_difteC1_Ta = sin(GeoParametersInput.teta_table * convrad - tetaref);

    cos_tetartabdete_para = cos(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
    sin_tetartabdete_para = sin(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);

    cos_tetartabdete_anti = cos(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_anti * convrad);
    sin_tetartabdete_anti = sin(GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_anti * convrad);

    max_hist[0] = 0;
    max_hist[1] = 0;


    if(tw_d1_anti < tw_d2_anti){
        a_lamds_uni = tw_d1_anti * sin(tetaref + delrot_min - 0.2);
        b_lamds_uni = tw_d2_anti * sin(tetaref + delrot_max + 0.2);
    }else{
        a_lamds_uni = tw_d2_anti * sin(tetaref + delrot_min - 0.2);
        b_lamds_uni = tw_d1_anti * sin(tetaref + delrot_max + 0.2);
    }

    b_lamds_uni = b_lamds_uni - a_lamds_uni;


    counts_sour = 0;
    counts_C1 = 0;
    counts_C2_para = 0;
    counts_C2_anti = 0;
    counts_detc_para = 0;
    counts_detc_anti = 0;
    counts_C2_para_t = 0;
    counts_detc_para_t = 0;
    counts_C2_anti_t = 0;
    counts_detc_anti_t = 0;

    min_plot[0][0] = Mini_angl + teta_crys1;
    min_plot[1][0] = Mini_angl - teta_crys1;
    max_plot[0][0] = Maxi_angl + teta_crys1;
    max_plot[1][0] = Maxi_angl - teta_crys1;


    LT_aper_Db = (double)GeoParapathlengthsInput.LT_aper;
    dist_T_Cr1_Db = (double)GeoParapathlengthsInput.dist_T_Cr1;
    dist_Cr1_Cr2_Db = (double)GeoParapathlengthsInput.dist_Cr1_Cr2;
    dist_Cr2_det_Db = (double)GeoParapathlengthsInput.dist_Cr2_Det;


    y_min_aper = GeolengthelementsInput.S_shi_hor_A - GeolengthelementsInput.y_aper / 2;
    y_max_aper = GeolengthelementsInput.S_shi_hor_A + GeolengthelementsInput.y_aper / 2;
    z_min_aper = GeolengthelementsInput.S_shi_ver_A - GeolengthelementsInput.z_aper / 2;
    z_max_aper = GeolengthelementsInput.S_shi_ver_A + GeolengthelementsInput.z_aper / 2;

    y_min_C1 = -GeolengthelementsInput.y_first_crys / 2;
    y_max_C1 = GeolengthelementsInput.y_first_crys / 2;
    y_min_C2 = y_min_C1;
    y_max_C2 = y_max_C1;


    if(UserSettingsInput.center_Mask){
        z_max_C1 = 0.6;
        z_min_C1 = - 0.6;

        y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
        y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
    }else{
        if(UserSettingsInput.mask_C1 == 0){
            z_max_C1 = GeolengthelementsInput.z_first_crys / 2;
            z_min_C1 = -GeolengthelementsInput.z_first_crys / 2;
        }else if(UserSettingsInput.mask_C1 == 1){
            z_max_C1 = GeolengthelementsInput.z_first_crys / 2 - 0.2;
            z_min_C1 = 0;
            y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
            y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        }else if(UserSettingsInput.mask_C1 == 2){
            z_max_C1 = 0;
            z_min_C1 = -GeolengthelementsInput.z_first_crys / 2 + 0.2;
            y_max_C1 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
            y_min_C1 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
        }
    }


    if(UserSettingsInput.mask_C2 == 0){
        z_max_C2 = GeolengthelementsInput.z_first_crys / 2;
        z_min_C2 = -GeolengthelementsInput.z_first_crys / 2;
    }else if(UserSettingsInput.mask_C2 == 1){
        z_max_C2 = GeolengthelementsInput.z_first_crys / 2 - 0.2;
        z_min_C2 = 0;
        y_max_C2 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
        y_min_C2 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
    }else if(UserSettingsInput.mask_C2 == 2){
        z_max_C2 = 0;
        z_min_C2 = -GeolengthelementsInput.z_first_crys / 2 + 0.2;
        y_max_C2 = GeolengthelementsInput.y_first_crys / 2 - 0.2;
        y_min_C2 = -GeolengthelementsInput.y_first_crys / 2 + 0.2;
    }

    if(GraphOptionsInput.make_graph_profile && (!GraphOptionsInput.make_image_plates)){
        //implement gui plotting
    }

    if(GraphOptionsInput.make_image_plates){
        Util::initPlates();
    }

    int_time_out.push_back(0);
    int_time_out.push_back(0);
    int_time_out = Obtain_time::simuTime(0, 0, int_time_out[0], int_time_out[1], w);
    int_time_out_begg = int_time_out[0];
    int_time_mili_out_begg = int_time_out[1];


    n_rota = 1;

    vector<double> energy_sum_para;
    vector<double> energy_sum_anti;

    energy_sum_para.resize(PlotParametersInput.nubins);
    energy_sum_anti.resize(PlotParametersInput.nubins);
    
    while(n_rota <= NumberRaysInput.number_rotati){
        if(n_rota % 2 == 1)
            numbins = 0;
        else
            numbins++;


        cond_rotation = true;

        while(cond_rotation){
            if(n_rota % 2 == 1)
                numbins++;
            else
                numbins--;


            toint_para = 0;
            toint_anti = 0;

            counts_c2_para = 0;
            counts_c2_anti = 0;


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

                tw_d1_para = 2 * d_lat1_para;
                tw_d1_anti = 2 * d_lat1_anti;
                tw_d2_para = 2 * d_lat2_para;
                tw_d2_anti = 2 * d_lat2_anti;


                logString << "tw_1: " << d_lat1_para << endl;
                logString << "tw_2: " << d_lat2_para << endl;
                
                #ifdef QT_EXISTS
                    emit w->LogLineSignal(logString.str());
                #else
                    cout << logString.str();
                #endif

            }

            angle_para = delrot * 180 / M_PI + teta_crys1;
            angle_anti = delrot * 180 / M_PI - teta_crys1;


            if(CurveVerticalTiltInput.make_CurveTilt){
                tilt_C2_para_temp = Util::ObtainVert(2, angle_para);
                tilt_C2_anti_temp = Util::ObtainVert(2, angle_anti);
            }else{
                tilt_C2_para_temp = GeoParametersInput.tilt_C2 * convrad;
                tilt_C2_anti_temp = GeoParametersInput.tilt_C2 * convrad;
            }


            n2x_para = cos(tilt_C2_para_temp) * sin(GeoParametersInput.teta_table * convrad - tetaref + delrot);
            n2y_para = - cos(tilt_C2_para_temp) * cos(GeoParametersInput.teta_table * convrad - tetaref + delrot);
            n2z_para = sin(tilt_C2_para_temp);
            n2x_anti = - cos(tilt_C2_anti_temp) * sin(GeoParametersInput.teta_table * convrad + tetaref + delrot);
            n2y_anti = cos(tilt_C2_anti_temp) * cos(GeoParametersInput.teta_table * convrad + tetaref + delrot);
            n2z_anti = sin(tilt_C2_anti_temp);

            if(CurvedCrystalInput.Curve_crystall){
                n2x_para_temp = n2x_para;
                n2y_para_temp = n2y_para;
                n2x_anti_temp = n2x_anti;
                n2y_anti_temp = n2y_anti;
            }


            cos_difteC2_det_para = cos(tetaref - delrot + GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);
            sin_difteC2_det_para = sin(tetaref - delrot + GeoParametersInput.teta_table * convrad + GeoParametersInput.teta_detec_para * convrad);

            I = 1;


            vector<vector<double>> eventsToTrace_para;
            vector<vector<double>> eventsToTrace_anti;

            
            while(I <= NumberRaysInput.nbeams){
                //Temporary event to show in the 3D view
                //If we have less than maxEventNum we just append otherwise we see
                //this temporary event stored reaches the exit before appending.
                //Each triple of values is 1 point and each event will have 4 points.
                //Source - Crystal1 - Crystal2 - Detector
                vector<double> tmpEvent;


                if(UserSettingsInput.Make_Horizontal){
                    p = del_teta_L * ((double)rand() / RAND_MAX) + teta_min_L;
                    tetadir = p;
                }else
                    tetadir = 0;


                sin_tetadir = sin(tetadir);
                cos_tetadir = cos(tetadir);
                tan_tetadir = sin_tetadir / cos_tetadir;
                cos_tetadirCry1 = cos(tetadir + GeoParametersInput.teta_table * convrad);


                if(UserSettingsInput.Make_Vertical){
                    p = del_fi_L * ((double)rand() / RAND_MAX) + fi_min_L;
                    fidir = p;
                }else
                    fidir = GeoParametersInput.xsi * convrad;


                sin_fidir = sin(fidir);
                cos_fidir = cos(fidir);
                tan_fidir = sin_fidir / cos_fidir;
                cos_fidirtilt = cos(fidir + tilt_C1_temp);


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
                    r = S_sour_2 + 1;
                    while(r > S_sour_2){
                        z = ((double)rand() / RAND_MAX) * GeolengthelementsInput.S_sour - S_sour_2;
                        y = ((double)rand() / RAND_MAX) * GeolengthelementsInput.S_sour - S_sour_2;
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
                    p = 2 * M_PI * ((double)rand() / RAND_MAX);
                    tetap = p;
                    sin_tetap = sin(tetap);
                    cos_tetap = cos(tetap);
                    r = S_aper_R_2 + 1;

                    while(r > S_aper_R_2)
                        r = Util::GaussianBox(S_aper_var_2, 0);

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
                    z = ((double)rand() / RAND_MAX) * GeolengthelementsInput.z_sour - z_sour_2;
                    y = ((double)rand() / RAND_MAX) * GeolengthelementsInput.y_sour - y_sour_2;

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

                vector<double> yz = Util::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, LT_aper_Db);

                y = yz[0];
                z = yz[1];

                var_temp = pow(y - GeolengthelementsInput.S_shi_hor_A, 2) + pow(z - GeolengthelementsInput.S_shi_ver_A, 2);

                if(var_temp < S_aper_sqr){

                    if(GraphOptionsInput.make_image_plates){
                        Util::Make(1, y, z);

                        //Event point at source
                        tmpEvent.push_back(0); //X
                        tmpEvent.push_back(y); //Y
                        tmpEvent.push_back(z); //Z
                        
                    }


                    r = sqrt(pow(y, 2) + pow(z, 2));

                    vector<double> yz = Util::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, LT_aper_Db);

                    y = yz[0];
                    z = yz[1];


                    y_pro_C1 = y * cos_tetadir / cos_tetadirCry1;
                    z_pro_C1 = z * cos_fidir / cos_fidirtilt;

                    //Deprecated???
                    //if(physical_parameters.make_length_corr){
                    //	sin_tetadirCry1 = sin(tetadir + GeoParametersInput.teta_table);
                    //	make_length_cor(z_pro_C1, y_pro_C1, sin_fidir, cos_fidir, sin_tetadirCry1, z_pro_C1, y_pro_C1);
                    //}

                    //logString.clear();
                    //logString << y_pro_C1 << "\t" << y_max_C1 << "\t" << y_min_C1 << "\t;\t" << z_pro_C1 << "\t" << z_max_C1 << "\t" << z_min_C1 << endl;
                    //emit w->LogLine(logString.str());

                    if(y_pro_C1 < y_max_C1 && y_pro_C1 > y_min_C1 && z_pro_C1 < z_max_C1 && z_pro_C1 > z_min_C1){
                        if(!GraphOptionsInput.make_imageC1_After_refle){
                            if(GraphOptionsInput.make_image_plates){
                                Util::Make(2, y_pro_C1, z_pro_C1);
                                
                                //Event point at the first crystal
                                tmpEvent.push_back(0); //X
                                tmpEvent.push_back(y_pro_C1); //Y
                                tmpEvent.push_back(z_pro_C1); //Z
                            }
                        }

                        rx = cos_fidir * cos_tetadir;
                        ry = cos_fidir * sin_tetadir;
                        rz = sin_fidir;


                        if(CurvedCrystalInput.Curve_crystall){
                            vector<double> corrRes = Util::horCorr(
                                y_pro_C1,
                                y_max_C1,
                                z_pro_C1,
                                z_max_C1,
                                true);

                            Costeta_CHC = corrRes[0];
                            Sinteta_CHC = corrRes[1];

                            n1x = n1x_temp * Costeta_CHC + n1y_temp * Sinteta_CHC;
                            n1y = n1y_temp * Costeta_CHC - n1x_temp * Sinteta_CHC;
                        }

                        vector<double> angleRes = Util::getFullAngle(rx, ry, rz, n1x, n1y, n1z);
                        angle = angleRes[0];
                        r2x = angleRes[1];
                        r2y = angleRes[2];
                        r2z = angleRes[3];


                        lamda = Util::getEnergy(a_lamds_uni, b_lamds_uni, tw_d1_para);

                        tetabra1 = asin(lamda / tw_d1_para);

                        //logString.clear();
                        //logString << angle << "\t" << tetabra1 << endl;
                        //emit w->LogLine(logString.str());

                        if(((double)rand() / RAND_MAX) < PolarizationParametersInput.relationP_S)
                            poliP = true;
                        else
                            poliP = false;


                        first_crystal_reach = Util::getReflection(
                            angle,
                            tetabra1,
                            lamda,
                            false,
                            poliP);


                        if(first_crystal_reach){


                            if(GraphOptionsInput.make_imageC1_After_refle){
                                if(GraphOptionsInput.make_image_plates){
                                    Util::Make(2, y_pro_C1, z_pro_C1);
                                }
                            }


                            rx_rot = cos_tetartab * r2x + sin_tetartab * r2y;
                            ry_rot = -sin_tetartab * r2x + cos_tetartab * r2y;

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
                            cos_fidirtilt2_para = cos(fidir + tilt_C2_para_temp);
                            cos_fidirtilt2_anti = cos(fidir + tilt_C2_anti_temp);

                            corr_dis = y_pro_C1 * cos_difteC1_Ta;

                            y_pro_C1 = -y_pro_C1 * sin_difteC1_Ta;


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

                            vector<double> yz = Util::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, dist_Cr1_Cr2_Db - corr_dis);

                            y = yz[0];
                            z = yz[1];


                            if(UserSettingsInput.see_para){
                                sin_tetatab_del_dir = sin(tetaref - delrot + tetadir);
                                y_pro_C1 = y * cos_tetadir / sin_tetatab_del_dir;
                                z_pro_C1 = z * cos_fidir / cos_fidirtilt2_para;

                                //Deprecated????
                                //if(make_length_corr)
                                //	make_length_cor(z_pro_C1, y_pro_C1, sin_fidir, cos_fidir, -sin_tetatab_del_dir, z_pro_C1, y_pro_C1);

                                //logString.clear();
                                //logString << y_pro_C1 << "\t" << y_max_C2 << "\t" << y_min_C2 << "\t;\t" << z_pro_C1 << "\t" << z_max_C2 << "\t" << z_min_C2 << endl;
                                //emit w->LogLine(logString.str());

                                if(y_pro_C1 < y_max_C2 && y_pro_C1 > y_min_C2 && z_pro_C1 < z_max_C2 && z_pro_C1 > z_min_C2){

                                    if(GraphOptionsInput.make_image_plates){
                                        if(!GraphOptionsInput.make_imageC2_After_refle){
                                            Util::Make(3, y_pro_C1, z_pro_C1);

                                            //Event point at second crystal in parallel
                                            tmpEvent.push_back(0); //X
                                            tmpEvent.push_back(y_pro_C1); //Y
                                            tmpEvent.push_back(z_pro_C1); //Z
                                        }
                                    }

                                    
                                    tetabra2 = asin(lamda / tw_d2_para);


                                    if(CurvedCrystalInput.Curve_crystall){
                                        vector<double> corrRes = Util::horCorr(
                                            y_pro_C1,
                                            y_max_C2,
                                            z_pro_C1,
                                            z_max_C2,
                                            false);

                                        Costeta_CHC = corrRes[0];
                                        Sinteta_CHC = corrRes[1];

                                        n2x_para = n2x_para_temp * Costeta_CHC + n2y_para_temp * Sinteta_CHC;
                                        n2y_para = n2y_para_temp * Costeta_CHC - n2x_para_temp * Sinteta_CHC;
                                    }


                                    vector<double> angleRes = Util::getFullAngle2(r2x, r2y, r2z, n2x_para, n2y_para, n2z_para);
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
                                                Util::Make(3, y_pro_C1, z_pro_C1);
                                            }
                                        }


                                        rx_rot_sec = cos_tetartabdete_para * r3x + sin_tetartabdete_para * r3y;
                                        ry_rot_sec = -sin_tetartabdete_para * r3x + cos_tetartabdete_para * r3y;

                                        var_temp = ry_rot_sec / rx_rot_sec;



                                        tetadir_det = atan(var_temp);
                                        tan_tetadir_det = var_temp;
                                        cos_tetadir_det = cos(tetadir_det);


                                        var_temp = rz / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                                        fidir_det = atan(var_temp);
                                        tan_fidir_det = var_temp;
                                        cos_fidir_det = cos(fidir_det);
                                        cos_fidirtilt2_para = cos(fidir + tilt_C2_para_temp);

                                        corr_dis_d_pa = -y_pro_C1 * cos_difteC2_det_para;
                                        y_pro_C1_d_pa = -y_pro_C1 * sin_difteC2_det_para;


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

                                        vector<double> yz = Util::getYZ(r_det, sin_tetap_det, cos_tetap_det, tan_tetadir_det, tan_fidir_det, dist_Cr2_det_Db - corr_dis_d_pa);

                                        y_det = yz[0];
                                        z_det = yz[1];

                                        if(y_det < ydetc_2_max && y_det > ydetc_2_min && z_det < zdetc_2_max && z_det > zdetc_2_min){

                                            if(GraphOptionsInput.make_image_plates){
                                                Util::Make(4, y_det, z_det);

                                                //Event point at detector in parallel
                                                tmpEvent.push_back(0); //X
                                                tmpEvent.push_back(y_det); //Y
                                                tmpEvent.push_back(z_det); //Z

                                                if (eventsToTrace_para.size() < NumberRaysInput.number_events) {
                                                    eventsToTrace_para.push_back(tmpEvent);
                                                }
                                                else {
                                                    eventsToTrace_para.erase(eventsToTrace_para.begin());
                                                    eventsToTrace_para.push_back(tmpEvent);
                                                }
                                            }

                                            toint_para++;

                                            energy_sum_para[numbins - 1] += Convert_Ag_minusone_eV / lamda;

                                        }
                                        else {
                                            if (GraphOptionsInput.make_image_plates) {
                                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                                if (eventsToTrace_para.size() < NumberRaysInput.number_events && tmpEvent.size() >= 6) {
                                                    eventsToTrace_para.push_back(tmpEvent);
                                                }
                                            }

                                        }

                                    }
                                    else {
                                        if (GraphOptionsInput.make_image_plates) {
                                            //If the event does not reach the detector then only add when we have less than maxEventNum
                                            if (eventsToTrace_para.size() < NumberRaysInput.number_events && tmpEvent.size() >= 6) {
                                                eventsToTrace_para.push_back(tmpEvent);
                                            }
                                        }
                                    }



                                }
                                else {
                                    if (GraphOptionsInput.make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        if (eventsToTrace_para.size() < NumberRaysInput.number_events && tmpEvent.size() >= 6) {
                                            eventsToTrace_para.push_back(tmpEvent);
                                        }
                                    }
                                }



                            }


                            if(UserSettingsInput.see_anti){


                                sin_tetatab_del_dir = sin(tetaref + delrot - tetadir);

                                y_pro_C1 = y * cos_tetadir / sin_tetatab_del_dir;
                                z_pro_C1 = z * cos_fidir / cos_fidirtilt2_anti;

                                //Deprecated?????
                                //if(make_length_corr)
                                //	make_length_cor(z_pro_C1, y_pro_C1, sin_fidir, cos_fidir, sin_tetatab_del_dir, z_pro_C1, y_pro_C1);

                                if(y_pro_C1 < y_max_C2 && y_pro_C1 > y_min_C2 && z_pro_C1 < z_max_C2 && z_pro_C1 > z_min_C2){


                                    if(GraphOptionsInput.make_imageC2_After_refle){
                                        if(GraphOptionsInput.make_image_plates){
                                            Util::Make(5, y_pro_C1, z_pro_C1);

                                            //Event point at second crystal in antiparallel
                                            tmpEvent.push_back(0); //X
                                            tmpEvent.push_back(y_pro_C1); //Y
                                            tmpEvent.push_back(z_pro_C1); //Z
                                        }
                                    }

                                    
                                    tetabra2 = asin(lamda / tw_d2_anti);

                                    if(CurvedCrystalInput.Curve_crystall){
                                        vector<double> corrRes = Util::horCorr(
                                            -y_pro_C1,
                                            y_max_C2,
                                            z_pro_C1,
                                            z_max_C2,
                                            false);

                                        Costeta_CHC = corrRes[0];
                                        Sinteta_CHC = corrRes[1];

                                        n2x_anti = n2x_anti_temp * Costeta_CHC + n2y_anti_temp * Sinteta_CHC;
                                        n2y_anti = n2y_anti_temp * Costeta_CHC - n2x_anti_temp * Sinteta_CHC;
                                    }


                                    vector<double> angleRes = Util::getFullAngle2(r2x, r2y, r2z, n2x_anti, n2y_anti, n2z_anti);
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
                                                Util::Make(5, y_pro_C1, z_pro_C1);
                                            }
                                        }


                                        rx_rot_sec = cos_tetartabdete_anti * r3x + sin_tetartabdete_anti * r3y;
                                        ry_rot_sec = - sin_tetartabdete_anti * r3x + cos_tetartabdete_anti * r3y;


                                        var_temp = - ry_rot_sec / rx_rot_sec;
                                        tetadir_det = atan(var_temp);
                                        tan_tetadir_det = var_temp;
                                        cos_tetadir_det = cos(tetadir_det);


                                        var_temp = rz / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                                        fidir_det = atan(var_temp);
                                        tan_fidir_det = var_temp;
                                        cos_fidir_det = cos(fidir_det);
                                        cos_fidirtilt2_anti = cos(fidir + tilt_C2_anti_temp);

                                        corr_dis_d_pa = y_pro_C1 * cos_difteC2_det_para;
                                        y_pro_C1_d_pa = y_pro_C1 * sin_difteC2_det_para;


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


                                        vector<double> yz = Util::getYZ(r_det, sin_tetap_det, cos_tetap_det, tan_tetadir_det, tan_fidir_det, dist_Cr2_det_Db - corr_dis_d_pa);

                                        y_det = yz[0];
                                        z_det = yz[1];

                                        if(y_det < ydetc_2_max && y_det > ydetc_2_min && z_det < zdetc_2_max && z_det > zdetc_2_min){

                                            if (GraphOptionsInput.make_image_plates) {
                                                Util::Make(6, y_det, z_det);

                                                //Event point at detector in antiparallel
                                                tmpEvent.push_back(0); //X
                                                tmpEvent.push_back(y_det); //Y
                                                tmpEvent.push_back(z_det); //Z

                                                if (eventsToTrace_anti.size() < NumberRaysInput.number_events) {
                                                    eventsToTrace_anti.push_back(tmpEvent);
                                                }
                                                else {
                                                    eventsToTrace_anti.erase(eventsToTrace_anti.begin());
                                                    eventsToTrace_anti.push_back(tmpEvent);
                                                }
                                            }

                                            toint_anti++;

                                            energy_sum_anti[numbins - 1] += Convert_Ag_minusone_eV / lamda;

                                        }
                                        else {
                                            if (GraphOptionsInput.make_image_plates) {
                                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                                if (eventsToTrace_anti.size() < NumberRaysInput.number_events && tmpEvent.size() >= 6) {
                                                    eventsToTrace_anti.push_back(tmpEvent);
                                                }
                                            }
                                        }

                                    }
                                    else {
                                        if (GraphOptionsInput.make_image_plates) {
                                            //If the event does not reach the detector then only add when we have less than maxEventNum
                                            if (eventsToTrace_anti.size() < NumberRaysInput.number_events && tmpEvent.size() >= 6) {
                                                eventsToTrace_anti.push_back(tmpEvent);
                                            }
                                        }
                                    }

                                }
                                else {
                                    if (GraphOptionsInput.make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        if (eventsToTrace_anti.size() < NumberRaysInput.number_events && tmpEvent.size() >= 6) {
                                            eventsToTrace_anti.push_back(tmpEvent);
                                        }
                                    }
                                }

                            }

                        }

                    }

                }

                I++;

                #ifdef QT_EXISTS
                    w->setPctDone(static_cast<float>(++total_current_bins) / total_expexted_bins);
                #endif
            }

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
                emit w->changeStats(
                    SimulationInterface::Stats
                    {
                        counts_sour,
                        counts_C1,
                        counts_C2_para,
                        counts_C2_anti,
                        counts_detc_para,
                        counts_detc_anti,
                        delrot,
                        eventsToTrace_para,
                        eventsToTrace_anti
                    }
                );
                }
            #endif


            toint_para_total[numbins - 1] += toint_para;
            total_para = toint_para_total[numbins - 1];

            toint_anti_total[numbins - 1] += toint_anti;
            total_anti = toint_anti_total[numbins - 1];


            angle_para = delrot * 180 / M_PI + teta_crys1;
            angle_anti = delrot * 180 / M_PI - teta_crys1;


            if(export_prof){
                hist_para << - angle_para << "\t" << toint_para << "\t" << sqrt((double)toint_para) << endl;
                hist_anti << - angle_anti << "\t" << toint_anti << "\t" << sqrt((double)toint_anti) << endl;
            }


            #ifdef QT_EXISTS
                if(GraphOptionsInput.make_graph_profile){
                    Make_plot_profiles::plotProfiles(
                        energy_sum_para[numbins - 1] / total_para,
                        angle_para,
                        total_para,
                        energy_sum_anti[numbins - 1] / total_anti,
                        angle_anti,
                        total_anti,
                        numbins,
                        w);
                }
            #endif


            if(n_rota == NumberRaysInput.number_rotati){
                if(UserSettingsInput.fitting){
                    Util::FitData(
                        numbins,
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


            if (numbins % 50 == 0) {
                int_time_out = Obtain_time::simuTime(
                    1,
                    (int)((PlotParametersInput.nubins - numbins) / 50),
                    int_time_out[0],
                    int_time_out[1],
                    w);
            }

            counts_C2_para_t += counts_C2_para;
            counts_detc_para_t += counts_detc_para;
            counts_C2_anti_t += counts_C2_anti;
            counts_detc_anti_t += counts_detc_anti;

            counts_C2_para = 0;
            counts_detc_para = 0;
            counts_detc_anti = 0;
            counts_C2_anti = 0;

            fill(&hist_image_plate_crystal2_para[0][0], &hist_image_plate_crystal2_para[0][0] + n_his_ima * n_his_ima, 0);
            fill(&hist_image_plate_crystal2_anti[0][0], &hist_image_plate_crystal2_anti[0][0] + n_his_ima * n_his_ima, 0);
            fill(&hist_image_plate_detc_para[0][0], &hist_image_plate_detc_para[0][0] + n_his_ima * n_his_ima, 0);
            fill(&hist_image_plate_detc_anti[0][0], &hist_image_plate_detc_anti[0][0] + n_his_ima * n_his_ima, 0);

            max_hist[2] = 0;
            max_hist[3] = 0;
            max_hist[4] = 0;
            max_hist[5] = 0;


            if(n_rota % 2 == 1){
                cond_rotation = numbins < PlotParametersInput.nubins;
                delrot = delrot - delrot_inc;
            }else{
                cond_rotation = numbins > 1;
                delrot = delrot + delrot_inc;
            }


        }

        n_rota++;

        #ifdef QT_EXISTS
            if(!w->isOpen())
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
        #endif
    }


    if(GraphOptionsInput.make_graph_profile){
        //TODO
        //gui stuff to implement
    }


    if(export_prof){
        gener_out << "********************************" << endl;
        gener_out << endl;
        gener_out << " Number of counts in the several geometric elements" << endl;
        gener_out << endl;
        gener_out << legen_counts[0] << "\t" << counts_sour << endl;
        gener_out << legen_counts[1] << "\t" << counts_C1 << endl;
        gener_out << legen_counts[2] << "\t" << counts_C2_para_t << endl;
        gener_out << legen_counts[3] << "\t" << counts_detc_para_t << endl;
        gener_out << legen_counts[4] << "\t" << counts_C2_anti_t << endl;
        gener_out << legen_counts[5] << "\t" << counts_detc_anti_t << endl;
    }

    return true;

}

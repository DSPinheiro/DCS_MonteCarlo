#pragma once

#ifndef SIMUCONSTS_H_
#define SIMUCONSTS_H_

#include <vector>
#include <string>
#include <fstream>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>


struct pick {
    double lamda;
    double natural_varia;
    double intensi;
};

struct energy_carac {
    double lamda;
    double intensity;
    double intensity_two_deriv;
};

struct energy_gen {
    double lamda;
    double intensity;
    double cum_int;
    double intensity_two_deriv;
    double lambda_two_deriv;
};

struct plotresponc {
    double degree;
    double reflecti_total;
    double reflecti_two_deriv;
    double reflecti_total_p;
    double reflecti_two_deriv_p;
    double reflecti_total_s;
    double reflecti_two_deriv_s;
};

struct plot {
    double energy;
    double x;
    double y;
    double error;
};

struct Geometry {
    bool mode_bragg_geo;
    int imh;
    int imk;
    int iml;
    bool crystal_Si;
};

struct UserSettings {
    bool see_para;
    bool see_anti;
    bool Make_Vertical;
    bool Make_Horizontal;
    int angle_aprox;
    bool fitting;
    bool TrueVoigt;
    bool Simple_simu;
    bool center_1crys;
    bool center_2crys;
    int mask_C1;
    int mask_C2;
    bool print_scan;
    bool center_Mask;
    bool make_mask_test;
};

struct GeoParapathlengths {
    std::string type_source;
    double LT_aper;
    double dist_T_Cr1;
    double dist_Cr1_Cr2;
    double dist_Cr2_Det;
    double dist_aper_det;
};

struct Geolengthelements {
    double S_aper;
    double S_aper_var;
    double S_sour;
    double y_sour;
    double z_sour;
    double y_aper;
    double z_aper;
    double S_shi_hor_B;
    double S_shi_hor_A;
    double S_shi_ver_B;
    double S_shi_ver_A;
    double y_first_crys;
    double z_first_crys;
    double ydetc;
    double zdetc;
    double shift_det_ver;
};

struct GeoParameters {
    double Exp_crys1;
    double teta_table;
    double OffsetRotCry1;
    double teta_detec_para;
    double teta_detec_anti;
    double tilt_C1;
    double tilt_C2;
    double xsi;
    double center_1cry_at;
    double center_2cry_at;
};

struct CurveVerticalTilt {
    bool make_CurveTilt;
    double phas_tilt1;
    double phas_tilt2;
    double offsettilt1;
    double offsettilt2;
    double consttilt1;
    double consttilt2;
};

struct Graph_options {
    bool make_graph_profile;
    bool make_image_plates;
    bool make_imageC1_After_refle;
    bool make_imageC2_After_refle;
};

struct plotparameters {
    double delta_angl;
    double shift_disp_window;
    int nubins;
};

struct numberrays {
    int nbeams;
    int number_rotati;
    int number_events;
};

struct physical_parameters {
    std::string Unit_energy;
    double linelamda;
    double naturalwidth;
    double gauss_Doop;
};

struct polarization_parameters {
    bool mka_poli;
    double relationP_S;
};

struct temperature_parameters {
    double T_crystal_1_para;
    double T_crystal_1_anti;
    double T_crystal_2_para;
    double T_crystal_2_anti;
    bool mk_temp_bin;
    double AA_tempera;
    double TT_tempera;
};

struct fullenergyspectrum {
    int make_more_lines;
    double linelamda1;
    double naturalwidth1;
    double p1_ener;
    double linelamda2;
    double naturalwidth2;
    double p2_ener;
    double linelamda3;
    double naturalwidth3;
    double p3_ener;
    double linelamda4;
    double naturalwidth4;
    bool Do_background;
    std::string energy_spectrum_file;
};

struct Curved_Crystal {
    bool Curve_crystall;
    double R_cur_crys_1;
    double R_cur_crys_2;
};

struct ParallelSettings {
    bool Make_GPU;
    int OMP_threads;
};

//TODO: UNIMPLEMENTED ANALYSIS FEATURE
// struct AnalysiesCrystaltilts {
    
// };


static inline std::vector<std::string> split(std::string s, std::string delimiter)
{
    size_t last = 0;
    size_t next = std::string::npos;

    std::vector<std::string> tokens;
    std::string token;

    while ((next = s.find(delimiter, last)) != std::string::npos)
    {
        token = s.substr(last, next - last);

        last = next + delimiter.length();

        tokens.push_back(token);
    }

    tokens.push_back(s.substr(last, next));

    return tokens;
}

// trim from start (in place)
static inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
        }));
}

// trim from end (in place)
static inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
        }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string& s) {
    ltrim(s);
    rtrim(s);
}

const double refra_corrNIST = 0.00351262;
const double refra_corrPARIS = 0.005952;


const double a_si_para = 5.431020457; // Paul Indelicato Value
const double a_Ge_para = 5.65735; // XOP
const double Convert_Ag_minusone_eV = 12398.41875;

const std::string evv[2] = { "eV ", "Ang" };

const int MA = 5;

const double convrad = M_PI / 180;
const double convdeg = 180 / M_PI;
const double one_micro = 1000000;

const double limitReflec = 0.001;

const int n_his_ima = 100;
const int n_his_g = 100;

const int fact_3 = 3 * 2;
const int fact_5 = 5 * 4 * 3 * 2;
const int fact_7 = 7 * 6 * 5 * 4 * 3 * 2;

const std::string legen_counts[6] = {
        "Number counts entrance:	",
        "Number counts C1:			",
        "Number counts C2_para:		",
        "Number counts detc_para:	",
        "Number counts C2_anti:		",
        "Number counts detc_anti:	"
};

const std::string legen_counts_1C[6] = {
        "Number counts entrance:	",
        "Number counts C1:			",
        "Number counts C2_para:		",
        "Number counts detc_para:	",
        "Number counts C2_anti:		",
        "Number counts detc_anti:	"
};

const double nm2 = n_his_ima / 2;
const double np2 = n_his_ima / 2;

const double const_back_para = 100, const_back_anti = 100;

const double width_Gaus_para = 0.009, width_Lore_para = 0.0009, width_Gaus_anti = 0.009, width_Lore_anti = 0.005;

const int do_amplitu_con_para = 1, do_amplitu_con_anti = 1, do_const_back_para = 1, do_const_back_anti = 1, do_firstcryst = 1, do_Gwidth_para = 1, do_Lwidth_para = 1, do_firstcryst_anti = 1, do_Gwidth_anti = 1, do_Lwidth_anti = 1;

const double c1 = 0.5346, c2 = 0.2166;

const double shape_corr = 0;

#endif /* SIMUCONSTS_H_ */

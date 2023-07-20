#pragma once

/*
 * simuGlobals.hh
 *
 *  Created on: Oct 29, 2020
 *      Author: danip
 */

#ifndef SIMUGLOBALS_HH_
#define SIMUGLOBALS_HH_

#include "simuConsts.h"
#ifdef QT_EXISTS
#include <QtWidgets/qplaintextedit.h>

extern QPlainTextEdit* logBox;
#endif

extern Geometry GeometryInput;
extern UserSettings UserSettingsInput;
extern GeoParapathlengths GeoParapathlengthsInput;
extern Geolengthelements GeolengthelementsInput;
extern GeoParameters GeoParametersInput;
extern CurveVerticalTilt CurveVerticalTiltInput;
extern Graph_options GraphOptionsInput;
extern plotparameters PlotParametersInput;
extern numberrays NumberRaysInput;
extern physical_parameters PhysicalParametersInput;
extern polarization_parameters PolarizationParametersInput;
extern temperature_parameters TemperatureParametersInput;
extern fullenergyspectrum FullEnergySpectrumInput;
extern Curved_Crystal CurvedCrystalInput;
extern ParallelSettings ParallelSettingsInput;
//TODO: UNIMPLEMENTED ANALYSIS FEATURE
//extern AnalysiesCrystaltilts AnalysiesCrystaltiltsInput;


extern double reques_energ[4];
extern double reques_width[4];

extern double linelamda, naturalwidth;

extern double linelambda1, naturalwidth1, linelambda2, naturalwidth2, linelambda3, naturalwidth3, linelambda4, naturalwidth4;

extern double request_width[4], request_energy[4];

extern pick picks[5];


extern std::ofstream gener_out;
extern std::ofstream hist_para;
extern std::ofstream hist_anti;


extern std::vector<plotresponc> Data_Fit_anti, Data_Fit_para;

extern std::vector<plot> plot_para, plot_anti;


extern double amplitu_con_para;
extern double peak_posi_para;
extern double amplitu_con_anti;
extern double peak_posi_anti;
extern double a_para_voig[MA];
extern double a_anti_voig[MA];
extern double vert_div_corr;
extern double refra_corr;
extern double middle_energy;
extern double d_lat;

extern double FWMH_V_para;
extern double FWMH_V_anti;

extern bool root_script;

extern double teta_crys1;

extern std::vector<energy_gen> Energy_spec;
extern energy_vecs Energy_Spectrum_Vectors;

extern char File_simu[1024];
extern char Output_dir[1024];


extern double Maxi_angl;
extern double Mini_angl;
extern double ang_para_pre;
extern double ang_anti_pre;


extern std::vector<double> available_energies;
extern std::vector<std::vector<plotresponc>> plotresponce_mide;
extern std::vector<plotresponc_vecs> Crystal_Responces;

extern double reflection_norm;


extern std::vector<double> min_angle_resp, max_angle_resp;


extern double max_plot_x[6];
extern double max_plot_y[6];
extern double max_hist[6];
extern double max_plot[2][2];
extern double min_plot[2][2];


extern double hist_image_plate_source[n_his_ima * n_his_ima];
extern double hist_image_plate_crystal1[n_his_ima * n_his_ima];
extern double hist_image_plate_crystal2_para[n_his_ima * n_his_ima];
extern double hist_image_plate_crystal2_anti[n_his_ima * n_his_ima];
extern double hist_image_plate_detc_para[n_his_ima * n_his_ima];
extern double hist_image_plate_detc_anti[n_his_ima * n_his_ima];




extern double theta_chk;

extern std::string Unit_energy;

extern bool never_set_angle;


extern double gauss_Doop_ev;

extern bool export_prof;

extern double ochisq;


#endif /* SIMUGLOBALS_HH_ */


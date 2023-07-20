#include "../include/simuGlobals.hh"


Geometry GeometryInput;
UserSettings UserSettingsInput;
GeoParapathlengths GeoParapathlengthsInput;
Geolengthelements GeolengthelementsInput;
GeoParameters GeoParametersInput;
CurveVerticalTilt CurveVerticalTiltInput;
Graph_options GraphOptionsInput;
plotparameters PlotParametersInput;
numberrays NumberRaysInput;
physical_parameters PhysicalParametersInput;
polarization_parameters PolarizationParametersInput;
temperature_parameters TemperatureParametersInput;
fullenergyspectrum FullEnergySpectrumInput;
Curved_Crystal CurvedCrystalInput;
ParallelSettings ParallelSettingsInput;
//TODO: UNIMPLEMENTED ANALYSIS FEATURE
//AnalysiesCrystaltilts AnalysiesCrystaltiltsInput;


double reques_energ[4];
double reques_width[4];

double linelamda, naturalwidth;

double linelambda1, naturalwidth1, linelambda2, naturalwidth2, linelambda3, naturalwidth3, linelambda4, naturalwidth4;

double request_width[4], request_energy[4];

pick picks[5];

#ifdef QT_EXISTS
QPlainTextEdit* logBox = nullptr;
#endif

std::ofstream gener_out;
std::ofstream hist_para;
std::ofstream hist_anti;


std::vector<plotresponc> Data_Fit_anti, Data_Fit_para;

std::vector<plot> plot_para, plot_anti;

double amplitu_con_para;
double peak_posi_para;
double amplitu_con_anti;
double peak_posi_anti;
double a_para_voig[MA];
double a_anti_voig[MA];
double vert_div_corr;
double refra_corr;
double middle_energy;
double d_lat;

double FWMH_V_para;
double FWMH_V_anti;

bool root_script;

double teta_crys1;

std::vector<energy_gen> Energy_spec;
energy_vecs Energy_Spectrum_Vectors;

char File_simu[1024];
char Output_dir[1024];


double Maxi_angl;
double Mini_angl;
double ang_para_pre;
double ang_anti_pre;


std::vector<double> available_energies;
std::vector<std::vector<plotresponc>> plotresponce_mide;
std::vector<plotresponc_vecs> Crystal_Responces;

double reflection_norm;


std::vector<double> min_angle_resp, max_angle_resp;


double max_plot_x[6];
double max_plot_y[6];
double max_hist[6];
double max_plot[2][2];
double min_plot[2][2];


double hist_image_plate_source[n_his_ima * n_his_ima];
double hist_image_plate_crystal1[n_his_ima * n_his_ima];
double hist_image_plate_crystal2_para[n_his_ima * n_his_ima];
double hist_image_plate_crystal2_anti[n_his_ima * n_his_ima];
double hist_image_plate_detc_para[n_his_ima * n_his_ima];
double hist_image_plate_detc_anti[n_his_ima * n_his_ima];


double theta_chk;

std::string Unit_energy;

bool never_set_angle;


double gauss_Doop_ev;

bool export_prof;

double ochisq = std::numeric_limits<double>::infinity();
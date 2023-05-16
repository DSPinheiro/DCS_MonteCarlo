/*
 * Double_Crystal_diffraction.cpp
 *
 *  Created on: Oct 30, 2020
 *      Author: danip
 */


#include "double_crystal_diffraction.hh"

using namespace std;

extern char Output_dir[200];
extern double teta_crys1;
extern pick picks[5];
extern vector<energy_gen> Energy_spec;

extern Geometry Geometry;
extern UserSettings UserSettings;
extern GeoParameters GeoParameters;
extern fullenergyspectrum fullenergyspectrum;
extern AnalysiesCrystaltilts AnalysiesCrystaltilts;
extern Geolengthelements Geolengthelements;
extern GeoParapathlengths GeoParapathlengths;
extern Graph_options Graph_options;
extern plotparameters plotparameters;
extern numberrays numberrays;
extern physical_parameters physical_parameters;
extern temperature_parameters temperature_parameters;

extern double d_lat, Maxi_angl, Mini_angl, ang_para_pre, ang_anti_pre;

ofstream hist_para;
ofstream hist_anti;
ofstream gener_out;

bool Make_Angle_brode, export_prof, rotate_C1 = false, never_set_angle = true;

double theta_chk;

double peak_posi_para, peak_posi_anti, max_plot[2][2] = {{0, 0}, {0, 0}}, min_plot[2][2] = {{0, 0}, {0, 0}};


void Double_Crystal_diffraction::Make_Simu(SimulationMain *w){

    double linelamda, naturalwidth, center_2cry_at_temp, sin_t, theta_b, line_ener, termFW;

    string paraPath = string(Output_dir) + "\\Histogram_parallel.txt";
    string antiPath = string(Output_dir) + "\\Histogram_antiparallel.txt";
    string generPath = string(Output_dir) + "\\general_output.txt";

    hist_para.open(paraPath, ofstream::out | ofstream::trunc);
    hist_anti.open(antiPath, ofstream::out | ofstream::trunc);
    gener_out.open(generPath, ofstream::out | ofstream::trunc);


    gener_out << 											endl;
    gener_out << "**************************************" << endl;
    gener_out << 											endl;
    gener_out << " Simulation done in Bragg geometry" << 	endl;
    gener_out << " for the reflected diffracted rays mode" << endl;
    gener_out << 											endl;
    gener_out << " crystal with crystallographic planes" << endl;
    gener_out << " parallel to surface" << 					endl;
    gener_out << 											endl;
    gener_out << "**************************************" << endl;
    gener_out << 											endl;

    if(UserSettings.Simple_simu){
        if(GeoParameters.Exp_crys1 < 0){
            throw runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be greater than 0. Exp_crys1 = 90 - tetabragg, tetabragg is the physical glancing angle of the first crystal to the x axis.");
        }

        teta_crys1 = GeoParameters.Exp_crys1;
    }else{
        if(GeoParameters.Exp_crys1 > 0){
            throw runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be less than 0. Exp_crys1 = - 90 - teta, teta is the physical angle of the table.");
        }

        teta_crys1 = - GeoParameters.teta_table - GeoParameters.Exp_crys1 + GeoParameters.OffsetRotCry1;
    }

    termFW = 2.0 * sqrt(2.0 * log(2.0));

    if(fullenergyspectrum.make_more_lines == 1){
        linelamda = picks[2].lamda;
        naturalwidth = picks[2].natural_varia;
    }else if(fullenergyspectrum.make_more_lines == 0){
        linelamda = picks[1].lamda;
        naturalwidth = picks[1].natural_varia;
    }else{
        //This is just to link the input energy spectrum with the rest of the simulation... not really usefull
        vector<double> inten;

        for(unsigned int i = 0; i < Energy_spec.size(); i++){
            inten.push_back(Energy_spec[i].intensity);
        }

        linelamda = Convert_Ag_minusone_eV / Energy_spec[FindLoc(inten, *max_element(inten.begin(), inten.end()))].lamda;
        naturalwidth = 0.0;
    }

    if(not UserSettings.see_para and not UserSettings.see_para){
        //dunno y this is commented... thats how it was in the original fortran code
        //UserSettings.fitting = false;
        //AnalysiesCrystaltilts.make_matrix_full = false;
        //AnalysiesCrystaltilts.make_graph_widths = false;
    }

    if(GeoParameters.center_1cry_at > Geolengthelements.z_first_crys / 2.0){
        throw runtime_error("The value for centering the spot on first crystal is higher then half of the crystal length");
    }

    if(UserSettings.center_1crys){
        Geolengthelements.S_shi_ver_A = (Geolengthelements.S_shi_ver_B - GeoParameters.center_1cry_at) * GeoParapathlengths.dist_T_Cr1 / (GeoParapathlengths.dist_T_Cr1 + GeoParapathlengths.LT_aper) + GeoParameters.center_1cry_at;
    }

    if(UserSettings.center_2crys){
        center_2cry_at_temp = (GeoParapathlengths.dist_T_Cr1 + GeoParapathlengths.dist_Cr1_Cr2) * (GeoParameters.center_1cry_at / GeoParapathlengths.dist_T_Cr1) - Geolengthelements.S_shi_ver_A * GeoParapathlengths.dist_Cr1_Cr2 / GeoParapathlengths.dist_T_Cr1;
        sin_t = sin(M_PI / 2.0 - teta_crys1 * M_PI / 180.0);

        cout << center_2cry_at_temp << "\t" << sin_t << "\t" << teta_crys1 << endl;
        GeoParameters.tilt_C1 = (GeoParameters.center_2cry_at - center_2cry_at_temp) * 180.0 / (2.0 * GeoParapathlengths.dist_Cr1_Cr2*sin_t) / M_PI;
        cout << GeoParameters.tilt_C1 << endl;
    }

    if(linelamda < 2 * d_lat){
        theta_b = asin(linelamda / (2 * d_lat)) * 180.0 / M_PI;
    }else{
        throw runtime_error("linelamda greater than 2 * d_lat, i.e. there is no diffraction: linelamda = " + to_string(linelamda) + "; d_lat = " + to_string(d_lat));
    }

    line_ener = Convert_Ag_minusone_eV / linelamda;
    theta_chk = 90 - theta_b;

    cout << "Theta bragg = " << theta_b << endl;
    cout << "Experimental first crystal angle = " << teta_crys1 << endl;
    cout << "Glancing angle for central ray = " << 90 - teta_crys1 << endl;
    cout << "theoretical = " << theta_chk << endl;

    theta_b = 2.0 * theta_b + teta_crys1 - 180.0;

    gener_out << endl;
    gener_out << "-------------------------------------------------------" << endl;
    gener_out << endl;
    gener_out << "	Input Parameters" << endl;
    gener_out << endl;
    gener_out << " Simulate Parallel: " << (UserSettings.see_para ? "True" : "False") << endl;
    gener_out << " Simulate Antiparallel: " << (UserSettings.see_anti ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << " Rays in planes: " << (not UserSettings.Make_Vertical ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << Out_angle(UserSettings.angle_aprox) << endl;
    gener_out << " Only valid in a simple simulation" << endl;
    gener_out << " otherwise the program uses without approximation" << endl;
    gener_out << endl;
    gener_out << " Make pos-simulation fitting: " << (UserSettings.fitting ? "True" : "False") << endl;
    gener_out << " with true Voigt: " << (UserSettings.TrueVoigt ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << " Make simple simulation: " << (UserSettings.Simple_simu ? "True" : "False") << endl;
    gener_out << " A more complete simulation takes account" << endl;
    gener_out << " table position and detectors position" << endl;
    gener_out << endl;
    gener_out << " Center the spot in the first crystal: " << (UserSettings.center_1crys ? "True" : "False") << endl;
    gener_out << " Only has effect in a complex simulation" << endl;
    gener_out << " The value of S_shi_ver_A is changed according to: " << endl;
    gener_out << "				S_shi_ver_B" << endl;
    gener_out << "				LT_aper" << endl;
    gener_out << "				dist_T_Cr1" << endl;
    gener_out << endl;
    gener_out << " Make the image plate after the diffraction" << endl;
    gener_out << " on first crystal: " << (Graph_options.make_imageC1_After_refle ? "True" : "False") << endl;
    gener_out << " and for second crystal: " << (Graph_options.make_imageC2_After_refle ? "True" : "False") << endl;
    gener_out << endl;

    if(GeoParapathlengths.type_source == "UC")
        gener_out << " This simulation is for an extended circular uniform source" << endl;
    else if(GeoParapathlengths.type_source == "UR")
        gener_out << " This simulation is for an extended rectangular uniform source" << endl;
    else if(GeoParapathlengths.type_source == "P")
        gener_out << " This simulation is for a point source" << endl;
    else if(GeoParapathlengths.type_source == "G")
        gener_out << " This simulation is for an extended gaussian source" << endl;
    else{
        cout << GeoParapathlengths.type_source << endl;
        throw runtime_error("Bad input for type_souce");
    }

    gener_out << " length of copper tube: " << GeoParapathlengths.LT_aper << "cm" << endl;
    gener_out << " Distance from copper tube to crystal 1: " << GeoParapathlengths.dist_T_Cr1 << "cm" << endl;
    gener_out << " Distance between Crystals: " << GeoParapathlengths.dist_Cr1_Cr2 << "cm" << endl;
    gener_out << " Distance from crystal 2 to detector: " << GeoParapathlengths.dist_Cr2_Det << "cm" << endl;
    gener_out << endl;
    gener_out << " Height of copper tube: " << Geolengthelements.S_aper << "cm" << endl;
    gener_out << " Width of crystal 1: " << Geolengthelements.y_first_crys << "cm" << endl;
    gener_out << " Height of crystal 1: " << Geolengthelements.z_first_crys << "cm" << endl;
    gener_out << " Width of detector: " << Geolengthelements.ydetc << "cm" << endl;
    gener_out << " Height of detector: " << Geolengthelements.zdetc << "cm" << endl;
    gener_out << endl;

    if(UserSettings.Simple_simu){
        gener_out << " Angle of crystal 1 normal vector to the x axis: " << teta_crys1 << "deg" << endl;
        gener_out << " Glancing angle for central ray: " << 90 - teta_crys1 << endl;
        gener_out << endl;
    }else{
        gener_out << " The angle of the first crystal in respect to the table: " << GeoParameters.Exp_crys1 << "deg" << endl;
        gener_out << " with an offset of: " << GeoParameters.OffsetRotCry1 << "deg" << endl;
        gener_out << " The table axis makes an angle to the x axis is: " << GeoParameters.teta_table << "deg" << endl;
        gener_out << " Angle of crystal 1 normal vector to the x axis is: " << teta_crys1 << "deg" << endl;
        gener_out << " Glancing angle for central ray: " << 90 - teta_crys1 << endl;
        gener_out << endl;
        gener_out << " Angle of the detector position to the table is in parallel:" << GeoParameters.teta_detec_para << "deg" << endl;
        gener_out << " In antiparallel: " << GeoParameters.teta_detec_anti << "deg" << endl;
        gener_out << endl;
    }

    gener_out << " Tilt on crystal 1:" << GeoParameters.tilt_C1 << "deg" << endl;
    gener_out << " Tilt on crystal 2:" << GeoParameters.tilt_C2 << "deg" << endl;
    gener_out << " Effective misalignement*:" << GeoParameters.xsi << "deg" << endl;
    gener_out << " *Only works on simple simulation" << GeoParameters.tilt_C1 << "deg" << endl;
    gener_out << endl;

    Make_Angle_brode = false;

    gener_out << endl;
    gener_out << " Profile plot scanning range: " << plotparameters.delta_angl << "deg" << endl;
    gener_out << " Profile plot display window shift: " << plotparameters.shift_disp_window << "deg" << endl;
    gener_out << " and has " << plotparameters.nubins << " bins" << endl;
    gener_out << "Make graphical plot: " << (Graph_options.make_graph_profile ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << "Number of rays used " << numberrays.nbeams << endl;
    gener_out << endl;

    if(fullenergyspectrum.make_more_lines == 0){
        gener_out << "Make more energies: No" << endl;
        gener_out << "Energy used in simulation: " << line_ener << evv[0] << endl;
        gener_out << "with corresponding wavelength: " << linelamda << evv[1] << endl;
        gener_out << "Natural width of: " << 2 * Convert_Ag_minusone_eV * naturalwidth / (pow(linelamda, 2) + pow(naturalwidth, 2)) << evv[0] << endl;
        gener_out << "with corresponding wavelength: " << 2 * naturalwidth << evv[1] << endl;
        gener_out << "Gaussian width FWHM (wavelength) = " << physical_parameters.gauss_Doop * termFW << " Ang" << endl;
        gener_out << "Gaussian width FWHM (energy) = " << physical_parameters.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    }else if(fullenergyspectrum.make_more_lines == 1){
        gener_out << "Make more energies: Yes" << endl;
        gener_out << "Central energy used in simulation: " << line_ener << evv[0] << endl;
        gener_out << "with corresponding wavelength: " << linelamda << evv[1] << endl;
        gener_out << "Natural width of: " << 2 * Convert_Ag_minusone_eV * naturalwidth / (pow(linelamda, 2) + pow(naturalwidth, 2)) << evv[0] << endl;
        gener_out << "with corresponding wavelength: " << 2 * naturalwidth << evv[1] << endl;
        gener_out << "Gaussian width FWHM (wavelength) = " << physical_parameters.gauss_Doop * termFW << " Ang" << endl;
        gener_out << "Gaussian width FWHM (energy) = " << physical_parameters.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    }else{
        gener_out << "Make more energies from file" << endl;
        gener_out << "Using energy spectrum from file Energy_spectrum_600.txt" << endl;
    }

    gener_out << "With a constant background: " << (fullenergyspectrum.Do_background ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << "Temperature of the Crystal 1: " << temperature_parameters.T_crystal_1_para << " ºC" << endl;
    gener_out << "Temperature of the Crystal 2 parallel: " << temperature_parameters.T_crystal_2_para << " ºC" << endl;
    gener_out << "Temperature of the Crystal 2 antiparallel: " << temperature_parameters.T_crystal_2_anti << " ºC" << endl;
    gener_out << endl;
    gener_out << "Make cycle over crystal tilts: " << (AnalysiesCrystaltilts.make_matrix_full ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << "Make graph of analysis of vertical misalignement: " << (AnalysiesCrystaltilts.make_graph_widths ? "True" : "False") << endl;

    if(AnalysiesCrystaltilts.make_graph_widths)
        gener_out << "Format of graph output: " << AnalysiesCrystaltilts.metafile << endl;

    gener_out << "-------------------------------------------------------" << endl;
    gener_out << endl;


    calculate_geo_corre::geo_corre();
    Test_input::test_In(theta_chk);

    Set_ang_rad::Set_angs();
    Obtain_curve_responce::Read_CurveResponce();

    cout << "Monte Carlo simulation of the double Crystal spectrometer" << endl;
    cout << "Parameters" << endl;

    if(fullenergyspectrum.make_more_lines <= 1){
        cout << "Line wavelength = " << linelamda << " Ang" << endl;
        cout << "Uncorrected Bragg angle (including position of parallel peak): " << - theta_b << endl;
        cout << "Line energy = " << line_ener << " eV" << endl;
        cout << "Line width (wavelength) = " << naturalwidth << " eV" << endl;
        cout << "Line width (energy) = " << naturalwidth * linelamda / line_ener << endl;
    }else{
        vector <double> ener;

        for(unsigned int i = 0; i < Energy_spec.size(); i++){
            ener.push_back(Energy_spec[i].lamda);
        }

        if(physical_parameters.Unit_energy == "keV"){
            cout << "Minimum energy loaded from file: " << *min_element(ener.begin(), ener.end()) << " eV" << endl;
            cout << "Maximum energy loaded from file: " << *max_element(ener.begin(), ener.end()) << " eV" << endl;
        }else{
            cout << "Minimum energy loaded from file: " << *min_element(ener.begin(), ener.end()) << physical_parameters.Unit_energy << endl;
            cout << "Maximum energy loaded from file: " << *max_element(ener.begin(), ener.end()) << physical_parameters.Unit_energy << endl;
        }
    }



    cout << "Crystal parameters: d = " << d_lat << "; Experimental first crystal angle = " << - teta_crys1 * 180 / M_PI << endl;
    cout << "Temperature first crystal = " << temperature_parameters.T_crystal_1_para << " ºC; Temperature second crystal = " << temperature_parameters.T_crystal_2_para << " ºC" << endl;
    cout << "Vertical tilt first crystal = " << GeoParameters.tilt_C1 << " deg; Vertical tilt second crystal = " << GeoParameters.tilt_C2 << " deg" << endl;

    hist_para << "# Line wavelength							= " << linelamda << " Ang" << endl;
    hist_para << "# Corresponding Bragg angle				= " << - theta_b << " Ang" << endl;
    hist_para << "# Line energy								= " << line_ener << " eV" << endl;
    hist_para << "# Natural line width (FWMH)				= " << 2 * naturalwidth * line_ener / linelamda << " eV" << endl;
    hist_para << "# Gaussian broadning (FWMH)				= " << physical_parameters.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    hist_para << "# Crystal lattice spacing: d				= " << d_lat << " Ang" << endl;

    hist_para << "# Crystal Miller indices					= " << Geometry.imh << ", " << Geometry.imk << ", " << Geometry.iml << " <- (h, k, l)" << endl;
    hist_para << "# Crystal Material						= " << (Geometry.crystal_Si ? "Si" : "Ge") << endl;
    hist_para << "# Diffraction order						= " << 1 << endl;
    hist_para << "# eV to Ang Conv. Constant				= " << Convert_Ag_minusone_eV << " eVAng" << endl;
    hist_para << "# Experimental first crystal angle		= " << -teta_crys1 * 180 / M_PI << " deg" << endl;


    hist_anti << "# Line wavelength							= " << linelamda << " Ang" << endl;
    hist_anti << "# Corresponding Bragg angle				= " << - theta_b << " Ang" << endl;
    hist_anti << "# Line energy								= " << line_ener << " eV" << endl;
    hist_anti << "# Natural line width (FWMH)				= " << 2 * naturalwidth * line_ener / linelamda << " eV" << endl;
    hist_anti << "# Gaussian broadning (FWMH)				= " << physical_parameters.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    hist_anti << "# Crystal lattice spacing: d				= " << d_lat << " Ang" << endl;

    hist_anti << "# Crystal Miller indices					= " << Geometry.imh << ", " << Geometry.imk << ", " << Geometry.iml << " <- (h, k, l)" << endl;
    hist_anti << "# Crystal Material						= " << (Geometry.crystal_Si ? "Si" : "Ge") << endl;
    hist_anti << "# Diffraction order						= " << 1 << endl;
    hist_anti << "# eV to Ang Conv. Constant				= " << Convert_Ag_minusone_eV << " eVAng" << endl;
    hist_anti << "# Experimental first crystal angle		= " << -teta_crys1 * 180 / M_PI << " deg" << endl;

    hist_para << "# Mini_angl								= " << -(Maxi_angl + teta_crys1 * 180 / M_PI) << " deg" << endl;
    hist_para << "# Maxi_angl								= " << -(Mini_angl + teta_crys1 * 180 / M_PI) << " Ang" << endl;
    hist_para << "# Position of parallel peak (approx.)		= " << -ang_para_pre << " deg" << endl;

    hist_para << "# Mini_angl								= " << -(Maxi_angl + teta_crys1 * 180 / M_PI) << " deg" << endl;
    hist_para << "# Maxi_angl								= " << -(Mini_angl + teta_crys1 * 180 / M_PI) << " Ang" << endl;
    hist_para << "# Position of parallel peak (approx.)		= " << -ang_anti_pre << " deg" << endl;


    hist_para << "# Temperature first crystal				= " << temperature_parameters.T_crystal_1_para << " ºC" << endl;
    hist_anti << "# Temperature first crystal				= " << temperature_parameters.T_crystal_1_para << " ºC" << endl;

    hist_para << "# Temperature second crystal				= " << temperature_parameters.T_crystal_2_para << " ºC" << endl;
    hist_anti << "# Temperature second crystal				= " << temperature_parameters.T_crystal_2_anti << " ºC" << endl;

    hist_para << "# Vertical tilt first crystal				= " << GeoParameters.tilt_C1 << " ºC" << endl;
    hist_para << "# Vertical tilt second crystal			= " << GeoParameters.tilt_C2 << " ºC" << endl;

    hist_anti << "# Vertical tilt first crystal				= " << GeoParameters.tilt_C1 << " ºC" << endl;
    hist_anti << "# Vertical tilt second crystal			= " << GeoParameters.tilt_C2 << " ºC" << endl;


    export_prof = true;

    if(UserSettings.Simple_simu)
        Source_simple::run_Source();
    else{
        Source_complex c;
        c.run_Source(w);
    }
    cout << "output of profiles in files" << endl;
    cout << "Histogram_antiparallel" << endl;
    cout << "Histogram_parallel" << endl;

    gener_out << "output of profiles in files" << endl;
    gener_out << "Histogram_antiparallel" << endl;
    gener_out << "Histogram_parallel" << endl;


    cout << "General output in file" << endl;
    cout << "general_output" << endl;
    cout << endl;

    if(UserSettings.fitting){
        if(UserSettings.see_para)
            MakefitVoigt::fit(true);
        if(UserSettings.see_anti)
            MakefitVoigt::fit(false);
        if(UserSettings.see_para and UserSettings.see_anti){}
            Analysie_Voigt::analyse();
    }

    if(UserSettings.make_mask_test)
        Mask_test_c2::test_c2();

    //old code? unimplemented because the required input is not used
    //if(rotate_C1)
        //Make_rotation_C1();

    //gui plotting related to the unimplemented c1 rotations
    //Make_plots_C1_table(theta_chk);

    //Make_plots_anna_wid();

}

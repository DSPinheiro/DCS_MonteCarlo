/*
 * Double_Crystal_diffraction.cpp
 *
 *  Created on: Oct 30, 2020
 *      Author: danip
 */


#include "../include/simu_starter.hh"
#include "../include/Util.h"

using namespace std;



bool Make_Angle_brode, rotate_C1 = false;


void Simu_Starter::Make_Simu(SimulationInterface* w){
    
    #ifdef QT_EXISTS
        w->setPctDone(0.0);
    #endif
    
    double center_2cry_at_temp, sin_t, theta_b, line_ener, termFW;

    stringstream logString;

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

    if(UserSettingsInput.Simple_simu){
        if(GeoParametersInput.Exp_crys1 < 0){
            throw runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be greater than 0. Exp_crys1 = 90 - tetabragg, tetabragg is the physical glancing angle of the first crystal to the x axis.");
        }

        teta_crys1 = GeoParametersInput.Exp_crys1;
    }else{
        if(GeoParametersInput.Exp_crys1 > 0){
            throw runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be less than 0. Exp_crys1 = - 90 - teta, teta is the physical angle of the table.");
        }

        teta_crys1 = -GeoParametersInput.teta_table - GeoParametersInput.Exp_crys1 + GeoParametersInput.OffsetRotCry1;
    }

    termFW = 2.0 * sqrt(2.0 * log(2.0));

    if(FullEnergySpectrumInput.make_more_lines == 1){
        linelamda = picks[2].lamda;
        naturalwidth = picks[2].natural_varia;
    }else if(FullEnergySpectrumInput.make_more_lines == 0){
        linelamda = picks[0].lamda;
        naturalwidth = picks[0].natural_varia;
    }else{
        //This is just to link the input energy spectrum with the rest of the simulation... not really usefull
        vector<double> inten;

        for(unsigned int i = 0; i < Energy_spec.size(); i++){
            inten.push_back(Energy_spec[i].intensity);
        }

        linelamda = Convert_Ag_minusone_eV / Energy_spec[Util::FindLoc(inten, *max_element(inten.begin(), inten.end()))].lamda;
        naturalwidth = 0.1;

        picks[0].lamda = linelamda;
        picks[0].natural_varia = naturalwidth;
    }

    if(GeoParametersInput.center_1cry_at > GeolengthelementsInput.z_first_crys / 2.0){
        throw runtime_error("The value for centering the spot on first crystal is higher then half of the crystal length");
    }

    if(UserSettingsInput.center_1crys){
        GeolengthelementsInput.S_shi_ver_A = (GeolengthelementsInput.S_shi_ver_B - GeoParametersInput.center_1cry_at) * GeoParapathlengthsInput.dist_T_Cr1 / (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.LT_aper) + GeoParametersInput.center_1cry_at;
    }

    if(UserSettingsInput.center_2crys){
        center_2cry_at_temp = (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2) * (GeoParametersInput.center_1cry_at / GeoParapathlengthsInput.dist_T_Cr1) - GeolengthelementsInput.S_shi_ver_A * GeoParapathlengthsInput.dist_Cr1_Cr2 / GeoParapathlengthsInput.dist_T_Cr1;
        sin_t = sin(M_PI / 2.0 - teta_crys1 * M_PI / 180.0);

        GeoParametersInput.tilt_C1 = (GeoParametersInput.center_2cry_at - center_2cry_at_temp) * 180.0 / (2.0 * GeoParapathlengthsInput.dist_Cr1_Cr2*sin_t) / M_PI;

        logString << center_2cry_at_temp << "\t" << sin_t << "\t" << teta_crys1 << endl;
        logString << GeoParametersInput.tilt_C1 << endl;
        
        #ifdef QT_EXISTS
            emit w->LogLineSignal(logString.str());
        #else
            cout << logString.str();
        #endif
    }

    if(linelamda < 2 * d_lat){
        theta_b = asin(linelamda / (2 * d_lat)) * 180.0 / M_PI;
    }else{
        throw runtime_error("linelamda greater than 2 * d_lat, i.e. there is no diffraction: linelamda = " + to_string(linelamda) + "; d_lat = " + to_string(d_lat));
    }

    line_ener = Convert_Ag_minusone_eV / linelamda;
    theta_chk = 90 - theta_b;
    
    logString.clear();
    logString << "Theta bragg = " << theta_b << endl;
    logString << "Experimental first crystal angle = " << teta_crys1 << endl;
    logString << "Glancing angle for central ray = " << 90 - teta_crys1 << endl;
    logString << "theoretical = " << theta_chk << endl;
    
    #ifdef QT_EXISTS
        emit w->LogLineSignal(logString.str());
    #else
        cout << logString.str();
    #endif

    theta_b = 2.0 * theta_b + teta_crys1 - 180.0;

    gener_out << endl;
    gener_out << "-------------------------------------------------------" << endl;
    gener_out << endl;
    gener_out << "	Input Parameters" << endl;
    gener_out << endl;
    gener_out << " Simulate Parallel: " << (UserSettingsInput.see_para ? "True" : "False") << endl;
    gener_out << " Simulate Antiparallel: " << (UserSettingsInput.see_anti ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << " Rays in planes: " << (!UserSettingsInput.Make_Vertical ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << Out_angle(UserSettingsInput.angle_aprox) << endl;
    gener_out << " Only valid in a simple simulation" << endl;
    gener_out << " otherwise the program uses without approximation" << endl;
    gener_out << endl;
    gener_out << " Make pos-simulation fitting: " << (UserSettingsInput.fitting ? "True" : "False") << endl;
    gener_out << " with true Voigt: " << (UserSettingsInput.TrueVoigt ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << " Make simple simulation: " << (UserSettingsInput.Simple_simu ? "True" : "False") << endl;
    gener_out << " A more complete simulation takes account" << endl;
    gener_out << " table position and detectors position" << endl;
    gener_out << endl;
    gener_out << " Center the spot in the first crystal: " << (UserSettingsInput.center_1crys ? "True" : "False") << endl;
    gener_out << " Only has effect in a complex simulation" << endl;
    gener_out << " The value of S_shi_ver_A is changed according to: " << endl;
    gener_out << "				S_shi_ver_B" << endl;
    gener_out << "				LT_aper" << endl;
    gener_out << "				dist_T_Cr1" << endl;
    gener_out << endl;
    gener_out << " Make the image plate after the diffraction" << endl;
    gener_out << " on first crystal: " << (GraphOptionsInput.make_imageC1_After_refle ? "True" : "False") << endl;
    gener_out << " and for second crystal: " << (GraphOptionsInput.make_imageC2_After_refle ? "True" : "False") << endl;
    gener_out << endl;

    if(GeoParapathlengthsInput.type_source == "UC")
        gener_out << " This simulation is for an extended circular uniform source" << endl;
    else if(GeoParapathlengthsInput.type_source == "UR")
        gener_out << " This simulation is for an extended rectangular uniform source" << endl;
    else if(GeoParapathlengthsInput.type_source == "P")
        gener_out << " This simulation is for a point source" << endl;
    else if(GeoParapathlengthsInput.type_source == "G")
        gener_out << " This simulation is for an extended gaussian source" << endl;
    else{
        logString.clear();
        logString << GeoParapathlengthsInput.type_source << endl;
        
        #ifdef QT_EXISTS
            emit w->LogLineSignal(logString.str());
        #else
            cout << logString.str();
        #endif
        
        throw runtime_error("Bad input for type_souce");
    }

    gener_out << " length of copper tube: " << GeoParapathlengthsInput.LT_aper << "cm" << endl;
    gener_out << " Distance from copper tube to crystal 1: " << GeoParapathlengthsInput.dist_T_Cr1 << "cm" << endl;
    gener_out << " Distance between Crystals: " << GeoParapathlengthsInput.dist_Cr1_Cr2 << "cm" << endl;
    gener_out << " Distance from crystal 2 to detector: " << GeoParapathlengthsInput.dist_Cr2_Det << "cm" << endl;
    gener_out << endl;
    gener_out << " Height of copper tube: " << GeolengthelementsInput.S_aper << "cm" << endl;
    gener_out << " Width of crystal 1: " << GeolengthelementsInput.y_first_crys << "cm" << endl;
    gener_out << " Height of crystal 1: " << GeolengthelementsInput.z_first_crys << "cm" << endl;
    gener_out << " Width of detector: " << GeolengthelementsInput.ydetc << "cm" << endl;
    gener_out << " Height of detector: " << GeolengthelementsInput.zdetc << "cm" << endl;
    gener_out << endl;

    if(UserSettingsInput.Simple_simu){
        gener_out << " Angle of crystal 1 normal vector to the x axis: " << teta_crys1 << "deg" << endl;
        gener_out << " Glancing angle for central ray: " << 90 - teta_crys1 << endl;
        gener_out << endl;
    }else{
        gener_out << " The angle of the first crystal in respect to the table: " << GeoParametersInput.Exp_crys1 << "deg" << endl;
        gener_out << " with an offset of: " << GeoParametersInput.OffsetRotCry1 << "deg" << endl;
        gener_out << " The table axis makes an angle to the x axis is: " << GeoParametersInput.teta_table << "deg" << endl;
        gener_out << " Angle of crystal 1 normal vector to the x axis is: " << teta_crys1 << "deg" << endl;
        gener_out << " Glancing angle for central ray: " << 90 - teta_crys1 << endl;
        gener_out << endl;
        gener_out << " Angle of the detector position to the table is in parallel:" << GeoParametersInput.teta_detec_para << "deg" << endl;
        gener_out << " In antiparallel: " << GeoParametersInput.teta_detec_anti << "deg" << endl;
        gener_out << endl;
    }

    gener_out << " Tilt on crystal 1:" << GeoParametersInput.tilt_C1 << "deg" << endl;
    gener_out << " Tilt on crystal 2:" << GeoParametersInput.tilt_C2 << "deg" << endl;
    gener_out << " Effective misalignement*:" << GeoParametersInput.xsi << "deg" << endl;
    gener_out << " *Only works on simple simulation" << GeoParametersInput.tilt_C1 << "deg" << endl;
    gener_out << endl;

    gener_out << "Using Interpolating Crystal Reflection Profiles from files in: " << GeoParametersInput.reflection_profiles_dir << endl;
    gener_out << endl;

    Make_Angle_brode = false;

    gener_out << endl;
    gener_out << " Profile plot scanning range: " << PlotParametersInput.delta_angl << "deg" << endl;
    gener_out << " Profile plot display window shift: " << PlotParametersInput.shift_disp_window << "deg" << endl;
    gener_out << " and has " << PlotParametersInput.nubins << " bins" << endl;
    gener_out << "Make graphical plot: " << (GraphOptionsInput.make_graph_profile ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << "Number of rays used " << NumberRaysInput.nbeams << endl;
    gener_out << endl;

    if(FullEnergySpectrumInput.make_more_lines == 0){
        gener_out << "Make more energies: No" << endl;
        gener_out << "Energy used in simulation: " << line_ener << evv_1 << endl;
        gener_out << "with corresponding wavelength: " << linelamda << evv_2 << endl;
        gener_out << "Natural width of: " << 2 * Convert_Ag_minusone_eV * naturalwidth / (pow(linelamda, 2) + pow(naturalwidth, 2)) << evv_1 << endl;
        gener_out << "with corresponding wavelength: " << 2 * naturalwidth << evv_2 << endl;
        gener_out << "Gaussian width FWHM (wavelength) = " << PhysicalParametersInput.gauss_Doop * termFW << " Ang" << endl;
        gener_out << "Gaussian width FWHM (energy) = " << PhysicalParametersInput.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    }else if(FullEnergySpectrumInput.make_more_lines == 1){
        gener_out << "Make more energies: Yes" << endl;
        gener_out << "Central energy used in simulation: " << line_ener << evv_1 << endl;
        gener_out << "with corresponding wavelength: " << linelamda << evv_2 << endl;
        gener_out << "Natural width of: " << 2 * Convert_Ag_minusone_eV * naturalwidth / (pow(linelamda, 2) + pow(naturalwidth, 2)) << evv_1 << endl;
        gener_out << "with corresponding wavelength: " << 2 * naturalwidth << evv_2 << endl;
        gener_out << "Gaussian width FWHM (wavelength) = " << PhysicalParametersInput.gauss_Doop * termFW << " Ang" << endl;
        gener_out << "Gaussian width FWHM (energy) = " << PhysicalParametersInput.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    }else{
        gener_out << "Make more energies from file" << endl;
        gener_out << "Using energy spectrum from file: " << FullEnergySpectrumInput.energy_spectrum_file << endl;
    }

    gener_out << "With a constant background: " << (FullEnergySpectrumInput.Do_background ? "True" : "False") << endl;
    gener_out << endl;
    gener_out << "Temperature of the Crystal 1: " << TemperatureParametersInput.T_crystal_1_para << " ºC" << endl;
    gener_out << "Temperature of the Crystal 2 parallel: " << TemperatureParametersInput.T_crystal_2_para << " ºC" << endl;
    gener_out << "Temperature of the Crystal 2 antiparallel: " << TemperatureParametersInput.T_crystal_2_anti << " ºC" << endl;
    gener_out << endl;
    gener_out << "Make cycle over crystal tilts: " << "UNIMPLEMENTED!!!" <<endl;
    gener_out << endl;
    gener_out << "Make graph of analysis of vertical misalignement: " << "UNIMPLEMENTED" << endl;

    //TODO: UNIMPLEMENTED ANALYSIS FEATURE

    gener_out << "-------------------------------------------------------" << endl;
    gener_out << endl;


    Util::geo_corre();

    Util::test_In();

    Util::Set_angs();
    Util::Read_CurveResponce(GeoParametersInput.reflection_profiles_dir);

    logString.clear();
    logString << "Monte Carlo simulation of the double Crystal spectrometer" << endl;
    logString << "Parameters" << endl;


    if(FullEnergySpectrumInput.make_more_lines <= 1){
        logString << "Line wavelength = " << linelamda << " Ang" << endl;
        logString << "Uncorrected Bragg angle (including position of parallel peak): " << - theta_b << endl;
        logString << "Line energy = " << line_ener << " eV" << endl;
        logString << "Line width (wavelength) = " << naturalwidth << " eV" << endl;
        logString << "Line width (energy) = " << naturalwidth * linelamda / line_ener << endl;

    }else{
        vector <double> ener;

        for(unsigned int i = 0; i < Energy_spec.size(); i++){
            ener.push_back(Energy_spec[i].lamda);
        }

        if(PhysicalParametersInput.Unit_energy == "keV"){
            logString << "Minimum energy loaded from file: " << *min_element(ener.begin(), ener.end()) << " eV" << endl;
            logString << "Maximum energy loaded from file: " << *max_element(ener.begin(), ener.end()) << " eV" << endl;

        }else{
            logString << "Minimum energy loaded from file: " << *min_element(ener.begin(), ener.end()) << PhysicalParametersInput.Unit_energy << endl;
            logString << "Maximum energy loaded from file: " << *max_element(ener.begin(), ener.end()) << PhysicalParametersInput.Unit_energy << endl;

        }
    }


    logString << "Crystal parameters: d = " << d_lat << "; Experimental first crystal angle = " << -teta_crys1 << endl;
    logString << "Temperature first crystal = " << TemperatureParametersInput.T_crystal_1_para << " ºC; Temperature second crystal = " << TemperatureParametersInput.T_crystal_2_para << " ºC" << endl;
    logString << "Vertical tilt first crystal = " << GeoParametersInput.tilt_C1 << " deg; Vertical tilt second crystal = " << GeoParametersInput.tilt_C2 << " deg" << endl;
    
    #ifdef QT_EXISTS
        emit w->LogLineSignal(logString.str());
    #else
        cout << logString.str();
    #endif


    hist_para << "# Line wavelength							= " << linelamda << " Ang" << endl;
    hist_para << "# Corresponding Bragg angle				= " << - theta_b << " Ang" << endl;
    hist_para << "# Line energy								= " << line_ener << " eV" << endl;
    hist_para << "# Natural line width (FWMH)				= " << 2 * naturalwidth * line_ener / linelamda << " eV" << endl;
    hist_para << "# Gaussian broadning (FWMH)				= " << PhysicalParametersInput.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    hist_para << "# Crystal lattice spacing: d				= " << d_lat << " Ang" << endl;
    
    hist_para << "# Crystal Miller indices					= " << GeometryInput.imh << ", " << GeometryInput.imk << ", " << GeometryInput.iml << " <- (h, k, l)" << endl;
    hist_para << "# Crystal Material						= " << (GeometryInput.crystal_Si ? "Si" : "Ge") << endl;
    hist_para << "# Diffraction order						= " << 1 << endl;
    hist_para << "# eV to Ang Conv. Constant				= " << Convert_Ag_minusone_eV << " eVAng" << endl;
    hist_para << "# Experimental first crystal angle		= " << -teta_crys1 << " deg" << endl;


    hist_anti << "# Line wavelength							= " << linelamda << " Ang" << endl;
    hist_anti << "# Corresponding Bragg angle				= " << - theta_b << " Ang" << endl;
    hist_anti << "# Line energy								= " << line_ener << " eV" << endl;
    hist_anti << "# Natural line width (FWMH)				= " << 2 * naturalwidth * line_ener / linelamda << " eV" << endl;
    hist_anti << "# Gaussian broadning (FWMH)				= " << PhysicalParametersInput.gauss_Doop * termFW * line_ener / linelamda << " eV" << endl;
    hist_anti << "# Crystal lattice spacing: d				= " << d_lat << " Ang" << endl;
    
    hist_anti << "# Crystal Miller indices					= " << GeometryInput.imh << ", " << GeometryInput.imk << ", " << GeometryInput.iml << " <- (h, k, l)" << endl;
    hist_anti << "# Crystal Material						= " << (GeometryInput.crystal_Si ? "Si" : "Ge") << endl;
    hist_anti << "# Diffraction order						= " << 1 << endl;
    hist_anti << "# eV to Ang Conv. Constant				= " << Convert_Ag_minusone_eV << " eVAng" << endl;
    hist_anti << "# Experimental first crystal angle		= " << -teta_crys1 << " deg" << endl;
    
    hist_para << "# Maxi_angl								= " << -(Maxi_angl + teta_crys1) << " deg" << endl;
    hist_para << "# Mini_angl								= " << -(Mini_angl + teta_crys1) << " Ang" << endl;
    hist_para << "# Position of parallel peak (approx.)		= " << -ang_para_pre << " deg" << endl;
    
    hist_para << "# Maxi_angl								= " << -(Maxi_angl + teta_crys1) << " deg" << endl;
    hist_para << "# Mini_angl								= " << -(Mini_angl + teta_crys1) << " Ang" << endl;
    hist_para << "# Position of antiparallel peak (approx.)		= " << -ang_anti_pre << " deg" << endl;
    
    
    hist_para << "# Temperature first crystal				= " << TemperatureParametersInput.T_crystal_1_para << " ºC" << endl;
    hist_anti << "# Temperature first crystal				= " << TemperatureParametersInput.T_crystal_1_para << " ºC" << endl;
    
    hist_para << "# Temperature second crystal				= " << TemperatureParametersInput.T_crystal_2_para << " ºC" << endl;
    hist_anti << "# Temperature second crystal				= " << TemperatureParametersInput.T_crystal_2_anti << " ºC" << endl;
    
    hist_para << "# Vertical tilt first crystal				= " << GeoParametersInput.tilt_C1 << " ºC" << endl;
    hist_para << "# Vertical tilt second crystal			= " << GeoParametersInput.tilt_C2 << " ºC" << endl;
    
    hist_anti << "# Vertical tilt first crystal				= " << GeoParametersInput.tilt_C1 << " ºC" << endl;
    hist_anti << "# Vertical tilt second crystal			= " << GeoParametersInput.tilt_C2 << " ºC" << endl;


    export_prof = true;
    bool finished = false;

    if (UserSettingsInput.Simple_simu) {
        Source_simple::run_Source();
    }
    else {
        Source_complex c;
        finished = c.run_Source(w);
    }

    if (finished)
    {
        logString.clear();
        logString << "output of profiles in files" << endl;
        logString << "Histogram_antiparallel" << endl;
        logString << "Histogram_parallel" << endl;


        gener_out << "output of profiles in files" << endl;
        gener_out << "Histogram_antiparallel" << endl;
        gener_out << "Histogram_parallel" << endl;


        logString << "General output in file" << endl;
        logString << "general_output" << endl;
        logString << endl;

        #ifdef QT_EXISTS
            emit w->LogLineSignal(logString.str());
        #else
            cout << logString.str();
        #endif

        if (UserSettingsInput.fitting) {
            // if (UserSettingsInput.see_para)
            //     Util::fit(true);
            // if (UserSettingsInput.see_anti)
            //     Util::fit(false);
            // if (UserSettingsInput.see_para && UserSettingsInput.see_anti) {}
            // Util::analyse();

            #ifdef QT_EXISTS
            emit w->LogLineSignal("Fitting analysis currently disabled!!!\n");
            #endif

            gener_out << endl;
            gener_out << "Fitting analysis currently disabled!!!" << endl;
            gener_out << endl;
        }

        if (UserSettingsInput.make_mask_test)
            Mask_test_c2::test_c2();

        //old code? unimplemented because the required input is not used
        //if(rotate_C1)
            //Make_rotation_C1();

        //gui plotting related to the unimplemented c1 rotations
        //Make_plots_C1_table(theta_chk);

        //Make_plots_anna_wid();
    }
}

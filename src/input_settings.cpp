#include "../include/input_settings.h"
#include <fstream>
#include <filesystem>
#include <iostream>

#ifdef QT_EXISTS
#include "ui_input_settings_prompt.h"
#include <QPushButton>
#include <QFileDialog>
#include <QMessageBox>
#endif

#ifdef OPENMP
#include <omp.h>
#include <thread>
#endif

int InputSettingsPrompt::configure(const std::string& inFile)
{
    // Parse the input file
    std::ifstream inputFile(inFile);
    std::string line;
    if(inputFile.is_open()){
        char str[1];
        char* firstChar = str;
        inputFile.read(firstChar, 1);
        inputFile.seekg(0);

        std::string currStruct = "";

        #ifdef QT_EXISTS

            QString message = "Path to simulation workspace: ";
            message.append(File_simu);
            QMessageBox::information(nullptr, message, tr("An input file with the experimental setup parameters has been found!"), QMessageBox::Ok);
        #else
            std::cout << "An input file with the C++ configuration model has been found." << std::endl;
        #endif

        while (getline(inputFile, line)){
            if (line.size() < 1) continue;
            if(line[0] != '/' && line[1] != '/'){

                if (line.find("//") != std::string::npos)
                    line = split(line, "//")[0];

                if(line.find("Geometry") != std::string::npos)
                    currStruct = "Geometry";
                else if(line.find("UserSettings") != std::string::npos)
                    currStruct = "UserSettings";
                else if(line.find("GeoParapathlenghts") != std::string::npos)
                    currStruct = "GeoParapathlenghts";
                else if(line.find("Geolenghtelemets") != std::string::npos)
                    currStruct = "Geolenghtelemets";
                else if(line.find("GeoParameters") != std::string::npos)
                    currStruct = "GeoParameters";
                else if(line.find("CurveVerticalTilt") != std::string::npos)
                    currStruct = "CurveVerticalTilt";
                else if(line.find("Graph_options") != std::string::npos)
                    currStruct = "Graph_options";
                else if(line.find("plotparameters") != std::string::npos)
                    currStruct = "plotparameters";
                else if(line.find("numberrays") != std::string::npos)
                    currStruct = "numberrays";
                else if(line.find("physical_parameters") != std::string::npos)
                    currStruct = "physical_parameters";
                else if(line.find("polarization_parameters") != std::string::npos)
                    currStruct = "polarization_parameters";
                else if(line.find("temperature_parameters") != std::string::npos)
                    currStruct = "temperature_parameters";
                else if(line.find("fullenergyspectrum") != std::string::npos)
                    currStruct = "fullenergyspectrum";
                else if(line.find("Curved_Crystal") != std::string::npos)
                    currStruct = "Curved_Crystal";
                else if(line.find("ParallelSettings") != std::string::npos)
                    currStruct = "ParallelSettings";
                else if(line.find("AnalysiesCrystaltilts") != std::string::npos)
                    currStruct = "AnalysiesCrystaltilts";



                if(currStruct == "Geometry"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "mode_bragg_geo"){
                        trim(items[1]);
                        GeometryInput.mode_bragg_geo = (items[1] == ".true.");
                    }else if(items[0] == "imh"){
                        trim(items[1]);
                        GeometryInput.imh = std::stoi(items[1]);
                    }else if(items[0] == "imk"){
                        trim(items[1]);
                        GeometryInput.imk = std::stoi(items[1]);
                    }else if(items[0] == "iml"){
                        trim(items[1]);
                        GeometryInput.iml = std::stoi(items[1]);
                    }else if(items[0] == "crystal_Si"){
                        trim(items[1]);
                        GeometryInput.crystal_Si = (items[1] == ".true.");
                    }

                }else if(currStruct == "UserSettings"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "see_para"){
                        trim(items[1]);
                        UserSettingsInput.see_para = (items[1] == ".true.");
                    }else if(items[0] == "see_anti"){
                        trim(items[1]);
                        UserSettingsInput.see_anti = (items[1] == ".true.");
                    }else if(items[0] == "Make_Vertical"){
                        trim(items[1]);
                        UserSettingsInput.Make_Vertical = (items[1] == ".true.");
                    }else if(items[0] == "Make_Horizontal"){
                        trim(items[1]);
                        UserSettingsInput.Make_Horizontal = (items[1] == ".true.");
                    }else if(items[0] == "angle_aprox"){
                        trim(items[1]);
                        UserSettingsInput.angle_aprox = std::stoi(items[1]);
                    }else if(items[0] == "fitting"){
                        trim(items[1]);
                        UserSettingsInput.fitting = (items[1] == ".true.");
                    }else if(items[0] == "TrueVoigt"){
                        trim(items[1]);
                        UserSettingsInput.TrueVoigt = (items[1] == ".true.");
                    }else if(items[0] == "Simple_simu"){
                        trim(items[1]);
                        UserSettingsInput.Simple_simu = (items[1] == ".true.");
                    }else if(items[0] == "center_1crys"){
                        trim(items[1]);
                        UserSettingsInput.center_1crys = (items[1] == ".true.");
                    }else if(items[0] == "center_2crys"){
                        trim(items[1]);
                        UserSettingsInput.center_2crys = (items[1] == ".true.");
                    }else if(items[0] == "mask_C1"){
                        trim(items[1]);
                        UserSettingsInput.mask_C1 = std::stoi(items[1]);
                    }else if(items[0] == "mask_C2"){
                        trim(items[1]);
                        UserSettingsInput.mask_C2 = std::stoi(items[1]);
                    }else if(items[0] == "print_scan"){
                        trim(items[1]);
                        UserSettingsInput.print_scan = (items[1] == ".true.");
                    }else if(items[0] == "center_Mask"){
                        trim(items[1]);
                        UserSettingsInput.center_Mask = (items[1] == ".true.");
                    }else if(items[0] == "make_mask_test"){
                        trim(items[1]);
                        UserSettingsInput.make_mask_test = (items[1] == ".true.");
                    }

                }else if(currStruct == "GeoParapathlenghts"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "type_source"){
                        trim(items[1]);
                        GeoParapathlengthsInput.type_source = split(items[1], "\"")[1];
                    }else if(items[0] == "LT_aper"){
                        trim(items[1]);
                        GeoParapathlengthsInput.LT_aper = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_T_Cr1"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_T_Cr1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_Cr1_Cr2"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_Cr1_Cr2 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_Cr2_Det"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_Cr2_Det = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "dist_aper_det"){
                        trim(items[1]);
                        GeoParapathlengthsInput.dist_aper_det = std::stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "Geolenghtelemets"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "S_aper"){
                        trim(items[1]);
                        GeolengthelementsInput.S_aper = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_aper_var"){
                        trim(items[1]);
                        GeolengthelementsInput.S_aper_var = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_sour"){
                        trim(items[1]);
                        GeolengthelementsInput.S_sour = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "y_sour"){
                        trim(items[1]);
                        GeolengthelementsInput.y_sour = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "z_sour"){
                        trim(items[1]);
                        GeolengthelementsInput.z_sour = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "y_aper"){
                        trim(items[1]);
                        GeolengthelementsInput.y_aper = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "z_aper"){
                        trim(items[1]);
                        GeolengthelementsInput.z_aper = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_hor_B"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_hor_B = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_hor_A"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_hor_A = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_ver_B"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_ver_B = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_ver_A"){
                        trim(items[1]);
                        GeolengthelementsInput.S_shi_ver_A = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "y_first_crys"){
                        trim(items[1]);
                        GeolengthelementsInput.y_first_crys = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "z_first_crys"){
                        trim(items[1]);
                        GeolengthelementsInput.z_first_crys = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "ydetc"){
                        trim(items[1]);
                        GeolengthelementsInput.ydetc = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "zdetc"){
                        trim(items[1]);
                        GeolengthelementsInput.zdetc = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "shift_det_ver"){
                        trim(items[1]);
                        GeolengthelementsInput.shift_det_ver = std::stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "GeoParameters"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Exp_crys1"){
                        trim(items[1]);
                        GeoParametersInput.Exp_crys1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "teta_table"){
                        trim(items[1]);
                        GeoParametersInput.teta_table = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "OffsetRotCry1"){
                        trim(items[1]);
                        GeoParametersInput.OffsetRotCry1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "teta_detec_para"){
                        trim(items[1]);
                        GeoParametersInput.teta_detec_para = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "teta_detec_anti"){
                        trim(items[1]);
                        GeoParametersInput.teta_detec_anti = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "tilt_C1"){
                        trim(items[1]);
                        GeoParametersInput.tilt_C1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "tilt_C2"){
                        trim(items[1]);
                        GeoParametersInput.tilt_C2 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "xsi"){
                        trim(items[1]);
                        GeoParametersInput.xsi = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "center_1cry_at"){
                        trim(items[1]);
                        GeoParametersInput.center_1cry_at = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "S_shi_ver_B"){
                        trim(items[1]);
                        GeoParametersInput.center_2cry_at = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "make_table_angleNoise"){
                        trim(items[1]);
                        GeoParametersInput.make_table_noise = (items[1] == ".true.");
                    }else if(items[0] == "table_resolution"){
                        trim(items[1]);
                        GeoParametersInput.table_resolution = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "crystal_profile_dir"){
                        trim(items[1]);
                        GeoParametersInput.reflection_profiles_dir = items[1];
                    }

                }else if(currStruct == "CurveVerticalTilt"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "make_CurveTilt"){
                        trim(items[1]);
                        CurveVerticalTiltInput.make_CurveTilt = (items[1] == ".true.");
                    }else if(items[0] == "phas_tilt1"){
                        trim(items[1]);
                        CurveVerticalTiltInput.phas_tilt1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "phas_tilt2"){
                        trim(items[1]);
                        CurveVerticalTiltInput.phas_tilt2 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "offsettilt1"){
                        trim(items[1]);
                        CurveVerticalTiltInput.offsettilt1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "offsettilt2"){
                        trim(items[1]);
                        CurveVerticalTiltInput.offsettilt2 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "consttilt1"){
                        trim(items[1]);
                        CurveVerticalTiltInput.consttilt1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "consttilt2"){
                        trim(items[1]);
                        CurveVerticalTiltInput.consttilt2 = std::stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "Graph_options"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "make_graph_profile"){
                        trim(items[1]);
                        GraphOptionsInput.make_graph_profile = (items[1] == ".true.");
                    }else if(items[0] == "make_image_plates"){
                        trim(items[1]);
                        GraphOptionsInput.make_image_plates = (items[1] == ".true.");
                    }else if(items[0] == "make_imageC1_After_refle"){
                        trim(items[1]);
                        GraphOptionsInput.make_imageC1_After_refle = (items[1] == ".true.");
                    }else if(items[0] == "make_imageC2_After_refle"){
                        trim(items[1]);
                        GraphOptionsInput.make_imageC2_After_refle = (items[1] == ".true.");
                    }

                }else if(currStruct == "plotparameters"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "delta_angl"){
                        trim(items[1]);
                        PlotParametersInput.delta_angl = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "shift_disp_window"){
                        trim(items[1]);
                        PlotParametersInput.shift_disp_window = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "nubins"){
                        trim(items[1]);
                        PlotParametersInput.nubins = std::stoi(items[1]);
                    }

                }else if(currStruct == "numberrays"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "nbeams"){
                        trim(items[1]);
                        NumberRaysInput.nbeams = std::stoi(items[1]);
                    }else if(items[0] == "number_rotati"){
                        trim(items[1]);
                        NumberRaysInput.number_rotati = std::stoi(items[1]);
                    }else if (items[0] == "number_graph_events") {
                        trim(items[1]);
                        NumberRaysInput.number_events = std::stoi(items[1]);
                    }

                }else if(currStruct == "physical_parameters"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Unit_energy"){
                        trim(items[1]);
                        PhysicalParametersInput.Unit_energy = split(items[1], "\"")[1];
                    }else if(items[0] == "linelamda"){
                        trim(items[1]);
                        PhysicalParametersInput.linelamda = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth"){
                        trim(items[1]);
                        PhysicalParametersInput.naturalwidth = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "gauss_Doop"){
                        trim(items[1]);
                        PhysicalParametersInput.gauss_Doop = std::stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "polarization_parameters"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "mka_poli"){
                        trim(items[1]);
                        PolarizationParametersInput.mka_poli = (items[1] == ".true.");
                    }else if(items[0] == "relationP_S"){
                        trim(items[1]);
                        PolarizationParametersInput.relationP_S = std::stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "temperature_parameters"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "T_crystal_1_para"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_1_para = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "T_crystal_1_anti"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_1_anti = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "T_crystal_2_para"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_2_para = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "T_crystal_2_anti"){
                        trim(items[1]);
                        TemperatureParametersInput.T_crystal_2_anti = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "mk_temp_bin"){
                        trim(items[1]);
                        TemperatureParametersInput.mk_temp_bin = (items[1] == ".true.");
                    }else if(items[0] == "AA_tempera"){
                        trim(items[1]);
                        TemperatureParametersInput.AA_tempera = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "TT_tempera"){
                        trim(items[1]);
                        TemperatureParametersInput.TT_tempera = std::stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "fullenergyspectrum"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "make_more_lines"){
                        trim(items[1]);
                        FullEnergySpectrumInput.make_more_lines = std::stoi(items[1]);
                    }else if(items[0] == "linelamda1"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth1"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "p1_ener"){
                        trim(items[1]);
                        FullEnergySpectrumInput.p1_ener = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "linelamda2"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda2 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth2"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth2 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "p2_ener"){
                        trim(items[1]);
                        FullEnergySpectrumInput.p2_ener = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "linelamda3"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda3 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth3"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth3 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "p3_ener"){
                        trim(items[1]);
                        FullEnergySpectrumInput.p3_ener = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "linelamda4"){
                        trim(items[1]);
                        FullEnergySpectrumInput.linelamda4 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "naturalwidth4"){
                        trim(items[1]);
                        FullEnergySpectrumInput.naturalwidth4 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "Do_background"){
                        trim(items[1]);
                        FullEnergySpectrumInput.Do_background = (items[1] == ".true.");
                    }else if(items[0] == "spectrum_filename"){
                        trim(items[1]);
                        FullEnergySpectrumInput.energy_spectrum_file = items[1];
                    }

                }else if(currStruct == "Curved_Crystal"){

                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Curve_crystall"){
                        trim(items[1]);
                        CurvedCrystalInput.Curve_crystall = (items[1] == ".true.");
                    }else if(items[0] == "R_cur_crys_1"){
                        trim(items[1]);
                        CurvedCrystalInput.R_cur_crys_1 = std::stod(split(items[1], "d0")[0]);
                    }else if(items[0] == "R_cur_crys_2"){
                        trim(items[1]);
                        CurvedCrystalInput.R_cur_crys_2 = std::stod(split(items[1], "d0")[0]);
                    }

                }else if(currStruct == "ParallelSettings"){
                    
                    
                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    if(items[0] == "Make_GPU"){
                        trim(items[1]);
                        ParallelSettingsInput.Make_GPU = (items[1] == ".true.");
                    }else if(items[0] == "OMP_threads"){
                        trim(items[1]);
                        ParallelSettingsInput.OMP_threads = std::stoi(items[1]);
                    }

                }else if(currStruct == "AnalysiesCrystaltilts"){
                    
                    
                    std::string elem = split(line, "//")[0];
                    std::vector<std::string> items = split(elem, "=");

                    trim(items[0]);

                    //TODO: UNIMPLEMENTED ANALYSIS FEATURE

                }
            }
        }
    }else{
    #ifdef QT_EXISTS
        QString message = "Could not open input file: ";
        message.append(QString::fromUtf8(inFile.c_str()));

        QMessageBox msgBox;
        msgBox.setText(message);
        msgBox.setInformativeText("Please check if the file still exists or has been corruped.");
        msgBox.setStandardButtons(QMessageBox::Close);
        msgBox.setDefaultButton(QMessageBox::Close);
        int ret = msgBox.exec();

        return ret;
    #else
        std::cout << "Could not open input file: " << inFile << std::endl;
    #endif
    }
    
    inputFile.close();
    //std::cout << "Input file read." << std::endl;

    // Make aditional configurations
    Unit_energy = PhysicalParametersInput.Unit_energy;
    if(GeometryInput.imh == 2 and GeometryInput.imk == 2 and GeometryInput.iml == 2)
    {
        refra_corr = refra_corrNIST;
    }
    else
    {
        refra_corr = refra_corrPARIS;
    }

    #ifdef OPENMP
        ParallelSettingsInput.system_max_threads = std::thread::hardware_concurrency();;

        if(ParallelSettingsInput.OMP_threads > ParallelSettingsInput.system_max_threads)
        {
            ParallelSettingsInput.OMP_threads = ParallelSettingsInput.system_max_threads;

            #ifdef QT_EXISTS
            QString message = "The user specified number of threads to use in the simulation is more than the system's logic processor count: ";
            message.append(std::to_string(ParallelSettingsInput.OMP_threads).c_str());
            message.append("\nSetting this number to the logic processor count.");
            QMessageBox::warning(nullptr, tr("Too Many OpenMP Threads!"), message, QMessageBox::Ok);
            #else
            std::cout << "The user specified number of threads to use in the simulation is more than the system's logic processor count: " << ParallelSettingsInput.OMP_threads << " / " << ParallelSettingsInput.system_max_threads << std::endl;
            #endif
        }
        else
        {
            #ifdef QT_EXISTS
            QString message = "Starting the simulation with the maximum thread count: ";
            message.append(std::to_string(ParallelSettingsInput.OMP_threads).c_str());
            message.append(" out of ");
            message.append(std::to_string(ParallelSettingsInput.system_max_threads).c_str());
            message.append(" maximum threads.");
            QMessageBox::information(nullptr, tr("OpenMP Maximum Thread Count"), message, QMessageBox::Ok);
            #else
            std::cout << "Starting the simulation with the maximum thread count: " << ParallelSettingsInput.OMP_threads << " / " << ParallelSettingsInput.system_max_threads << std::endl;
            #endif
        }
    #endif

    return 0; // Dirty workaround
}

#ifdef QT_EXISTS
InputSettingsPrompt::InputSettingsPrompt(GUISettingsWindow* simWindow, QWidget *parent) : ui(new Ui::InputSettingsPrompt), simWindow(simWindow)
{
    ui->setupUi(this);
    (void)connect(ui->pushButton, &QPushButton::clicked, this, &InputSettingsPrompt::loadInputConfig);

    connect(ui->pushButton_2, &QPushButton::clicked, this, [this]() {
        QString dirname = QFileDialog::getExistingDirectory(this,
            tr("Open Workspace Directory"),
            QString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
        );

        if(dirname.isEmpty())
        {
            return;
        }

        ui->lineEdit->setText(dirname + "/");
    });

    connect(ui->pushButton_3, &QPushButton::clicked, this, [this]() {
        QString dirname = QFileDialog::getExistingDirectory(this,
            tr("Open Output Directory"),
            QString(),
            QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
        );

        if(dirname.isEmpty())
        {
            return;
        }

        ui->lineEdit_3->setText(dirname + "/");
    });
}

void InputSettingsPrompt::loadInputConfig()
{
    char inFile[200];
    std::string input_dir_path = ui->lineEdit->text().toStdString();
    std::string input_config_file_path = ui->lineEdit_2->text().toStdString();
    std::string output_dir_path = ui->lineEdit_3->text().toStdString();

    // Setup workspace dir
    if(!input_dir_path.empty())
    {
        strcpy(File_simu, input_dir_path.c_str());
    }
    else // Else try and use the default workspace name
    {
        strcpy(File_simu, ui->lineEdit->placeholderText().toStdString().c_str());
    }
    //std::cout << "Path to simulation workspace: " << File_simu << "\n\n" << std::endl;

    // Setup input config file name
    strcpy(inFile, File_simu);
    if(!input_config_file_path.empty())
    {
        strcat(inFile, "/");
        strcat(inFile, input_config_file_path.c_str());
    }
    else // Else try and use the default filename
    {
        strcat(inFile, ui->lineEdit_2->placeholderText().toStdString().c_str());
    }

    // Configure the output directory
    if(!output_dir_path.empty())
    {
        
        if(std::filesystem::path(output_dir_path).is_relative())
        {
            strcat(Output_dir, File_simu);
            strcat(Output_dir, "/");
            strcat(Output_dir, output_dir_path.c_str());
        }
        else // Just use a path relative to the executable's location
        {
            strcat(Output_dir, output_dir_path.c_str());
        }
    }
    else
    {
        strcat(Output_dir, ui->lineEdit_3->placeholderText().toStdString().c_str());
    }

    if(!std::filesystem::is_directory(Output_dir) || !std::filesystem::exists(Output_dir))
    {
        std::filesystem::create_directories(Output_dir);
    }

    if(configure(inFile) == 0)
    {
        simWindow->setup();
        simWindow->updateElements();
        simWindow->show();
        close();
    }
    else
    {
        QCoreApplication::exit(1);
    }
}
#endif

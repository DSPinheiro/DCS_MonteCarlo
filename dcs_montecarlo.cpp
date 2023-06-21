//#ifndef LIB_DEF
//#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")
//#endif

//============================================================================
// Name        : DCS_MonteCarlo.cpp
// Author      : Daniel Pinheiro, Pedro Amaro, César Godinho
// Version     : 2.0
// Copyright   : Your copyright notice
// Description : Entry point for the DCS simulation
//============================================================================

#include "dcs_montecarlo.hh"
#include <Util.h>
#include <QtWidgets/qmessagebox.h>

#include "configure.inl"

#include <filesystem>


using namespace std;

char inFile[200];

static int configure(int argc, char* argv[])
{
    CommandLineParams input_params = command_line_parser(argc, argv);
    if(!input_params.valid) return 0;

    // Setup workspace dir
    if(!input_params.input_dir_path.empty())
    {
        strcpy(File_simu, input_params.input_dir_path.c_str());
    }
    else // Else try and use the default workspace name
    {
        strcpy(File_simu, "simuWorkspace");
    }
    cout << "Path to simulation workspace: " << File_simu << "\n\n" << endl;

    // Setup input config file name
    strcat(inFile, File_simu);
    if(!input_params.input_config_file_path.empty())
    {
        strcat(inFile, "/");
        strcat(inFile, input_params.input_config_file_path.c_str());
    }
    else // Else try and use the default filename
    {
        strcat(inFile, "/DCrystal_input.input");
    }

    // Configure the output directory
    if(!input_params.output_dir_path.empty())
    {
        if(input_params.use_rel_output)
        {
            strcat(Output_dir, File_simu);
            strcat(Output_dir, "/");
            strcat(Output_dir, input_params.output_dir_path.c_str());
        }
        else // Just use a path relative to the executable's location
        {
            strcat(Output_dir, input_params.output_dir_path.c_str());
        }
    }
    else
    {
        strcat(Output_dir, "output");
    }

    if(!filesystem::is_directory(Output_dir) || !filesystem::exists(Output_dir))
    {
        filesystem::create_directories(Output_dir);
    }

    // Parse the input file
    ifstream inputFile(inFile);
    string line;
    if(inputFile.is_open()){
        char str[1];
        char* firstChar = str;
        inputFile.read(firstChar, 1);
        inputFile.seekg(0);

        string currStruct = "";

        if(strcmp(firstChar, "&") == 181){
            QString message = "Path to simulation workspace: ";
            message.append(File_simu);

            QMessageBox msgBox;
            msgBox.setText(message);
            msgBox.setInformativeText("An input file with the FORTRAN configuration model has been found.\nWould you like to continue?");
            msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
            msgBox.setDefaultButton(QMessageBox::Yes);
            int ret = msgBox.exec();


            if (ret == QMessageBox::No) {
                return ret;
            }


            while (getline(inputFile, line)){
                if(line.find("&Geometry") != string::npos)
                    currStruct = "&Geometry";
                else if(line.find("&UserSettings") != string::npos)
                    currStruct = "&UserSettings";
                else if(line.find("&GeoParapathlenghts") != string::npos)
                    currStruct = "&GeoParapathlenghts";
                else if(line.find("&Geolenghtelemets") != string::npos)
                    currStruct = "&Geolenghtelemets";
                else if(line.find("&GeoParameters") != string::npos)
                    currStruct = "&GeoParameters";
                else if(line.find("&CurveVerticalTilt") != string::npos)
                    currStruct = "&CurveVerticalTilt";
                else if(line.find("&Graph_options") != string::npos)
                    currStruct = "&Graph_options";
                else if(line.find("&plotparameters") != string::npos)
                    currStruct = "&plotparameters";
                else if(line.find("&numberrays") != string::npos)
                    currStruct = "&numberrays";
                else if(line.find("&physical_parameters") != string::npos)
                    currStruct = "&physical_parameters";
                else if(line.find("&polarization_parameters") != string::npos)
                    currStruct = "&polarization_parameters";
                else if(line.find("&temperature_parameters") != string::npos)
                    currStruct = "&temperature_parameters";
                else if(line.find("&fullenergyspectrum") != string::npos)
                    currStruct = "&fullenergyspectrum";
                else if(line.find("&Curved_Crystal") != string::npos)
                    currStruct = "&Curved_Crystal";
                else if(line.find("&AnalysiesCrystaltilts") != string::npos)
                    currStruct = "&AnalysiesCrystaltilts";



                if(currStruct == "&Geometry" && line.find("&end") == string::npos){

                    if(line.find("&Geometry") != string::npos){

                        for(string elem : split(split(line, "&Geometry ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mode_bragg_geo")
                                GeometryInput.mode_bragg_geo = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "imh")
                                GeometryInput.imh = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "imk")
                                GeometryInput.imk = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "iml")
                                GeometryInput.iml = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "crystal_Si")
                                GeometryInput.crystal_Si = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mode_bragg_geo")
                                GeometryInput.mode_bragg_geo = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "imh")
                                GeometryInput.imh = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "imk")
                                GeometryInput.imk = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "iml")
                                GeometryInput.iml = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "crystal_Si")
                                GeometryInput.crystal_Si = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&UserSettings" && line.find("&end") == string::npos){

                    if(line.find("&UserSettings") != string::npos){

                        for(string elem : split(split(line, "&UserSettings ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "see_para")
                                UserSettingsInput.see_para = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "see_anti")
                                UserSettingsInput.see_anti = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "Make_Vertical")
                                UserSettingsInput.Make_Vertical = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Make_Horizontal")
                                UserSettingsInput.Make_Horizontal = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "angle_aprox")
                                UserSettingsInput.angle_aprox = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "fitting")
                                UserSettingsInput.fitting = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "TrueVoigt")
                                UserSettingsInput.TrueVoigt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Simple_simu")
                                UserSettingsInput.Simple_simu = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_1crys")
                                UserSettingsInput.center_1crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_2crys")
                                UserSettingsInput.center_2crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "mask_C1")
                                UserSettingsInput.mask_C1 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == " mask_C2")
                                UserSettingsInput.mask_C2 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "print_scan")
                                UserSettingsInput.print_scan = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_Mask")
                                UserSettingsInput.center_Mask = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_mask_test")
                                UserSettingsInput.make_mask_test = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "see_para")
                                UserSettingsInput.see_para = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "see_anti")
                                UserSettingsInput.see_anti = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "Make_Vertical")
                                UserSettingsInput.Make_Vertical = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Make_Horizontal")
                                UserSettingsInput.Make_Horizontal = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "angle_aprox")
                                UserSettingsInput.angle_aprox = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "fitting")
                                UserSettingsInput.fitting = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "TrueVoigt")
                                UserSettingsInput.TrueVoigt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Simple_simu")
                                UserSettingsInput.Simple_simu = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_1crys")
                                UserSettingsInput.center_1crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_2crys")
                                UserSettingsInput.center_2crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "mask_C1")
                                UserSettingsInput.mask_C1 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == " mask_C2")
                                UserSettingsInput.mask_C2 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "print_scan")
                                UserSettingsInput.print_scan = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_Mask")
                                UserSettingsInput.center_Mask = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_mask_test")
                                UserSettingsInput.make_mask_test = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&GeoParapathlenghts" && line.find("&end") == string::npos){

                    if(line.find("&GeoParapathlenghts") != string::npos){

                        for(string elem : split(split(line, "&GeoParapathlenghts ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "type_source")
                                GeoParapathlengthsInput.type_source = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "LT_aper")
                                GeoParapathlengthsInput.LT_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_T_Cr1")
                                GeoParapathlengthsInput.dist_T_Cr1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " dist_Cr1_Cr2")
                                GeoParapathlengthsInput.dist_Cr1_Cr2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_Cr2_Det")
                                GeoParapathlengthsInput.dist_Cr2_Det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_aper_det")
                                GeoParapathlengthsInput.dist_aper_det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "type_source")
                                GeoParapathlengthsInput.type_source = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "LT_aper")
                                GeoParapathlengthsInput.LT_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_T_Cr1")
                                GeoParapathlengthsInput.dist_T_Cr1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " dist_Cr1_Cr2")
                                GeoParapathlengthsInput.dist_Cr1_Cr2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_Cr2_Det")
                                GeoParapathlengthsInput.dist_Cr2_Det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_aper_det")
                                GeoParapathlengthsInput.dist_aper_det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);

                        }
                    }

                }else if(currStruct == "&Geolenghtelemets" && line.find("&end") == string::npos){

                    if(line.find("&Geolenghtelemets") != string::npos){

                        for(string elem : split(split(line, "&Geolenghtelemets ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "S_aper")
                                GeolengthelementsInput.S_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_aper_var")
                                GeolengthelementsInput.S_aper_var = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_sour")
                                GeolengthelementsInput.S_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_sour")
                                GeolengthelementsInput.y_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_sour")
                                GeolengthelementsInput.z_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "y_aper")
                                GeolengthelementsInput.y_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_aper")
                                GeolengthelementsInput.z_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " S_shi_hor_B")
                                GeolengthelementsInput.S_shi_hor_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_hor_A")
                                GeolengthelementsInput.S_shi_hor_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                GeolengthelementsInput.S_shi_ver_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_A")
                                GeolengthelementsInput.S_shi_ver_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_first_crys")
                                GeolengthelementsInput.y_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_first_crys")
                                GeolengthelementsInput.z_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "ydetc")
                                GeolengthelementsInput.ydetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "zdetc")
                                GeolengthelementsInput.zdetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "shift_det_ver")
                                GeolengthelementsInput.shift_det_ver = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "S_aper")
                                GeolengthelementsInput.S_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_aper_var")
                                GeolengthelementsInput.S_aper_var = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_sour")
                                GeolengthelementsInput.S_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_sour")
                                GeolengthelementsInput.y_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_sour")
                                GeolengthelementsInput.z_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "y_aper")
                                GeolengthelementsInput.y_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_aper")
                                GeolengthelementsInput.z_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " S_shi_hor_B")
                                GeolengthelementsInput.S_shi_hor_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_hor_A")
                                GeolengthelementsInput.S_shi_hor_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                GeolengthelementsInput.S_shi_ver_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_A")
                                GeolengthelementsInput.S_shi_ver_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_first_crys")
                                GeolengthelementsInput.y_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_first_crys")
                                GeolengthelementsInput.z_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "ydetc")
                                GeolengthelementsInput.ydetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "zdetc")
                                GeolengthelementsInput.zdetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "shift_det_ver")
                                GeolengthelementsInput.shift_det_ver = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&GeoParameters" && line.find("&end") == string::npos){

                    if(line.find("&GeoParameters") != string::npos){

                        for(string elem : split(split(line, "&GeoParameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Exp_crys1")
                                GeoParametersInput.Exp_crys1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_table")
                                GeoParametersInput.teta_table = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "OffsetRotCry1")
                                GeoParametersInput.OffsetRotCry1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " teta_detec_para")
                                GeoParametersInput.teta_detec_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_detec_anti")
                                GeoParametersInput.teta_detec_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C1")
                                GeoParametersInput.tilt_C1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C2")
                                GeoParametersInput.tilt_C2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " xsi")
                                GeoParametersInput.xsi = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "center_1cry_at")
                                GeoParametersInput.center_1cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                GeoParametersInput.center_2cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Exp_crys1")
                                GeoParametersInput.Exp_crys1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_table")
                                GeoParametersInput.teta_table = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "OffsetRotCry1")
                                GeoParametersInput.OffsetRotCry1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " teta_detec_para")
                                GeoParametersInput.teta_detec_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_detec_anti")
                                GeoParametersInput.teta_detec_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C1")
                                GeoParametersInput.tilt_C1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C2")
                                GeoParametersInput.tilt_C2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " xsi")
                                GeoParametersInput.xsi = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "center_1cry_at")
                                GeoParametersInput.center_1cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                GeoParametersInput.center_2cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&CurveVerticalTilt" && line.find("&end") == string::npos){

                    if(line.find("&CurveVerticalTilt") != string::npos){

                        for(string elem : split(split(line, "&CurveVerticalTilt ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_CurveTilt")
                                CurveVerticalTiltInput.make_CurveTilt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "phas_tilt1")
                                CurveVerticalTiltInput.phas_tilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "phas_tilt2")
                                CurveVerticalTiltInput.phas_tilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " offsettilt1")
                                CurveVerticalTiltInput.offsettilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "offsettilt2")
                                CurveVerticalTiltInput.offsettilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt1")
                                CurveVerticalTiltInput.consttilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt2")
                                CurveVerticalTiltInput.consttilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_CurveTilt")
                                CurveVerticalTiltInput.make_CurveTilt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "phas_tilt1")
                                CurveVerticalTiltInput.phas_tilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "phas_tilt2")
                                CurveVerticalTiltInput.phas_tilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " offsettilt1")
                                CurveVerticalTiltInput.offsettilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "offsettilt2")
                                CurveVerticalTiltInput.offsettilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt1")
                                CurveVerticalTiltInput.consttilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt2")
                                CurveVerticalTiltInput.consttilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&Graph_options" && line.find("&end") == string::npos){

                    if(line.find("&Graph_options") != string::npos){

                        for(string elem : split(split(line, "&Graph_options ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "MakeDislin")
                                GraphOptionsInput.MakeDislin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_profile")
                                GraphOptionsInput.make_graph_profile = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_image_plates")
                                GraphOptionsInput.make_image_plates = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " make_imageC1_After_refle")
                                GraphOptionsInput.make_imageC1_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_imageC2_After_refle")
                                GraphOptionsInput.make_imageC2_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "MakeDislin")
                                GraphOptionsInput.MakeDislin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_profile")
                                GraphOptionsInput.make_graph_profile = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_image_plates")
                                GraphOptionsInput.make_image_plates = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " make_imageC1_After_refle")
                                GraphOptionsInput.make_imageC1_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_imageC2_After_refle")
                                GraphOptionsInput.make_imageC2_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&plotparameters" && line.find("&end") == string::npos){

                    if(line.find("&plotparameters") != string::npos){

                        for(string elem : split(split(line, "&plotparameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "delta_angl")
                                PlotParametersInput.delta_angl = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " shift_disp_window")
                                PlotParametersInput.shift_disp_window = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " nubins")
                                PlotParametersInput.nubins = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "delta_angl")
                                PlotParametersInput.delta_angl = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " shift_disp_window")
                                PlotParametersInput.shift_disp_window = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " nubins")
                                PlotParametersInput.nubins = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }

                }else if(currStruct == "&numberrays" && line.find("&end") == string::npos){

                    if(line.find("&numberrays") != string::npos){

                        for(string elem : split(split(line, "&numberrays ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "nbeams")
                                NumberRaysInput.nbeams = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "number_rotati")
                                NumberRaysInput.number_rotati = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if (item == "number_graph_events")
                                NumberRaysInput.number_events = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "nbeams")
                                NumberRaysInput.nbeams = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "number_rotati")
                                NumberRaysInput.number_rotati = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if (item == "number_graph_events")
                                NumberRaysInput.number_events = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }

                }else if(currStruct == "&physical_parameters" && line.find("&end") == string::npos){

                    if(line.find("&physical_parameters") != string::npos){

                        for(string elem : split(split(line, "&physical_parameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Unit_energy")
                                PhysicalParametersInput.Unit_energy = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "linelamda")
                                PhysicalParametersInput.linelamda = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth")
                                PhysicalParametersInput.naturalwidth = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "gauss_Doop")
                                PhysicalParametersInput.gauss_Doop = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Unit_energy")
                                PhysicalParametersInput.Unit_energy = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "linelamda")
                                PhysicalParametersInput.linelamda = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth")
                                PhysicalParametersInput.naturalwidth = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "gauss_Doop")
                                PhysicalParametersInput.gauss_Doop = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&polarization_parameters" && line.find("&end") == string::npos){

                    if(line.find("&polarization_parameters") != string::npos){

                        for(string elem : split(split(line, "&polarization_parameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mka_poli")
                                PolarizationParametersInput.mka_poli = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "relationP_S")
                                PolarizationParametersInput.relationP_S = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mka_poli")
                                PolarizationParametersInput.mka_poli = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "relationP_S")
                                PolarizationParametersInput.relationP_S = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&temperature_parameters" && line.find("&end") == string::npos){

                    if(line.find("&temperature_parameters") != string::npos){

                        for(string elem : split(split(line, "&temperature_parameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "T_crystal_1_para")
                                TemperatureParametersInput.T_crystal_1_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_1_anti")
                                TemperatureParametersInput.T_crystal_1_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_para")
                                TemperatureParametersInput.T_crystal_2_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_anti")
                                TemperatureParametersInput.T_crystal_2_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "mk_temp_bin")
                                TemperatureParametersInput.mk_temp_bin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "  AA_tempera")
                                TemperatureParametersInput.AA_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "TT_tempera")
                                TemperatureParametersInput.TT_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "T_crystal_1_para")
                                TemperatureParametersInput.T_crystal_1_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_1_anti")
                                TemperatureParametersInput.T_crystal_1_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_para")
                                TemperatureParametersInput.T_crystal_2_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_anti")
                                TemperatureParametersInput.T_crystal_2_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "mk_temp_bin")
                                TemperatureParametersInput.mk_temp_bin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "  AA_tempera")
                                TemperatureParametersInput.AA_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "TT_tempera")
                                TemperatureParametersInput.TT_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&fullenergyspectrum" && line.find("&end") == string::npos){

                    if(line.find("&fullenergyspectrum") != string::npos){

                        for(string elem : split(split(line, "&fullenergyspectrum ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_more_lines")
                                FullEnergySpectrumInput.make_more_lines = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "linelamda1")
                                FullEnergySpectrumInput.linelamda1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth1")
                                FullEnergySpectrumInput.naturalwidth1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p1_ener")
                                FullEnergySpectrumInput.p1_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda2")
                                FullEnergySpectrumInput.linelamda2 = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "naturalwidth2")
                                FullEnergySpectrumInput.naturalwidth2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p2_ener")
                                FullEnergySpectrumInput.p2_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " linelamda3")
                                FullEnergySpectrumInput.linelamda3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth3")
                                FullEnergySpectrumInput.naturalwidth3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p3_ener")
                                FullEnergySpectrumInput.p3_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda4")
                                FullEnergySpectrumInput.linelamda4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth4")
                                FullEnergySpectrumInput.naturalwidth4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "Do_background")
                                FullEnergySpectrumInput.Do_background = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_more_lines")
                                FullEnergySpectrumInput.make_more_lines = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "linelamda1")
                                FullEnergySpectrumInput.linelamda1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth1")
                                FullEnergySpectrumInput.naturalwidth1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p1_ener")
                                FullEnergySpectrumInput.p1_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda2")
                                FullEnergySpectrumInput.linelamda2 = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "naturalwidth2")
                                FullEnergySpectrumInput.naturalwidth2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p2_ener")
                                FullEnergySpectrumInput.p2_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " linelamda3")
                                FullEnergySpectrumInput.linelamda3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth3")
                                FullEnergySpectrumInput.naturalwidth3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p3_ener")
                                FullEnergySpectrumInput.p3_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda4")
                                FullEnergySpectrumInput.linelamda4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth4")
                                FullEnergySpectrumInput.naturalwidth4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "Do_background")
                                FullEnergySpectrumInput.Do_background = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&Curved_Crystal" && line.find("&end") == string::npos){

                    if(line.find("&Curved_Crystal") != string::npos){

                        for(string elem : split(split(line, "&Curved_Crystal ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Curve_crystall")
                                CurvedCrystalInput.Curve_crystall = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "R_cur_crys_1")
                                CurvedCrystalInput.R_cur_crys_1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "R_cur_crys_2")
                                CurvedCrystalInput.R_cur_crys_2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Curve_crystall")
                                CurvedCrystalInput.Curve_crystall = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "R_cur_crys_1")
                                CurvedCrystalInput.R_cur_crys_1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "R_cur_crys_2")
                                CurvedCrystalInput.R_cur_crys_2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&AnalysiesCrystaltilts" && line.find("&end") == string::npos){

                    if(line.find("&AnalysiesCrystaltilts") != string::npos){

                        for(string elem : split(split(line, "&AnalysiesCrystaltilts ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_matrix_full")
                                AnalysiesCrystaltiltsInput.make_matrix_full = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_widths")
                                AnalysiesCrystaltiltsInput.make_graph_widths = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "metafile")
                                AnalysiesCrystaltiltsInput.metafile = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == " make_an_C1_ta")
                                AnalysiesCrystaltiltsInput.make_an_C1_ta = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_plot_C1_table")
                                AnalysiesCrystaltiltsInput.make_plot_C1_table = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_matrix_full")
                                AnalysiesCrystaltiltsInput.make_matrix_full = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_widths")
                                AnalysiesCrystaltiltsInput.make_graph_widths = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "metafile")
                                AnalysiesCrystaltiltsInput.metafile = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == " make_an_C1_ta")
                                AnalysiesCrystaltiltsInput.make_an_C1_ta = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_plot_C1_table")
                                AnalysiesCrystaltiltsInput.make_plot_C1_table = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }
            }
        }else{
            QString message = "Path to simulation workspace: ";
            message.append(File_simu);

            QMessageBox msgBox;
            msgBox.setText(message);
            msgBox.setInformativeText("An input file with the C++ configuration model has been found.\nWould you like to continue?");
            msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
            msgBox.setDefaultButton(QMessageBox::Yes);
            int ret = msgBox.exec();


            if (ret == QMessageBox::No) {
                return ret;
            }

            while (getline(inputFile, line)){
                if (line.size() < 1) continue;
                if(line[0] != '/' && line[1] != '/'){

                    if(line.find("Geometry") != string::npos)
                        currStruct = "&Geometry";
                    else if(line.find("UserSettings") != string::npos)
                        currStruct = "&UserSettings";
                    else if(line.find("GeoParapathlenghts") != string::npos)
                        currStruct = "&GeoParapathlenghts";
                    else if(line.find("Geolenghtelemets") != string::npos)
                        currStruct = "&Geolenghtelemets";
                    else if(line.find("GeoParameters") != string::npos)
                        currStruct = "&GeoParameters";
                    else if(line.find("CurveVerticalTilt") != string::npos)
                        currStruct = "&CurveVerticalTilt";
                    else if(line.find("Graph_options") != string::npos)
                        currStruct = "&Graph_options";
                    else if(line.find("plotparameters") != string::npos)
                        currStruct = "&plotparameters";
                    else if(line.find("numberrays") != string::npos)
                        currStruct = "&numberrays";
                    else if(line.find("physical_parameters") != string::npos)
                        currStruct = "&physical_parameters";
                    else if(line.find("polarization_parameters") != string::npos)
                        currStruct = "&polarization_parameters";
                    else if(line.find("temperature_parameters") != string::npos)
                        currStruct = "&temperature_parameters";
                    else if(line.find("fullenergyspectrum") != string::npos)
                        currStruct = "&fullenergyspectrum";
                    else if(line.find("Curved_Crystal") != string::npos)
                        currStruct = "&Curved_Crystal";
                    else if(line.find("AnalysiesCrystaltilts") != string::npos)
                        currStruct = "&AnalysiesCrystaltilts";



                    if(currStruct == "&Geometry"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "mode_bragg_geo"){
                            trim(items[1]);
                            GeometryInput.mode_bragg_geo = (items[1] == ".true.");
                        }else if(items[0] == "imh"){
                            trim(items[1]);
                            GeometryInput.imh = stoi(items[1]);
                        }else if(items[0] == "imk"){
                            trim(items[1]);
                            GeometryInput.imk = stoi(items[1]);
                        }else if(items[0] == "iml"){
                            trim(items[1]);
                            GeometryInput.iml = stoi(items[1]);
                        }else if(items[0] == "crystal_Si"){
                            trim(items[1]);
                            GeometryInput.crystal_Si = (items[1] == ".true.");
                        }

                    }else if(currStruct == "&UserSettings"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

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
                            UserSettingsInput.angle_aprox = stoi(items[1]);
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
                            UserSettingsInput.mask_C1 = stoi(items[1]);
                        }else if(items[0] == "mask_C2"){
                            trim(items[1]);
                            UserSettingsInput.mask_C2 = stoi(items[1]);
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

                    }else if(currStruct == "&GeoParapathlenghts"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "type_source"){
                            trim(items[1]);
                            GeoParapathlengthsInput.type_source = split(items[1], "\"")[1];
                        }else if(items[0] == "LT_aper"){
                            trim(items[1]);
                            GeoParapathlengthsInput.LT_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_T_Cr1"){
                            trim(items[1]);
                            GeoParapathlengthsInput.dist_T_Cr1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_Cr1_Cr2"){
                            trim(items[1]);
                            GeoParapathlengthsInput.dist_Cr1_Cr2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_Cr2_Det"){
                            trim(items[1]);
                            GeoParapathlengthsInput.dist_Cr2_Det = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_aper_det"){
                            trim(items[1]);
                            GeoParapathlengthsInput.dist_aper_det = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&Geolenghtelemets"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "S_aper"){
                            trim(items[1]);
                            GeolengthelementsInput.S_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_aper_var"){
                            trim(items[1]);
                            GeolengthelementsInput.S_aper_var = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_sour"){
                            trim(items[1]);
                            GeolengthelementsInput.S_sour = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "y_sour"){
                            trim(items[1]);
                            GeolengthelementsInput.y_sour = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "z_sour"){
                            trim(items[1]);
                            GeolengthelementsInput.z_sour = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "y_aper"){
                            trim(items[1]);
                            GeolengthelementsInput.y_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "z_aper"){
                            trim(items[1]);
                            GeolengthelementsInput.z_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_hor_B"){
                            trim(items[1]);
                            GeolengthelementsInput.S_shi_hor_B = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_hor_A"){
                            trim(items[1]);
                            GeolengthelementsInput.S_shi_hor_A = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_ver_B"){
                            trim(items[1]);
                            GeolengthelementsInput.S_shi_ver_B = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_ver_A"){
                            trim(items[1]);
                            GeolengthelementsInput.S_shi_ver_A = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "y_first_crys"){
                            trim(items[1]);
                            GeolengthelementsInput.y_first_crys = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "z_first_crys"){
                            trim(items[1]);
                            GeolengthelementsInput.z_first_crys = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "ydetc"){
                            trim(items[1]);
                            GeolengthelementsInput.ydetc = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "zdetc"){
                            trim(items[1]);
                            GeolengthelementsInput.zdetc = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "shift_det_ver"){
                            trim(items[1]);
                            GeolengthelementsInput.shift_det_ver = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&GeoParameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "Exp_crys1"){
                            trim(items[1]);
                            GeoParametersInput.Exp_crys1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "teta_table"){
                            trim(items[1]);
                            GeoParametersInput.teta_table = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "OffsetRotCry1"){
                            trim(items[1]);
                            GeoParametersInput.OffsetRotCry1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "teta_detec_para"){
                            trim(items[1]);
                            GeoParametersInput.teta_detec_para = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "teta_detec_anti"){
                            trim(items[1]);
                            GeoParametersInput.teta_detec_anti = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "tilt_C1"){
                            trim(items[1]);
                            GeoParametersInput.tilt_C1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "tilt_C2"){
                            trim(items[1]);
                            GeoParametersInput.tilt_C2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "xsi"){
                            trim(items[1]);
                            GeoParametersInput.xsi = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "center_1cry_at"){
                            trim(items[1]);
                            GeoParametersInput.center_1cry_at = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_ver_B"){
                            trim(items[1]);
                            GeoParametersInput.center_2cry_at = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&CurveVerticalTilt"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "make_CurveTilt"){
                            trim(items[1]);
                            CurveVerticalTiltInput.make_CurveTilt = (items[1] == ".true.");
                        }else if(items[0] == "phas_tilt1"){
                            trim(items[1]);
                            CurveVerticalTiltInput.phas_tilt1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "phas_tilt2"){
                            trim(items[1]);
                            CurveVerticalTiltInput.phas_tilt2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "offsettilt1"){
                            trim(items[1]);
                            CurveVerticalTiltInput.offsettilt1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "offsettilt2"){
                            trim(items[1]);
                            CurveVerticalTiltInput.offsettilt2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "consttilt1"){
                            trim(items[1]);
                            CurveVerticalTiltInput.consttilt1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "consttilt2"){
                            trim(items[1]);
                            CurveVerticalTiltInput.consttilt2 = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&Graph_options"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "MakeDislin"){
                            trim(items[1]);
                            GraphOptionsInput.MakeDislin = (items[1] == ".true.");
                        }else if(items[0] == "make_graph_profile"){
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

                    }else if(currStruct == "&plotparameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "delta_angl"){
                            trim(items[1]);
                            PlotParametersInput.delta_angl = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "shift_disp_window"){
                            trim(items[1]);
                            PlotParametersInput.shift_disp_window = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "nubins"){
                            trim(items[1]);
                            PlotParametersInput.nubins = stoi(items[1]);
                        }

                    }else if(currStruct == "&numberrays"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "nbeams"){
                            trim(items[1]);
                            NumberRaysInput.nbeams = stoi(items[1]);
                        }else if(items[0] == "number_rotati"){
                            trim(items[1]);
                            NumberRaysInput.number_rotati = stoi(items[1]);
                        }else if (items[0] == "number_graph_events") {
                            trim(items[1]);
                            NumberRaysInput.number_events = stoi(items[1]);
                        }

                    }else if(currStruct == "&physical_parameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "Unit_energy"){
                            trim(items[1]);
                            PhysicalParametersInput.Unit_energy = split(items[1], "\"")[1];
                        }else if(items[0] == "linelamda"){
                            trim(items[1]);
                            PhysicalParametersInput.linelamda = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth"){
                            trim(items[1]);
                            PhysicalParametersInput.naturalwidth = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "gauss_Doop"){
                            trim(items[1]);
                            PhysicalParametersInput.gauss_Doop = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&polarization_parameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "mka_poli"){
                            trim(items[1]);
                            PolarizationParametersInput.mka_poli = (items[1] == ".true.");
                        }else if(items[0] == "relationP_S"){
                            trim(items[1]);
                            PolarizationParametersInput.relationP_S = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&temperature_parameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "T_crystal_1_para"){
                            trim(items[1]);
                            TemperatureParametersInput.T_crystal_1_para = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "T_crystal_1_anti"){
                            trim(items[1]);
                            TemperatureParametersInput.T_crystal_1_anti = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "T_crystal_2_para"){
                            trim(items[1]);
                            TemperatureParametersInput.T_crystal_2_para = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "T_crystal_2_anti"){
                            trim(items[1]);
                            TemperatureParametersInput.T_crystal_2_anti = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "mk_temp_bin"){
                            trim(items[1]);
                            TemperatureParametersInput.mk_temp_bin = (items[1] == ".true.");
                        }else if(items[0] == "AA_tempera"){
                            trim(items[1]);
                            TemperatureParametersInput.AA_tempera = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "TT_tempera"){
                            trim(items[1]);
                            TemperatureParametersInput.TT_tempera = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&fullenergyspectrum"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "make_more_lines"){
                            trim(items[1]);
                            FullEnergySpectrumInput.make_more_lines = stoi(items[1]);
                        }else if(items[0] == "linelamda1"){
                            trim(items[1]);
                            FullEnergySpectrumInput.linelamda1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth1"){
                            trim(items[1]);
                            FullEnergySpectrumInput.naturalwidth1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "p1_ener"){
                            trim(items[1]);
                            FullEnergySpectrumInput.p1_ener = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "linelamda2"){
                            trim(items[1]);
                            FullEnergySpectrumInput.linelamda2 = (items[1] == ".true.");
                        }else if(items[0] == "naturalwidth2"){
                            trim(items[1]);
                            FullEnergySpectrumInput.naturalwidth2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "p2_ener"){
                            trim(items[1]);
                            FullEnergySpectrumInput.p2_ener = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "linelamda3"){
                            trim(items[1]);
                            FullEnergySpectrumInput.linelamda3 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth3"){
                            trim(items[1]);
                            FullEnergySpectrumInput.naturalwidth3 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "p3_ener"){
                            trim(items[1]);
                            FullEnergySpectrumInput.p3_ener = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "linelamda4"){
                            trim(items[1]);
                            FullEnergySpectrumInput.linelamda4 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth4"){
                            trim(items[1]);
                            FullEnergySpectrumInput.naturalwidth4 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "Do_background"){
                            trim(items[1]);
                            FullEnergySpectrumInput.Do_background = (items[1] == ".true.");
                        }

                    }else if(currStruct == "&Curved_Crystal"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "Curve_crystall"){
                            trim(items[1]);
                            CurvedCrystalInput.Curve_crystall = (items[1] == ".true.");
                        }else if(items[0] == "R_cur_crys_1"){
                            trim(items[1]);
                            CurvedCrystalInput.R_cur_crys_1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "R_cur_crys_2"){
                            trim(items[1]);
                            CurvedCrystalInput.R_cur_crys_2 = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&AnalysiesCrystaltilts"){
                        
                        
                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "make_matrix_full"){
                            trim(items[1]);
                            AnalysiesCrystaltiltsInput.make_matrix_full = (items[1] == ".true.");
                        }else if(items[0] == "make_graph_widths"){
                            trim(items[1]);
                            AnalysiesCrystaltiltsInput.make_graph_widths = (items[1] == ".true.");
                        }else if(items[0] == "metafile"){
                            trim(items[1]);
                            AnalysiesCrystaltiltsInput.metafile = split(items[1], "\"")[1];
                        }else if(items[0] == "make_an_C1_ta"){
                            trim(items[1]);
                            AnalysiesCrystaltiltsInput.make_an_C1_ta = (items[1] == ".true.");
                        }else if(items[0] == "make_plot_C1_table"){
                            trim(items[1]);
                            AnalysiesCrystaltiltsInput.make_plot_C1_table = (items[1] == ".true.");
                        }

                    }
                }
            }
        }
    }else{
        QString message = "Could not open input file: ";
        message.append(inFile);

        QMessageBox msgBox;
        msgBox.setText(message);
        msgBox.setInformativeText("Please check if the file still exists or has been corruped.");
        msgBox.setStandardButtons(QMessageBox::Close);
        msgBox.setDefaultButton(QMessageBox::Close);
        int ret = msgBox.exec();

        return ret;
    }
    inputFile.close();
    //cout << "Input file read." << endl;

    // Make aditional configurations
    Unit_energy = physical_parameters.Unit_energy;
    if(GeometryInput.imh == 2 and GeometryInput.imk == 2 and GeometryInput.iml == 2)
    {
        refra_corr = refra_corrNIST;
    }
    else
    {
        refra_corr = refra_corrPARIS;
    }

    return 1; // Dirty workaround
}


int ENTRYPOINT(int argc, char *argv[]){

    QApplication a(argc, argv);

    Unit_energy = PhysicalParametersInput.Unit_energy;

    // This is my proposal (César) Just use an argument on the standalone executable instead of a .path file
    // The new version does not require the input file
    // However, there are a few option that need to be added to the GUI before we can remove this feature
    // Also just leaving for compatibility is ok I believe
    if(!configure(argc, argv)) return 0;

    root_script = false;

    if(!root_script){
        if(GraphOptionsInput.MakeDislin){
            // DCS_GUI w;
            GUISettingsWindow w;
            w.show();
            return a.exec();
        }
    }else{
        GraphOptionsInput.MakeDislin = false;
        //old unimplemented code?
        //Obtain_input_root();
    }


    if(UserSettingsInput.mask_C1 < 0 || UserSettingsInput.mask_C1 > 2){
        QString message = "Bad input for first crystal mask: ";
        message.append(UserSettingsInput.mask_C1);

        QMessageBox msgBox;
        msgBox.setText(message);
        msgBox.setInformativeText("The first crystal mask has to be 0, 1 or 2.\nValue of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top.");
        msgBox.setStandardButtons(QMessageBox::Close);
        msgBox.setDefaultButton(QMessageBox::Close);
        int ret = msgBox.exec();

        return ret;
    }

    if(UserSettingsInput.mask_C2 < 0 || UserSettingsInput.mask_C2 > 2){
        QString message = "Bad input for second crystal mask: ";
        message.append(UserSettingsInput.mask_C2);

        QMessageBox msgBox;
        msgBox.setText(message);
        msgBox.setInformativeText("The second crystal mask has to be 0, 1 or 2.\nValue of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top.");
        msgBox.setStandardButtons(QMessageBox::Close);
        msgBox.setDefaultButton(QMessageBox::Close);
        int ret = msgBox.exec();

        return ret;
    }


    if(!GraphOptionsInput.MakeDislin){
        GraphOptionsInput.make_graph_profile = false;
        GraphOptionsInput.make_image_plates = false;
        AnalysiesCrystaltiltsInput.make_graph_widths = false;
    }

    if(FullEnergySpectrumInput.make_more_lines == 1){
        if(!FullEnergySpectrumInput.Do_background){
            if(FullEnergySpectrumInput.p3_ener + FullEnergySpectrumInput.p2_ener + FullEnergySpectrumInput.p1_ener > 1.0){
                QString message = "The sum of ratios: ";
                message.append(to_string(FullEnergySpectrumInput.p1_ener).c_str());
                message.append(" + ");
                message.append(to_string(FullEnergySpectrumInput.p2_ener).c_str());
                message.append(" + ");
                message.append(to_string(FullEnergySpectrumInput.p3_ener).c_str());
                message.append(" is greater than 1.");

                QMessageBox msgBox;
                msgBox.setText("Bad input for lines proportion.");
                msgBox.setInformativeText(message);
                msgBox.setStandardButtons(QMessageBox::Close);
                msgBox.setDefaultButton(QMessageBox::Close);
                int ret = msgBox.exec();

                return ret;
            }
        }

        reques_energ[0] = FullEnergySpectrumInput.linelamda1;
        reques_energ[1] = FullEnergySpectrumInput.linelamda2;
        reques_energ[2] = FullEnergySpectrumInput.linelamda3;
        reques_energ[3] = FullEnergySpectrumInput.linelamda4;

        reques_width[0] = FullEnergySpectrumInput.naturalwidth1;
        reques_width[1] = FullEnergySpectrumInput.naturalwidth2;
        reques_width[2] = FullEnergySpectrumInput.naturalwidth3;
        reques_width[3] = FullEnergySpectrumInput.naturalwidth4;

    }else if(FullEnergySpectrumInput.make_more_lines == 0){
        reques_energ[0] = linelamda;
        //reques_energ[1] = FullEnergySpectrumInput.linelamda1;
        reques_energ[1] = fullenergyspectrum.linelamda2;
        // TODO(César) : Is this correct now ? Also using arrays with index 1
        //               I suppose this is due to the porting from FORTRAN
        reques_energ[2] = FullEnergySpectrumInput.linelamda3;
        reques_energ[3] = FullEnergySpectrumInput.linelamda4;

        reques_width[0] = naturalwidth;
        //reques_width[1] = FullEnergySpectrumInput.naturalwidth1;
        reques_width[1] = fullenergyspectrum.naturalwidth2;
        // TODO(César) : Is this correct now ? Also using arrays with index 1
        //               I suppose this is due to the porting from FORTRAN
        reques_width[2] = FullEnergySpectrumInput.naturalwidth3;
        reques_width[3] = FullEnergySpectrumInput.naturalwidth4;
    }else{
        //cout << "Reading input energy spectrum..." << endl;

        Util::Read_EnergySpectrum(fullenergyspectrum.energy_spectrum_file);

        //cout << "Input energy spectrum read." << endl;
    }

    if(GeometryInput.crystal_Si){
        d_lat = a_si_para / sqrt(pow(GeometryInput.imh, 2) + pow(GeometryInput.imk, 2) + pow(GeometryInput.iml, 2));
    }else{
        d_lat = a_Ge_para / sqrt(pow(GeometryInput.imh, 2) + pow(GeometryInput.imk, 2) + pow(GeometryInput.iml, 2));
    }

    if(FullEnergySpectrumInput.make_more_lines == 0 || FullEnergySpectrumInput.make_more_lines == 1){
        if(PhysicalParametersInput.Unit_energy == evv[0]){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] < 10.0){
                    QString message = "Bad input on the energies: ";
                    message.append(to_string(reques_energ[i]).c_str());

                    QMessageBox msgBox;
                    msgBox.setText(message);
                    msgBox.setInformativeText("Requested energy less than 10 eV.");
                    msgBox.setStandardButtons(QMessageBox::Close);
                    msgBox.setDefaultButton(QMessageBox::Close);
                    int ret = msgBox.exec();

                    return ret;
                }
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] > 10.0){
                    QString message = "Bad input on the energies: ";
                    message.append(to_string(reques_energ[i]).c_str());

                    QMessageBox msgBox;
                    msgBox.setText(message);
                    msgBox.setInformativeText("Requested energy more than 10 A.");
                    msgBox.setStandardButtons(QMessageBox::Close);
                    msgBox.setDefaultButton(QMessageBox::Close);
                    int ret = msgBox.exec();

                    return ret;
                }
            }
        }else{
            QString message = "Bad input on the energy unit: ";
            message.append(PhysicalParametersInput.Unit_energy.c_str());

            QMessageBox msgBox;
            msgBox.setText(message);
            msgBox.setInformativeText("The accepted energy units are keV, eV and A.");
            msgBox.setStandardButtons(QMessageBox::Close);
            msgBox.setDefaultButton(QMessageBox::Close);
            int ret = msgBox.exec();

            return ret;
        }
    }else{
        bool usable;

        if(PhysicalParametersInput.Unit_energy == "keV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                QString message = "Bad input on the energies for energy unit: ";
                message.append(PhysicalParametersInput.Unit_energy.c_str());

                QMessageBox msgBox;
                msgBox.setText(message);
                msgBox.setInformativeText("Requested energy spectrum will not be visible in output.");
                msgBox.setStandardButtons(QMessageBox::Close);
                msgBox.setDefaultButton(QMessageBox::Close);
                int ret = msgBox.exec();

                return ret;
            }
        }else if(PhysicalParametersInput.Unit_energy == "eV"){
            usable = Util::CheckSpectrum("eV");

            if(! usable){
                QString message = "Bad input on the energies for energy unit: ";
                message.append(PhysicalParametersInput.Unit_energy.c_str());

                QMessageBox msgBox;
                msgBox.setText(message);
                msgBox.setInformativeText("Requested energy spectrum will not be visible in output.");
                msgBox.setStandardButtons(QMessageBox::Close);
                msgBox.setDefaultButton(QMessageBox::Close);
                int ret = msgBox.exec();

                return ret;
            }
        }else if(PhysicalParametersInput.Unit_energy == "A"){
            usable = Util::CheckSpectrum("A");

            if(! usable){
                QString message = "Bad input on the energies for energy unit: ";
                message.append(PhysicalParametersInput.Unit_energy.c_str());

                QMessageBox msgBox;
                msgBox.setText(message);
                msgBox.setInformativeText("Requested energy spectrum will not be visible in output.");
                msgBox.setStandardButtons(QMessageBox::Close);
                msgBox.setDefaultButton(QMessageBox::Close);
                int ret = msgBox.exec();

                return ret;
            }
        }else{
            QString message = "Bad input on the energy unit: ";
            message.append(PhysicalParametersInput.Unit_energy.c_str());

            QMessageBox msgBox;
            msgBox.setText(message);
            msgBox.setInformativeText("The accepted energy units are keV, eV and A.");
            msgBox.setStandardButtons(QMessageBox::Close);
            msgBox.setDefaultButton(QMessageBox::Close);
            int ret = msgBox.exec();

            return ret;
        }
    }


    if(FullEnergySpectrumInput.make_more_lines == 1){
        for(int i = 0; i < 4; i++){
            reques_width[i] = reques_width[i] / 2.0;

            if(PhysicalParametersInput.Unit_energy == evv[0]){
                picks[i].lamda = Convert_Ag_minusone_eV / reques_energ[i];
                picks[i].natural_varia = Convert_Ag_minusone_eV * reques_width[i] / (pow(reques_energ[i], 2) - pow(reques_width[i], 2));
            }else{
                picks[i].lamda = reques_energ[i];
                picks[i].natural_varia = reques_width[i];
            }
        }
    }else if(FullEnergySpectrumInput.make_more_lines == 0){
        reques_width[1] = reques_width[1] / 2.0;

        if(PhysicalParametersInput.Unit_energy == evv[0]){
            picks[1].lamda = Convert_Ag_minusone_eV / reques_energ[1];
            picks[1].natural_varia = Convert_Ag_minusone_eV * reques_width[1] / (pow(reques_energ[1], 2) - pow(reques_width[1], 2));
        }else{
            picks[1].lamda = reques_energ[1];
            picks[1].natural_varia = reques_width[1];
        }
    }


    gauss_Doop_ev = PhysicalParametersInput.gauss_Doop;
    PhysicalParametersInput.gauss_Doop = Convert_Ag_minusone_eV * PhysicalParametersInput.gauss_Doop / (pow(reques_energ[1], 2) - pow(PhysicalParametersInput.gauss_Doop, 2));


    if(GeometryInput.mode_bragg_geo){
        Double_Crystal_diffraction::Make_Simu(nullptr);
    }else{
        QString message = "Unimplemented transmission mode.";
        
        QMessageBox msgBox;
        msgBox.setText(message);
        msgBox.setInformativeText("The transmition mode is currently unimplemented.\nCheck the wGeant branch to check if this mode has been implemented using the Geant4 libraries.");
        msgBox.setStandardButtons(QMessageBox::Close);
        msgBox.setDefaultButton(QMessageBox::Close);
        int ret = msgBox.exec();

        return ret;
    }

    return 0;
}

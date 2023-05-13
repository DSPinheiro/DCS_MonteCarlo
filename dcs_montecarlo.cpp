//============================================================================
// Name        : DCS_MonteCarlo.cpp
// Author      : Daniel Pinheiro, Pedro Amaro
// Version     : 1.0
// Copyright   : Your copyright notice
// Description : Entry point for the DCS simulation
//============================================================================

#include "dcs_montecarlo.hh"


using namespace std;

char File_simu[200];
std::string Unit_energy;
double refra_corr;
bool root_script;
pick picks[5];
double gauss_Doop_ev;
double d_lat;// = 3.13560123; // CODATA


int main(int argc, char *argv[]){

    QApplication a(argc, argv);

    char pathName[500];

    strcpy(pathName, QApplication::applicationDirPath().toUtf8().constData());
    strcat(pathName, "/DCrystal_input.path");

    ifstream pathFile(pathName);

    string line;


    if(pathFile){
        while (getline(pathFile, line)){
            if(line[0] != '/' and line[1] != '/'){
                if(line.substr(0, 9) == "File_simu"){
                    strcpy(File_simu, split(line, "=")[1].c_str());
                }
            }
        }
    }else{
        cout << "Could not open path file: " << pathName << endl;
    }

    pathFile.close();

    cout << "Path to simulation workspace: " << File_simu << endl;
    cout << endl;
    cout << endl;


    char inFile[223] = { 0 };
    strcat(inFile, File_simu);
    strcat(inFile, "/DCrystal_input.input");

    ifstream inputFile(inFile);

    if(inputFile.is_open()){
        char str[1];
        char* firstChar = str;
        inputFile.read(firstChar, 1);
        inputFile.seekg(0);

        string currStruct = "";

        if(strcmp(firstChar, "&") == 181){
            cout << "Reading input configuration file as FORTRAN model..." << endl;

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



                if(currStruct == "&Geometry" and line.find("&end") == string::npos){

                    if(line.find("&Geometry") != string::npos){

                        for(string elem : split(split(line, "&Geometry ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mode_bragg_geo")
                                Geometry.mode_bragg_geo = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "imh")
                                Geometry.imh = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "imk")
                                Geometry.imk = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "iml")
                                Geometry.iml = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "crystal_Si")
                                Geometry.crystal_Si = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mode_bragg_geo")
                                Geometry.mode_bragg_geo = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "imh")
                                Geometry.imh = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "imk")
                                Geometry.imk = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "iml")
                                Geometry.iml = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "crystal_Si")
                                Geometry.crystal_Si = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&UserSettings" and line.find("&end") == string::npos){

                    if(line.find("&UserSettings") != string::npos){

                        for(string elem : split(split(line, "&UserSettings ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "see_para")
                                UserSettings.see_para = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "see_anti")
                                UserSettings.see_anti = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "Make_Vertical")
                                UserSettings.Make_Vertical = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Make_Horizontal")
                                UserSettings.Make_Horizontal = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "angle_aprox")
                                UserSettings.angle_aprox = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "fitting")
                                UserSettings.fitting = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "TrueVoigt")
                                UserSettings.TrueVoigt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Simple_simu")
                                UserSettings.Simple_simu = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_1crys")
                                UserSettings.center_1crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_2crys")
                                UserSettings.center_2crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "mask_C1")
                                UserSettings.mask_C1 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == " mask_C2")
                                UserSettings.mask_C2 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "print_scan")
                                UserSettings.print_scan = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_Mask")
                                UserSettings.center_Mask = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_mask_test")
                                UserSettings.make_mask_test = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "see_para")
                                UserSettings.see_para = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "see_anti")
                                UserSettings.see_anti = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "Make_Vertical")
                                UserSettings.Make_Vertical = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Make_Horizontal")
                                UserSettings.Make_Horizontal = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "angle_aprox")
                                UserSettings.angle_aprox = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "fitting")
                                UserSettings.fitting = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "TrueVoigt")
                                UserSettings.TrueVoigt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " Simple_simu")
                                UserSettings.Simple_simu = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_1crys")
                                UserSettings.center_1crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_2crys")
                                UserSettings.center_2crys = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "mask_C1")
                                UserSettings.mask_C1 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == " mask_C2")
                                UserSettings.mask_C2 = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "print_scan")
                                UserSettings.print_scan = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "center_Mask")
                                UserSettings.center_Mask = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_mask_test")
                                UserSettings.make_mask_test = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&GeoParapathlenghts" and line.find("&end") == string::npos){

                    if(line.find("&GeoParapathlenghts") != string::npos){

                        for(string elem : split(split(line, "&GeoParapathlenghts ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "type_source")
                                GeoParapathlengths.type_source = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "LT_aper")
                                GeoParapathlengths.LT_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_T_Cr1")
                                GeoParapathlengths.dist_T_Cr1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " dist_Cr1_Cr2")
                                GeoParapathlengths.dist_Cr1_Cr2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_Cr2_Det")
                                GeoParapathlengths.dist_Cr2_Det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_aper_det")
                                GeoParapathlengths.dist_aper_det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "type_source")
                                GeoParapathlengths.type_source = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "LT_aper")
                                GeoParapathlengths.LT_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_T_Cr1")
                                GeoParapathlengths.dist_T_Cr1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " dist_Cr1_Cr2")
                                GeoParapathlengths.dist_Cr1_Cr2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_Cr2_Det")
                                GeoParapathlengths.dist_Cr2_Det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "dist_aper_det")
                                GeoParapathlengths.dist_aper_det = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);

                        }
                    }

                }else if(currStruct == "&Geolenghtelemets" and line.find("&end") == string::npos){

                    if(line.find("&Geolenghtelemets") != string::npos){

                        for(string elem : split(split(line, "&Geolenghtelemets ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "S_aper")
                                Geolengthelements.S_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_aper_var")
                                Geolengthelements.S_aper_var = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_sour")
                                Geolengthelements.S_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_sour")
                                Geolengthelements.y_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_sour")
                                Geolengthelements.z_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "y_aper")
                                Geolengthelements.y_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_aper")
                                Geolengthelements.z_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " S_shi_hor_B")
                                Geolengthelements.S_shi_hor_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_hor_A")
                                Geolengthelements.S_shi_hor_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                Geolengthelements.S_shi_ver_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_A")
                                Geolengthelements.S_shi_ver_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_first_crys")
                                Geolengthelements.y_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_first_crys")
                                Geolengthelements.z_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "ydetc")
                                Geolengthelements.ydetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "zdetc")
                                Geolengthelements.zdetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "shift_det_ver")
                                Geolengthelements.shift_det_ver = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "S_aper")
                                Geolengthelements.S_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_aper_var")
                                Geolengthelements.S_aper_var = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_sour")
                                Geolengthelements.S_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_sour")
                                Geolengthelements.y_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_sour")
                                Geolengthelements.z_sour = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "y_aper")
                                Geolengthelements.y_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_aper")
                                Geolengthelements.z_aper = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " S_shi_hor_B")
                                Geolengthelements.S_shi_hor_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_hor_A")
                                Geolengthelements.S_shi_hor_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                Geolengthelements.S_shi_ver_B = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_A")
                                Geolengthelements.S_shi_ver_A = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " y_first_crys")
                                Geolengthelements.y_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "z_first_crys")
                                Geolengthelements.z_first_crys = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "ydetc")
                                Geolengthelements.ydetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "zdetc")
                                Geolengthelements.zdetc = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "shift_det_ver")
                                Geolengthelements.shift_det_ver = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&GeoParameters" and line.find("&end") == string::npos){

                    if(line.find("&GeoParameters") != string::npos){

                        for(string elem : split(split(line, "&GeoParameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Exp_crys1")
                                GeoParameters.Exp_crys1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_table")
                                GeoParameters.teta_table = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "OffsetRotCry1")
                                GeoParameters.OffsetRotCry1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " teta_detec_para")
                                GeoParameters.teta_detec_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_detec_anti")
                                GeoParameters.teta_detec_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C1")
                                GeoParameters.tilt_C1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C2")
                                GeoParameters.tilt_C2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " xsi")
                                GeoParameters.xsi = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "center_1cry_at")
                                GeoParameters.center_1cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                GeoParameters.center_2cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Exp_crys1")
                                GeoParameters.Exp_crys1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_table")
                                GeoParameters.teta_table = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "OffsetRotCry1")
                                GeoParameters.OffsetRotCry1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " teta_detec_para")
                                GeoParameters.teta_detec_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "teta_detec_anti")
                                GeoParameters.teta_detec_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C1")
                                GeoParameters.tilt_C1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "tilt_C2")
                                GeoParameters.tilt_C2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " xsi")
                                GeoParameters.xsi = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "center_1cry_at")
                                GeoParameters.center_1cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "S_shi_ver_B")
                                GeoParameters.center_2cry_at = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&CurveVerticalTilt" and line.find("&end") == string::npos){

                    if(line.find("&CurveVerticalTilt") != string::npos){

                        for(string elem : split(split(line, "&CurveVerticalTilt ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_CurveTilt")
                                CurveVerticalTilt.make_CurveTilt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "phas_tilt1")
                                CurveVerticalTilt.phas_tilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "phas_tilt2")
                                CurveVerticalTilt.phas_tilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " offsettilt1")
                                CurveVerticalTilt.offsettilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "offsettilt2")
                                CurveVerticalTilt.offsettilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt1")
                                CurveVerticalTilt.consttilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt2")
                                CurveVerticalTilt.consttilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_CurveTilt")
                                CurveVerticalTilt.make_CurveTilt = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "phas_tilt1")
                                CurveVerticalTilt.phas_tilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "phas_tilt2")
                                CurveVerticalTilt.phas_tilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " offsettilt1")
                                CurveVerticalTilt.offsettilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "offsettilt2")
                                CurveVerticalTilt.offsettilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt1")
                                CurveVerticalTilt.consttilt1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "consttilt2")
                                CurveVerticalTilt.consttilt2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&Graph_options" and line.find("&end") == string::npos){

                    if(line.find("&Graph_options") != string::npos){

                        for(string elem : split(split(line, "&Graph_options ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "MakeDislin")
                                Graph_options.MakeDislin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_profile")
                                Graph_options.make_graph_profile = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_image_plates")
                                Graph_options.make_image_plates = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " make_imageC1_After_refle")
                                Graph_options.make_imageC1_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_imageC2_After_refle")
                                Graph_options.make_imageC2_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "MakeDislin")
                                Graph_options.MakeDislin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_profile")
                                Graph_options.make_graph_profile = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_image_plates")
                                Graph_options.make_image_plates = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == " make_imageC1_After_refle")
                                Graph_options.make_imageC1_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_imageC2_After_refle")
                                Graph_options.make_imageC2_After_refle = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&plotparameters" and line.find("&end") == string::npos){

                    if(line.find("&plotparameters") != string::npos){

                        for(string elem : split(split(line, "&plotparameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "delta_angl")
                                plotparameters.delta_angl = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " shift_disp_window")
                                plotparameters.shift_disp_window = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " nubins")
                                plotparameters.nubins = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "delta_angl")
                                plotparameters.delta_angl = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " shift_disp_window")
                                plotparameters.shift_disp_window = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " nubins")
                                plotparameters.nubins = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }

                }else if(currStruct == "&numberrays" and line.find("&end") == string::npos){

                    if(line.find("&numberrays") != string::npos){

                        for(string elem : split(split(line, "&numberrays ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "nbeams")
                                numberrays.nbeams = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "number_rotati")
                                numberrays.number_rotati = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "nbeams")
                                numberrays.nbeams = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "number_rotati")
                                numberrays.number_rotati = stoi(split(split(elem, "=")[1], "/")[0]);
                        }
                    }

                }else if(currStruct == "&physical_parameters" and line.find("&end") == string::npos){

                    if(line.find("&physical_parameters") != string::npos){

                        for(string elem : split(split(line, "&physical_parameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Unit_energy")
                                physical_parameters.Unit_energy = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "linelamda")
                                physical_parameters.linelamda = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth")
                                physical_parameters.naturalwidth = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "gauss_Doop")
                                physical_parameters.gauss_Doop = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Unit_energy")
                                physical_parameters.Unit_energy = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == "linelamda")
                                physical_parameters.linelamda = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth")
                                physical_parameters.naturalwidth = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "gauss_Doop")
                                physical_parameters.gauss_Doop = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&polarization_parameters" and line.find("&end") == string::npos){

                    if(line.find("&polarization_parameters") != string::npos){

                        for(string elem : split(split(line, "&polarization_parameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mka_poli")
                                polarization_parameters.mka_poli = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "relationP_S")
                                polarization_parameters.relationP_S = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "mka_poli")
                                polarization_parameters.mka_poli = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "relationP_S")
                                polarization_parameters.relationP_S = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&temperature_parameters" and line.find("&end") == string::npos){

                    if(line.find("&temperature_parameters") != string::npos){

                        for(string elem : split(split(line, "&temperature_parameters ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "T_crystal_1_para")
                                temperature_parameters.T_crystal_1_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_1_anti")
                                temperature_parameters.T_crystal_1_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_para")
                                temperature_parameters.T_crystal_2_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_anti")
                                temperature_parameters.T_crystal_2_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "mk_temp_bin")
                                temperature_parameters.mk_temp_bin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "  AA_tempera")
                                temperature_parameters.AA_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "TT_tempera")
                                temperature_parameters.TT_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "T_crystal_1_para")
                                temperature_parameters.T_crystal_1_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_1_anti")
                                temperature_parameters.T_crystal_1_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_para")
                                temperature_parameters.T_crystal_2_para = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "T_crystal_2_anti")
                                temperature_parameters.T_crystal_2_anti = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "mk_temp_bin")
                                temperature_parameters.mk_temp_bin = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "  AA_tempera")
                                temperature_parameters.AA_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "TT_tempera")
                                temperature_parameters.TT_tempera = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&fullenergyspectrum" and line.find("&end") == string::npos){

                    if(line.find("&fullenergyspectrum") != string::npos){

                        for(string elem : split(split(line, "&fullenergyspectrum ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_more_lines")
                                fullenergyspectrum.make_more_lines = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "linelamda1")
                                fullenergyspectrum.linelamda1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth1")
                                fullenergyspectrum.naturalwidth1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p1_ener")
                                fullenergyspectrum.p1_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda2")
                                fullenergyspectrum.linelamda2 = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "naturalwidth2")
                                fullenergyspectrum.naturalwidth2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p2_ener")
                                fullenergyspectrum.p2_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " linelamda3")
                                fullenergyspectrum.linelamda3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth3")
                                fullenergyspectrum.naturalwidth3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p3_ener")
                                fullenergyspectrum.p3_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda4")
                                fullenergyspectrum.linelamda4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth4")
                                fullenergyspectrum.naturalwidth4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "Do_background")
                                fullenergyspectrum.Do_background = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_more_lines")
                                fullenergyspectrum.make_more_lines = stoi(split(split(elem, "=")[1], "/")[0]);
                            else if(item == "linelamda1")
                                fullenergyspectrum.linelamda1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth1")
                                fullenergyspectrum.naturalwidth1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p1_ener")
                                fullenergyspectrum.p1_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda2")
                                fullenergyspectrum.linelamda2 = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "naturalwidth2")
                                fullenergyspectrum.naturalwidth2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p2_ener")
                                fullenergyspectrum.p2_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == " linelamda3")
                                fullenergyspectrum.linelamda3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth3")
                                fullenergyspectrum.naturalwidth3 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "p3_ener")
                                fullenergyspectrum.p3_ener = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "linelamda4")
                                fullenergyspectrum.linelamda4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "naturalwidth4")
                                fullenergyspectrum.naturalwidth4 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "Do_background")
                                fullenergyspectrum.Do_background = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }else if(currStruct == "&Curved_Crystal" and line.find("&end") == string::npos){

                    if(line.find("&Curved_Crystal") != string::npos){

                        for(string elem : split(split(line, "&Curved_Crystal ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Curve_crystall")
                                Curved_Crystal.Curve_crystall = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "R_cur_crys_1")
                                Curved_Crystal.R_cur_crys_1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "R_cur_crys_2")
                                Curved_Crystal.R_cur_crys_2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "Curve_crystall")
                                Curved_Crystal.Curve_crystall = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "R_cur_crys_1")
                                Curved_Crystal.R_cur_crys_1 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                            else if(item == "R_cur_crys_2")
                                Curved_Crystal.R_cur_crys_2 = stod(split(split(split(elem, "=")[1], "/")[0], "d0")[0]);
                        }
                    }

                }else if(currStruct == "&AnalysiesCrystaltilts" and line.find("&end") == string::npos){

                    if(line.find("&AnalysiesCrystaltilts") != string::npos){

                        for(string elem : split(split(line, "&AnalysiesCrystaltilts ")[1], ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_matrix_full")
                                AnalysiesCrystaltilts.make_matrix_full = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_widths")
                                AnalysiesCrystaltilts.make_graph_widths = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "metafile")
                                AnalysiesCrystaltilts.metafile = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == " make_an_C1_ta")
                                AnalysiesCrystaltilts.make_an_C1_ta = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_plot_C1_table")
                                AnalysiesCrystaltilts.make_plot_C1_table = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }else{
                        for(string elem : split(line, ", ")){
                            string item = split(elem, "=")[0];

                            if(item == "make_matrix_full")
                                AnalysiesCrystaltilts.make_matrix_full = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_graph_widths")
                                AnalysiesCrystaltilts.make_graph_widths = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "metafile")
                                AnalysiesCrystaltilts.metafile = split(split(split(elem, "=")[1], "/")[0], "\"")[1];
                            else if(item == " make_an_C1_ta")
                                AnalysiesCrystaltilts.make_an_C1_ta = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                            else if(item == "make_plot_C1_table")
                                AnalysiesCrystaltilts.make_plot_C1_table = (split(split(elem, "=")[1], "/")[0] == ".true.") ? true : false;
                        }
                    }

                }
            }
        }else{
            cout << "Reading input configuration file as new model..." << endl;

            while (getline(inputFile, line)){

                if (line.size() < 1) continue;

                if(line[0] != '/' and line[1] != '/'){

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
                            Geometry.mode_bragg_geo = (items[1] == ".true.");
                        }else if(items[0] == "imh"){
                            trim(items[1]);
                            Geometry.imh = stoi(items[1]);
                        }else if(items[0] == "imk"){
                            trim(items[1]);
                            Geometry.imk = stoi(items[1]);
                        }else if(items[0] == "iml"){
                            trim(items[1]);
                            Geometry.iml = stoi(items[1]);
                        }else if(items[0] == "crystal_Si"){
                            trim(items[1]);
                            Geometry.crystal_Si = (items[1] == ".true.");
                        }

                    }else if(currStruct == "&UserSettings"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "see_para"){
                            trim(items[1]);
                            UserSettings.see_para = (items[1] == ".true.");
                        }else if(items[0] == "see_anti"){
                            trim(items[1]);
                            UserSettings.see_anti = (items[1] == ".true.");
                        }else if(items[0] == "Make_Vertical"){
                            trim(items[1]);
                            UserSettings.Make_Vertical = (items[1] == ".true.");
                        }else if(items[0] == "Make_Horizontal"){
                            trim(items[1]);
                            UserSettings.Make_Horizontal = (items[1] == ".true.");
                        }else if(items[0] == "angle_aprox"){
                            trim(items[1]);
                            UserSettings.angle_aprox = stoi(items[1]);
                        }else if(items[0] == "fitting"){
                            trim(items[1]);
                            UserSettings.fitting = (items[1] == ".true.");
                        }else if(items[0] == "TrueVoigt"){
                            trim(items[1]);
                            UserSettings.TrueVoigt = (items[1] == ".true.");
                        }else if(items[0] == "Simple_simu"){
                            trim(items[1]);
                            UserSettings.Simple_simu = (items[1] == ".true.");
                        }else if(items[0] == "center_1crys"){
                            trim(items[1]);
                            UserSettings.center_1crys = (items[1] == ".true.");
                        }else if(items[0] == "center_2crys"){
                            trim(items[1]);
                            UserSettings.center_2crys = (items[1] == ".true.");
                        }else if(items[0] == "mask_C1"){
                            trim(items[1]);
                            UserSettings.mask_C1 = stoi(items[1]);
                        }else if(items[0] == "mask_C2"){
                            trim(items[1]);
                            UserSettings.mask_C2 = stoi(items[1]);
                        }else if(items[0] == "print_scan"){
                            trim(items[1]);
                            UserSettings.print_scan = (items[1] == ".true.");
                        }else if(items[0] == "center_Mask"){
                            trim(items[1]);
                            UserSettings.center_Mask = (items[1] == ".true.");
                        }else if(items[0] == "make_mask_test"){
                            trim(items[1]);
                            UserSettings.make_mask_test = (items[1] == ".true.");
                        }

                    }else if(currStruct == "&GeoParapathlenghts"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "type_source"){
                            trim(items[1]);
                            GeoParapathlengths.type_source = split(items[1], "\"")[1];
                        }else if(items[0] == "LT_aper"){
                            trim(items[1]);
                            GeoParapathlengths.LT_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_T_Cr1"){
                            trim(items[1]);
                            GeoParapathlengths.dist_T_Cr1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_Cr1_Cr2"){
                            trim(items[1]);
                            GeoParapathlengths.dist_Cr1_Cr2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_Cr2_Det"){
                            trim(items[1]);
                            GeoParapathlengths.dist_Cr2_Det = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "dist_aper_det"){
                            trim(items[1]);
                            GeoParapathlengths.dist_aper_det = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&Geolenghtelemets"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "S_aper"){
                            trim(items[1]);
                            Geolengthelements.S_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_aper_var"){
                            trim(items[1]);
                            Geolengthelements.S_aper_var = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_sour"){
                            trim(items[1]);
                            Geolengthelements.S_sour = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "y_sour"){
                            trim(items[1]);
                            Geolengthelements.y_sour = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "z_sour"){
                            trim(items[1]);
                            Geolengthelements.z_sour = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "y_aper"){
                            trim(items[1]);
                            Geolengthelements.y_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "z_aper"){
                            trim(items[1]);
                            Geolengthelements.z_aper = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_hor_B"){
                            trim(items[1]);
                            Geolengthelements.S_shi_hor_B = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_hor_A"){
                            trim(items[1]);
                            Geolengthelements.S_shi_hor_A = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_ver_B"){
                            trim(items[1]);
                            Geolengthelements.S_shi_ver_B = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_ver_A"){
                            trim(items[1]);
                            Geolengthelements.S_shi_ver_A = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "y_first_crys"){
                            trim(items[1]);
                            Geolengthelements.y_first_crys = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "z_first_crys"){
                            trim(items[1]);
                            Geolengthelements.z_first_crys = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "ydetc"){
                            trim(items[1]);
                            Geolengthelements.ydetc = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "zdetc"){
                            trim(items[1]);
                            Geolengthelements.zdetc = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "shift_det_ver"){
                            trim(items[1]);
                            Geolengthelements.shift_det_ver = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&GeoParameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "Exp_crys1"){
                            trim(items[1]);
                            GeoParameters.Exp_crys1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "teta_table"){
                            trim(items[1]);
                            GeoParameters.teta_table = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "OffsetRotCry1"){
                            trim(items[1]);
                            GeoParameters.OffsetRotCry1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "teta_detec_para"){
                            trim(items[1]);
                            GeoParameters.teta_detec_para = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "teta_detec_anti"){
                            trim(items[1]);
                            GeoParameters.teta_detec_anti = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "tilt_C1"){
                            trim(items[1]);
                            GeoParameters.tilt_C1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "tilt_C2"){
                            trim(items[1]);
                            GeoParameters.tilt_C2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "xsi"){
                            trim(items[1]);
                            GeoParameters.xsi = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "center_1cry_at"){
                            trim(items[1]);
                            GeoParameters.center_1cry_at = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "S_shi_ver_B"){
                            trim(items[1]);
                            GeoParameters.center_2cry_at = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&CurveVerticalTilt"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "make_CurveTilt"){
                            trim(items[1]);
                            CurveVerticalTilt.make_CurveTilt = (items[1] == ".true.");
                        }else if(items[0] == "phas_tilt1"){
                            trim(items[1]);
                            CurveVerticalTilt.phas_tilt1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "phas_tilt2"){
                            trim(items[1]);
                            CurveVerticalTilt.phas_tilt2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "offsettilt1"){
                            trim(items[1]);
                            CurveVerticalTilt.offsettilt1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "offsettilt2"){
                            trim(items[1]);
                            CurveVerticalTilt.offsettilt2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "consttilt1"){
                            trim(items[1]);
                            CurveVerticalTilt.consttilt1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "consttilt2"){
                            trim(items[1]);
                            CurveVerticalTilt.consttilt2 = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&Graph_options"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "MakeDislin"){
                            trim(items[1]);
                            Graph_options.MakeDislin = (items[1] == ".true.");
                        }else if(items[0] == "make_graph_profile"){
                            trim(items[1]);
                            Graph_options.make_graph_profile = (items[1] == ".true.");
                        }else if(items[0] == "make_image_plates"){
                            trim(items[1]);
                            Graph_options.make_image_plates = (items[1] == ".true.");
                        }else if(items[0] == "make_imageC1_After_refle"){
                            trim(items[1]);
                            Graph_options.make_imageC1_After_refle = (items[1] == ".true.");
                        }else if(items[0] == "make_imageC2_After_refle"){
                            trim(items[1]);
                            Graph_options.make_imageC2_After_refle = (items[1] == ".true.");
                        }

                    }else if(currStruct == "&plotparameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "delta_angl"){
                            trim(items[1]);
                            plotparameters.delta_angl = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "shift_disp_window"){
                            trim(items[1]);
                            plotparameters.shift_disp_window = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "nubins"){
                            trim(items[1]);
                            plotparameters.nubins = stoi(items[1]);
                        }

                    }else if(currStruct == "&numberrays"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "nbeams"){
                            trim(items[1]);
                            numberrays.nbeams = stoi(items[1]);
                        }else if(items[0] == "number_rotati"){
                            trim(items[1]);
                            numberrays.number_rotati = stoi(items[1]);
                        }

                    }else if(currStruct == "&physical_parameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "Unit_energy"){
                            trim(items[1]);
                            physical_parameters.Unit_energy = split(items[1], "\"")[1];
                        }else if(items[0] == "linelamda"){
                            trim(items[1]);
                            physical_parameters.linelamda = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth"){
                            trim(items[1]);
                            physical_parameters.naturalwidth = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "gauss_Doop"){
                            trim(items[1]);
                            physical_parameters.gauss_Doop = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&polarization_parameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "mka_poli"){
                            trim(items[1]);
                            polarization_parameters.mka_poli = (items[1] == ".true.");
                        }else if(items[0] == "relationP_S"){
                            trim(items[1]);
                            polarization_parameters.relationP_S = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&temperature_parameters"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "T_crystal_1_para"){
                            trim(items[1]);
                            temperature_parameters.T_crystal_1_para = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "T_crystal_1_anti"){
                            trim(items[1]);
                            temperature_parameters.T_crystal_1_anti = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "T_crystal_2_para"){
                            trim(items[1]);
                            temperature_parameters.T_crystal_2_para = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "T_crystal_2_anti"){
                            trim(items[1]);
                            temperature_parameters.T_crystal_2_anti = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "mk_temp_bin"){
                            trim(items[1]);
                            temperature_parameters.mk_temp_bin = (items[1] == ".true.");
                        }else if(items[0] == "AA_tempera"){
                            trim(items[1]);
                            temperature_parameters.AA_tempera = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "TT_tempera"){
                            trim(items[1]);
                            temperature_parameters.TT_tempera = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&fullenergyspectrum"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "make_more_lines"){
                            trim(items[1]);
                            fullenergyspectrum.make_more_lines = stoi(items[1]);
                        }else if(items[0] == "linelamda1"){
                            trim(items[1]);
                            fullenergyspectrum.linelamda1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth1"){
                            trim(items[1]);
                            fullenergyspectrum.naturalwidth1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "p1_ener"){
                            trim(items[1]);
                            fullenergyspectrum.p1_ener = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "linelamda2"){
                            trim(items[1]);
                            fullenergyspectrum.linelamda2 = (items[1] == ".true.");
                        }else if(items[0] == "naturalwidth2"){
                            trim(items[1]);
                            fullenergyspectrum.naturalwidth2 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "p2_ener"){
                            trim(items[1]);
                            fullenergyspectrum.p2_ener = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "linelamda3"){
                            trim(items[1]);
                            fullenergyspectrum.linelamda3 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth3"){
                            trim(items[1]);
                            fullenergyspectrum.naturalwidth3 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "p3_ener"){
                            trim(items[1]);
                            fullenergyspectrum.p3_ener = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "linelamda4"){
                            trim(items[1]);
                            fullenergyspectrum.linelamda4 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "naturalwidth4"){
                            trim(items[1]);
                            fullenergyspectrum.naturalwidth4 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "Do_background"){
                            trim(items[1]);
                            fullenergyspectrum.Do_background = (items[1] == ".true.");
                        }

                    }else if(currStruct == "&Curved_Crystal"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "Curve_crystall"){
                            trim(items[1]);
                            Curved_Crystal.Curve_crystall = (items[1] == ".true.");
                        }else if(items[0] == "R_cur_crys_1"){
                            trim(items[1]);
                            Curved_Crystal.R_cur_crys_1 = stod(split(items[1], "d0")[0]);
                        }else if(items[0] == "R_cur_crys_2"){
                            trim(items[1]);
                            Curved_Crystal.R_cur_crys_2 = stod(split(items[1], "d0")[0]);
                        }

                    }else if(currStruct == "&AnalysiesCrystaltilts"){

                        string elem = split(line, "//")[0];
                        vector<string> items = split(elem, "=");

                        trim(items[0]);

                        if(items[0] == "make_matrix_full"){
                            trim(items[1]);
                            AnalysiesCrystaltilts.make_matrix_full = (items[1] == ".true.");
                        }else if(items[0] == "make_graph_widths"){
                            trim(items[1]);
                            AnalysiesCrystaltilts.make_graph_widths = (items[1] == ".true.");
                        }else if(items[0] == "metafile"){
                            trim(items[1]);
                            AnalysiesCrystaltilts.metafile = split(items[1], "\"")[1];
                        }else if(items[0] == "make_an_C1_ta"){
                            trim(items[1]);
                            AnalysiesCrystaltilts.make_an_C1_ta = (items[1] == ".true.");
                        }else if(items[0] == "make_plot_C1_table"){
                            trim(items[1]);
                            AnalysiesCrystaltilts.make_plot_C1_table = (items[1] == ".true.");
                        }

                    }
                }
            }
        }
    }else{
        cout << "Could not open input file on path: " << inFile << endl;

        return 0;
    }

    inputFile.close();

    cout << "Input file read." << endl;

    if(Geometry.imh == 2 and Geometry.imk == 2 and Geometry.iml == 2)
        refra_corr = refra_corrNIST;
    else
        refra_corr = refra_corrPARIS;


    Unit_energy = physical_parameters.Unit_energy;



    root_script = false;

    if(not root_script){
        if(Graph_options.MakeDislin){
            DCS_GUI w;
            w.show();
            return a.exec();
        }
    }else{
        Graph_options.MakeDislin = false;
        //old unimplemented code?
        //Obtain_input_root();
    }


    if(UserSettings.mask_C1 < 0 or UserSettings.mask_C1 > 2){
        cout << "bad input for first crystal mask: " << UserSettings.mask_C1 << endl;
        cout << "value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top" << endl;

        return 0;
    }

    if(UserSettings.mask_C2 < 0 or UserSettings.mask_C2 > 2){
            cout << "bad input for second crystal mask: " << UserSettings.mask_C2 << endl;
            cout << "value of 0 for no mask, 1 for mask on the bottom and 2 for mask on the top" << endl;

            return 0;
    }


    if(not Graph_options.MakeDislin){
        Graph_options.make_graph_profile = false;
        Graph_options.make_image_plates = false;
        AnalysiesCrystaltilts.make_graph_widths = false;
    }

    if(fullenergyspectrum.make_more_lines == 1){
        if(not fullenergyspectrum.Do_background){
            if(fullenergyspectrum.p3_ener + fullenergyspectrum.p2_ener + fullenergyspectrum.p1_ener > 1.0){
                cout << "bad input for lines proportion: " << fullenergyspectrum.p1_ener << " + " << fullenergyspectrum.p2_ener << " + " << fullenergyspectrum.p3_ener << " is greater than 1" << endl;

                return 0;
            }
        }

        reques_energ[0] = fullenergyspectrum.linelamda1;
        reques_energ[1] = fullenergyspectrum.linelamda2;
        reques_energ[2] = fullenergyspectrum.linelamda3;
        reques_energ[3] = fullenergyspectrum.linelamda4;

        reques_width[0] = fullenergyspectrum.naturalwidth1;
        reques_width[1] = fullenergyspectrum.naturalwidth2;
        reques_width[2] = fullenergyspectrum.naturalwidth3;
        reques_width[3] = fullenergyspectrum.naturalwidth4;

    }else if(fullenergyspectrum.make_more_lines == 0){
        reques_energ[0] = linelamda;
        reques_energ[1] = fullenergyspectrum.linelamda2;
        reques_energ[2] = fullenergyspectrum.linelamda3;
        reques_energ[3] = fullenergyspectrum.linelamda4;

        reques_width[0] = naturalwidth;
        reques_width[1] = fullenergyspectrum.naturalwidth2;
        reques_width[2] = fullenergyspectrum.naturalwidth3;
        reques_width[3] = fullenergyspectrum.naturalwidth4;
    }else{
        cout << "Reading input energy spectrum..." << endl;

        Obtain_EnergySpectrum::Read_EnergySpectrum();

        cout << "Input energy spectrum read." << endl;
    }

    if(Geometry.crystal_Si){
        d_lat = a_si_para / sqrt(pow(Geometry.imh, 2) + pow(Geometry.imk, 2) + pow(Geometry.iml, 2));
    }else{
        d_lat = a_Ge_para / sqrt(pow(Geometry.imh, 2) + pow(Geometry.imk, 2) + pow(Geometry.iml, 2));
    }

    if(fullenergyspectrum.make_more_lines == 0 or fullenergyspectrum.make_more_lines == 1){
        if(physical_parameters.Unit_energy == evv[0]){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] < 10.0){
                    cout << "bad input on the energies. requested energy less than 10 eV" << endl;

                    return 0;
                }
            }
        }else if(physical_parameters.Unit_energy == "A"){
            for(int i = 0; i < 4; i++){
                if(reques_energ[i] > 10.0){
                    cout << "bad input on the energies. requested energy more than 10 A" << endl;

                    return 0;
                }
            }
        }else{
            cout << "bad input on the energy unit: " << physical_parameters.Unit_energy << endl;

            return 0;
        }
    }else{
        if(physical_parameters.Unit_energy == "keV"){
            usable = CheckInputSpectrum::CheckSpectrum("eV");

            if(not usable){
                cout << "bad input on the energies. requested energy spectrum will not be visible in output" << endl;

                return 0;
            }
        }else if(physical_parameters.Unit_energy == "eV"){
            usable = CheckInputSpectrum::CheckSpectrum("eV");

            if(not usable){
                cout << "bad input on the energies. requested energy spectrum will not be visible in output" << endl;

                return 0;
            }
        }else if(physical_parameters.Unit_energy == "A"){
            usable = CheckInputSpectrum::CheckSpectrum("A");

            if(not usable){
                cout << "bad input on the energies. requested energy spectrum will not be visible in output" << endl;

                return 0;
            }
        }else{
            cout << "bad input on the energy unit: " << physical_parameters.Unit_energy << endl;

            return 0;
        }
    }


    if(fullenergyspectrum.make_more_lines == 1){
        for(int i = 0; i < 4; i++){
            reques_width[i] = reques_width[i] / 2.0;

            if(physical_parameters.Unit_energy == evv[0]){
                picks[i].lamda = Convert_Ag_minusone_eV / reques_energ[i];
                picks[i].natural_varia = Convert_Ag_minusone_eV * reques_width[i] / (pow(reques_energ[i], 2) - pow(reques_width[i], 2));
            }else{
                picks[i].lamda = reques_energ[i];
                picks[i].natural_varia = reques_width[i];
            }
        }
    }else if(fullenergyspectrum.make_more_lines == 0){
        reques_width[1] = reques_width[1] / 2.0;

        if(physical_parameters.Unit_energy == evv[0]){
            picks[1].lamda = Convert_Ag_minusone_eV / reques_energ[1];
            picks[1].natural_varia = Convert_Ag_minusone_eV * reques_width[1] / (pow(reques_energ[1], 2) - pow(reques_width[1], 2));
        }else{
            picks[1].lamda = reques_energ[1];
            picks[1].natural_varia = reques_width[1];
        }
    }


    gauss_Doop_ev = physical_parameters.gauss_Doop;
    physical_parameters.gauss_Doop = Convert_Ag_minusone_eV * physical_parameters.gauss_Doop / (pow(reques_energ[1], 2) - pow(physical_parameters.gauss_Doop, 2));


    if(Geometry.mode_bragg_geo){
        Double_Crystal_diffraction::Make_Simu();
    }else{
        cout << "unimplemented transmission mode" << endl;
    }

    return 0;
}

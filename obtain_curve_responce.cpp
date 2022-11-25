/*
 * Obtain_curve_responce.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */


#include "obtain_curve_responce.hh"

using namespace std;

extern Geometry Geometry;
extern polarization_parameters polarization_parameters;

extern char File_simu[200];

extern double theta_chk;

vector<double> available_energies;
vector<string> _available_energies;
vector<vector<plotresponc>> plotresponce_mide;

vector<double> min_angle_resp, max_angle_resp;

void Obtain_curve_responce::Read_CurveResponce(){

    bool exist_file;
    double y1d, yud;// theta_dire;

    ifstream pathEnergies;
    ifstream pathFile;

    if(Geometry.mode_bragg_geo){

        char inEnergies[1024] = "";
        strcat(inEnergies, File_simu);
        strcat(inEnergies, "\\Crystal_Profiles");
        strcat(inEnergies, "\\sorted_ens.txt");

        struct stat buffer;
        exist_file = (stat (inEnergies, &buffer) == 0);

        if(exist_file)
            pathEnergies.open(inEnergies);
        else{
            cout << "File \"sorted_ens.txt\" does not exist" << endl;
            throw runtime_error("A file with the available crystal profile energies is required.");
        }


        string ener;
        if(pathEnergies.is_open()){
            while(pathEnergies >> ener){
                _available_energies.push_back(ener);
                available_energies.push_back(stod(ener));
            }
        }

    }else{

        char inFile[223] = "";
        strcat(inFile, File_simu);
        strcat(inFile, "\\Plot_crystall_responce_transm.txt");


        struct stat buffer;
        exist_file = (stat (inFile, &buffer) == 0);

        if(exist_file)
            pathFile.open(inFile);
        else{
            cout << "File \"Plot_crystal_responce_transm.txt\" does not exist" << endl;
            throw runtime_error("A file with crystall responce from XOP in Laue geometry is required");
        }

    }

    double cel_re1, cel_re2;

    int energyIndex = 0;

    for (string ener : _available_energies) {
        ifstream pathFile_p;

        char inFile_p[1024] = "";
        strcat(inFile_p, File_simu);
        strcat(inFile_p, "\\Crystal_Profiles");
        strcat(inFile_p, (string("\\") + ener + string("keV_p")).c_str());

        struct stat buffer1;
        exist_file = (stat (inFile_p, &buffer1) == 0);

        if(exist_file){
            pathFile_p.open(inFile_p);
            //cout << "Reading file: " << ener << "keV_p" << endl;
        }else{
            cout << "File " << inFile_p << " does not exist" << endl;
            throw runtime_error("Expected a crystall responce from XOP in bragg geometry for energy" + ener);
        }

        vector<plotresponc> responce;
        if(pathFile_p.is_open()){
            while(pathFile_p >> cel_re1 >> cel_re2){
                plotresponc tmp;
                tmp.degree = cel_re1 / one_micro;
                tmp.reflecti_total = 0;
                tmp.reflecti_two_deriv = 0;
                tmp.reflecti_total_s = 0;
                tmp.reflecti_two_deriv_s = 0;
                tmp.reflecti_total_p = cel_re2;
                tmp.reflecti_two_deriv_p = 0;
                responce.push_back(tmp);
            }

            pathFile_p.close();
        }

        ifstream pathFile_s;

        char inFile_s[1024] = "";
        strcat(inFile_s, File_simu);
        strcat(inFile_s, "\\Crystal_Profiles");
        strcat(inFile_s, (string("\\") + ener + string("keV_s")).c_str());

        struct stat buffer2;
        exist_file = (stat (inFile_s, &buffer2) == 0);

        if(exist_file){
            pathFile_s.open(inFile_s);
            //cout << "Reading file: " << ener << "keV_p" << endl;
        }else{
            cout << "File " << inFile_s << " does not exist" << endl;
            throw runtime_error("Expected a crystall responce from XOP in bragg geometry for energy" + ener);
        }

        int index = 0;
        if(pathFile_s.is_open()){
            while(pathFile_s >> cel_re1 >> cel_re2){
                responce[index].reflecti_total = responce[index].reflecti_total_p + cel_re2;
                responce[index].reflecti_total_s = cel_re2;

                index++;
            }

            pathFile_s.close();
        }

        plotresponce_mide.push_back(responce);

        min_angle_resp.push_back(plotresponce_mide[energyIndex][0].degree);
        max_angle_resp.push_back(plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree);

        y1d = (plotresponce_mide[energyIndex][1].reflecti_total - plotresponce_mide[energyIndex][0].reflecti_total) / (plotresponce_mide[energyIndex][1].degree - plotresponce_mide[energyIndex][0].degree);
        yud = (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].reflecti_total - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].reflecti_total) / (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].degree);

        //Total
        vector<double> x, y, y2;

        for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
            x.push_back(plotresponce_mide[energyIndex][i].degree);
            y.push_back(plotresponce_mide[energyIndex][i].reflecti_total);
        }

        y2 = spline(x, y, y1d, yud);

        for(unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++){
            plotresponce_mide[energyIndex][i].reflecti_two_deriv = y2[i];
        }


        y1d = (plotresponce_mide[energyIndex][1].reflecti_total_s - plotresponce_mide[energyIndex][0].reflecti_total_s) / (plotresponce_mide[energyIndex][1].degree - plotresponce_mide[energyIndex][0].degree);
        yud = (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].reflecti_total_s - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].reflecti_total_s) / (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].degree);

        //S Polarization
        vector<double> x_s, y_s, y2_s;

        for(unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++){
            x_s.push_back(plotresponce_mide[energyIndex][i].degree);
            y_s.push_back(plotresponce_mide[energyIndex][i].reflecti_total_s);
        }

        y2_s = spline(x_s, y_s, y1d, yud);

        for(unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++){
            plotresponce_mide[energyIndex][i].reflecti_two_deriv_s = y2_s[i];
        }


        y1d = (plotresponce_mide[energyIndex][1].reflecti_total_p - plotresponce_mide[energyIndex][0].reflecti_total_p) / (plotresponce_mide[energyIndex][1].degree - plotresponce_mide[energyIndex][0].degree);
        yud = (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].reflecti_total_p - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].reflecti_total_p) / (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].degree);

        //P Polarization
        vector<double> x_p, y_p, y2_p;

        for(unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++){
            x_p.push_back(plotresponce_mide[energyIndex][i].degree);
            y_p.push_back(plotresponce_mide[energyIndex][i].reflecti_total_p);
        }

        y2_p = spline(x_p, y_p, y1d, yud);

        for(unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++){
            plotresponce_mide[energyIndex][i].reflecti_two_deriv_p = y2_p[i];
        }

        energyIndex++;

    }

}

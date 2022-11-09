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

vector<plotresponc> plotresponce_mide;

double min_angle_resp, max_angle_resp;

void Obtain_curve_responce::Read_CurveResponce(){

    bool exist_file;
    double y1d, yud;// theta_dire;

    ifstream pathFile;

    if(Geometry.mode_bragg_geo){

        char inFile[223] = "";
        strcat(inFile, File_simu);
        strcat(inFile, "\\Plot_crystall_responce_600_600.txt");


        struct stat buffer;
        exist_file = (stat (inFile, &buffer) == 0);

        if(exist_file)
            pathFile.open(inFile);
        else{
            cout << "File \"Plot_crystal_responce_600_600.txt\" does not exist" << endl;
            throw runtime_error("A file with crystall responce from XOP in bragg geometry is required");
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

    if(pathFile.is_open()){
        while(pathFile >> cel_re1 >> cel_re2){
            plotresponc tmp;
            tmp.degree = cel_re1 / one_micro;
            tmp.reflecti_total = cel_re2;
            tmp.reflecti_two_deriv = 0;
            tmp.reflecti_total_s = 0;
            tmp.reflecti_two_deriv_s = 0;
            tmp.reflecti_total_p = 0;
            tmp.reflecti_two_deriv_p = 0;
            plotresponce_mide.push_back(tmp);
        }

        pathFile.close();
    }

    min_angle_resp = plotresponce_mide[0].degree;
    max_angle_resp = plotresponce_mide[plotresponce_mide.size() - 1].degree;

    y1d = (plotresponce_mide[1].reflecti_total - plotresponce_mide[0].reflecti_total) / (plotresponce_mide[1].degree - plotresponce_mide[0].degree);
    yud = (plotresponce_mide[plotresponce_mide.size() - 1].reflecti_total - plotresponce_mide[plotresponce_mide.size() - 2].reflecti_total) / (plotresponce_mide[plotresponce_mide.size() - 1].degree - plotresponce_mide[plotresponce_mide.size() - 2].degree);

    std::vector<double> x, y, y2;

    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
        x.push_back(plotresponce_mide[i].degree);
        y.push_back(plotresponce_mide[i].reflecti_total);
    }

    y2 = spline(x, y, y1d, yud);

    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
        plotresponce_mide[i].reflecti_two_deriv = y2[i];
    }


    if(polarization_parameters.mka_poli){
        //theta_dire = 2 * (90 - theta_chk);

        char inFile[223] = "";
        strcat(inFile, File_simu);
        strcat(inFile, "\\Plot_crystall_responce_s_f.txt");


        struct stat buffer;
        exist_file = (stat (inFile, &buffer) == 0);

        if(exist_file)
            pathFile.open(inFile);
        else{
            cout << "File \"Plot_crystal_responce_s_f.txt\" does not exist" << endl;
            throw runtime_error("A file with crystall responce from XOP in bragg geometry is required for s poli only");
        }

        if(pathFile.is_open()){
            while(pathFile >> cel_re1 >> cel_re2){

                vector<double> x;

                for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                    x.push_back(plotresponce_mide[i].degree);
                }

                int ind = find(x.begin(), x.end(), cel_re1 / one_micro) - x.begin();

                plotresponce_mide[ind].reflecti_total_s = cel_re2;
            }

            pathFile.close();
        }

        char inFile2[223] = "";
        strcat(inFile2, File_simu);
        strcat(inFile2, "\\Plot_crystall_responce_p_f.txt");


        exist_file = (stat (inFile2, &buffer) == 0);

        if(exist_file)
            pathFile.open(inFile2);
        else{
            cout << "File \"Plot_crystal_responce_p_f.txt\" does not exist" << endl;
            throw runtime_error("A file with crystall responce from XOP in bragg geometry is required for p poli only");
        }

        if(pathFile.is_open()){
            while(pathFile >> cel_re1 >> cel_re2){

                vector<double> x;

                for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                    x.push_back(plotresponce_mide[i].degree);
                }

                int ind = find(x.begin(), x.end(), cel_re1 / one_micro) - x.begin();

                plotresponce_mide[ind].reflecti_total_p = cel_re2;
            }

            pathFile.close();
        }



        y1d = (plotresponce_mide[1].reflecti_total_s - plotresponce_mide[0].reflecti_total_s) / (plotresponce_mide[1].degree - plotresponce_mide[0].degree);
        yud = (plotresponce_mide[plotresponce_mide.size() - 1].reflecti_total_s - plotresponce_mide[plotresponce_mide.size() - 2].reflecti_total_s) / (plotresponce_mide[plotresponce_mide.size() - 1].degree - plotresponce_mide[plotresponce_mide.size() - 2].degree);

        vector<double> x, y, y2;

        for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
            x.push_back(plotresponce_mide[i].degree);
            y.push_back(plotresponce_mide[i].reflecti_total_s);
        }

        y2 = spline(x, y, y1d, yud);

        for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
            plotresponce_mide[i].reflecti_two_deriv_s = y2[i];
        }





        y1d = (plotresponce_mide[1].reflecti_total_p - plotresponce_mide[0].reflecti_total_p) / (plotresponce_mide[1].degree - plotresponce_mide[0].degree);
        yud = (plotresponce_mide[plotresponce_mide.size() - 1].reflecti_total_p - plotresponce_mide[plotresponce_mide.size() - 2].reflecti_total_p) / (plotresponce_mide[plotresponce_mide.size() - 1].degree - plotresponce_mide[plotresponce_mide.size() - 2].degree);


        for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
            x.push_back(plotresponce_mide[i].degree);
            y.push_back(plotresponce_mide[i].reflecti_total_p);
        }

        y2 = spline(x, y, y1d, yud);

        for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
            plotresponce_mide[i].reflecti_two_deriv_p = y2[i];
        }

    }


}

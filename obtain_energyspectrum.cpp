/*
 * Obtain_EnergySpectrum.cpp
 *
 *  Created on: Oct 29, 2020
 *      Author: danip
 */

#include "obtain_energyspectrum.hh"

using namespace std;

extern char File_simu[200];
extern string Unit_energy;

std::vector<energy_gen> Energy_spec;

void Obtain_EnergySpectrum::Read_EnergySpectrum(){

    bool exist_file;
    double y1d, yud;
    double x1d, xud;
    int mult;


    ifstream pathFile;

    char inFile[223] = "";
    strcat(inFile, File_simu);
    strcat(inFile, "\\Energy_spectrum_600.txt");


    struct stat buffer;
    exist_file = (stat (inFile, &buffer) == 0);

    if(exist_file){
        pathFile.open(inFile);
    }else{
        throw runtime_error("input energy spectrum file \"Energy_spectrum_600.txt\" not found");
    }

    if(Unit_energy == "keV")
        mult = 1000;
    else if(Unit_energy == "eV")
        mult = 1;
    else
        mult = 1;

    double cel_re1, cel_re2;
    double cum = 0;

    if(pathFile.is_open()){
        while(pathFile >> cel_re1 >> cel_re2){
            cum += cel_re2;

            energy_gen tmp;
            tmp.lamda = cel_re1 * mult;
            tmp.intensity = cel_re2;
            tmp.cum_int = cum;
            tmp.intensity_two_deriv = 0;
            tmp.lambda_two_deriv = 0;
            Energy_spec.push_back(tmp);
        }

        pathFile.close();
    }

    y1d = (Energy_spec[1].intensity - Energy_spec[0].intensity) / (Energy_spec[1].lamda - Energy_spec[0].lamda);
    yud = (Energy_spec[Energy_spec.size() - 1].intensity - Energy_spec[Energy_spec.size() - 2].intensity) / (Energy_spec[Energy_spec.size() - 1].lamda - Energy_spec[Energy_spec.size() - 2].lamda);

    std::vector<double> x, y, yc, ycn, x2, y2;

    for(unsigned int i = 0; i < Energy_spec.size(); i++){
        x.push_back(Energy_spec[i].lamda);
        y.push_back(Energy_spec[i].intensity);
        yc.push_back(Energy_spec[i].cum_int);
    }

    double min_cum = *min_element(yc.begin(), yc.end());
    double max_cum = *max_element(yc.begin(), yc.end()) - min_cum;

    for(unsigned int i = 0; i < Energy_spec.size(); i++){
        Energy_spec[i].cum_int -= min_cum;
        Energy_spec[i].cum_int /= max_cum;
        ycn.push_back(Energy_spec[i].cum_int);
    }

    x1d = (Energy_spec[1].lamda - Energy_spec[0].lamda) / (Energy_spec[1].cum_int - Energy_spec[0].cum_int);
    xud = (Energy_spec[Energy_spec.size() - 1].lamda - Energy_spec[Energy_spec.size() - 2].lamda) / (Energy_spec[Energy_spec.size() - 1].cum_int - Energy_spec[Energy_spec.size() - 2].cum_int);

    y2 = spline(x, y, y1d, yud);
    x2 = spline(ycn, x, x1d, xud);

    for(unsigned int i = 0; i < Energy_spec.size(); i++){
        Energy_spec[i].intensity_two_deriv = y2[i];
        Energy_spec[i].lambda_two_deriv = x2[i];
    }

}

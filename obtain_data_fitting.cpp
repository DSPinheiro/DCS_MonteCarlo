/*
 * Obtain_data_fitting.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "obtain_data_fitting.hh"
#include <iostream>

using namespace std;


void Obtain_data_fitting::FitData(
    int numbins,
    double angle_para,
    int toint_para,
    double angle_anti,
    int toint_anti
){

    if(Data_Fit_anti.size() < (unsigned int)numbins)
        Data_Fit_anti.resize(PlotParametersInput.nubins);
    if(Data_Fit_para.size() < (unsigned int)numbins)
        Data_Fit_para.resize(PlotParametersInput.nubins);

    Data_Fit_para.at(numbins - 1).degree = angle_para;
    Data_Fit_para.at(numbins - 1).reflecti_total = (double)(toint_para + 10);
    Data_Fit_para.at(numbins - 1).reflecti_two_deriv = sqrt((double)(toint_para + 10));

    Data_Fit_anti.at(numbins - 1).degree = angle_anti;
    Data_Fit_anti.at(numbins - 1).reflecti_total = (double)(toint_anti + 10);
    Data_Fit_anti.at(numbins - 1).reflecti_two_deriv = sqrt((double)(toint_anti + 10));
}

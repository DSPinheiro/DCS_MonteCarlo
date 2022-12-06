/*
 * Get_new_temp_for_bin.cpp
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */


#include "get_new_temp_for_bin.hh"

using namespace std;


double Get_new_temp_for_bins::getNewTemp(int bin_tem, int bin_fas, double pha_tem){

    if(bin_fas > TemperatureParametersInput.TT_tempera){
        pha_tem = 2 * M_PI * ((double)rand() / RAND_MAX);
        bin_fas = 0;
    }else
        bin_fas++;


    if(TemperatureParametersInput.TT_tempera == 0)
        cout << "Warning: Value for TT:tempera is 0" << endl;


    return TemperatureParametersInput.AA_tempera * cos(2 * M_PI * bin_tem / TemperatureParametersInput.TT_tempera + pha_tem);

}

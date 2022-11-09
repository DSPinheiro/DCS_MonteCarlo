/*
 * Intensity_source.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "intensitysource.hh"

using namespace std;

extern fullenergyspectrum fullenergyspectrum;
extern physical_parameters physical_parameters;

extern ofstream gener_out;

extern pick picks[5];

extern vector<energy_gen> Energy_spec;

extern double min_angle_resp, max_angle_resp;

double IntensitySource::getEnergy(double a_lamds_uni, double db_lamds_uni, double tw_d){

    double p1, p2, natur_li, pm1, pm2, pm3, pm4, hit, rnd_inten, energy_t;
    int I_picks;

    if(fullenergyspectrum.make_more_lines == 0)
        I_picks = 2;
    else if(fullenergyspectrum.make_more_lines == 1){
        if(fullenergyspectrum.Do_background){
            pm1 = 0.1875;
            pm2 = 0.34375;
            pm3 = 0.421875;
            pm4 = 0.5;
        }else{
            pm1 = fullenergyspectrum.p1_ener;
            pm2 = pm1 + fullenergyspectrum.p2_ener;
            pm3 = pm2 + fullenergyspectrum.p3_ener;
            pm4 = 1;
        }


        p1 = ((double)rand() / RAND_MAX);


        if(p1 < pm1)
            I_picks = 1;
        else{
            if(p1 < pm2)
                I_picks = 2;
            else{
                if(p1 < pm3)
                    I_picks = 3;
                else{
                    if(p1 < pm4)
                        I_picks = 4;
                    else
                        I_picks = 5;
                }
            }
        }
    }else{
        rnd_inten = (double)rand() / RAND_MAX;
        vector<double> x, y, x2;

        for(unsigned int i = 0; i < Energy_spec.size(); i++){
            x.push_back(Energy_spec[i].lamda);
            y.push_back(Energy_spec[i].cum_int);
            x2.push_back(Energy_spec[i].lambda_two_deriv);
        }

        energy_t = splint_te(y, x, x2, rnd_inten);

        return Convert_Ag_minusone_eV / energy_t;

    }

    if(fullenergyspectrum.make_more_lines == 0 or fullenergyspectrum.make_more_lines == 1){
        if(I_picks == 5){
            p2 = ((double)rand() / RAND_MAX);
            return a_lamds_uni + db_lamds_uni * p2;
        }else{
            hit = -1;

            while(hit < 0 or hit > tw_d){
                p1 = ((double)rand() / RAND_MAX) * M_PI;

                natur_li = picks[I_picks - 1].natural_varia;

                hit = picks[I_picks - 1].lamda + natur_li * tan(p1);
            }
        }

        hit = Box(physical_parameters.gauss_Doop, hit);
        return hit;

    }else{
        throw runtime_error("Error in intensity_source: energy could not be generated from input");
    }

}

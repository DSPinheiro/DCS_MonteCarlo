/*
 * Gaussi_rockin.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "gaussi_rockin.hh"


using namespace std;

extern polarization_parameters polarization_parameters;

extern vector<plotresponc> plotresponce_mide;

extern double min_angle_resp, max_angle_resp;


bool Gaussi_rockin::getReflection(double angle, double tetabra, bool type_crystal, bool poli_p){

    double p, dif, inte;

    dif = angle - tetabra;

    if(dif < min_angle_resp){
        return false;
    }else{
        if(dif < max_angle_resp){
            if(type_crystal and polarization_parameters.mka_poli){
                if(poli_p){
                    vector<double> x, y, y2;

                    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                        x.push_back(plotresponce_mide[i].degree);
                        y.push_back(plotresponce_mide[i].reflecti_total_p);
                        y2.push_back(plotresponce_mide[i].reflecti_two_deriv_p);
                    }

                    inte = splint_te(x, y, y2, dif);
                }else{
                    vector<double> x, y, y2;

                    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                        x.push_back(plotresponce_mide[i].degree);
                        y.push_back(plotresponce_mide[i].reflecti_total_s);
                        y2.push_back(plotresponce_mide[i].reflecti_two_deriv_s);
                    }

                    inte = splint_te(x, y, y2, dif);
                }
            }else{
                vector<double> x, y, y2;

                for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                    x.push_back(plotresponce_mide[i].degree);
                    y.push_back(plotresponce_mide[i].reflecti_total);
                    y2.push_back(plotresponce_mide[i].reflecti_two_deriv);
                }

                inte = splint_te(x, y, y2, dif);
            }

            p = ((double)rand() / RAND_MAX);

            if(p < inte)
                return true;
            else
                return false;

        }else{
            return false;
        }
    }

}

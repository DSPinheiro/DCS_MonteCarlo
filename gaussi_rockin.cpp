/*
 * Gaussi_rockin.cpp
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */


#include "gaussi_rockin.hh"


using namespace std;

extern polarization_parameters polarization_parameters;

extern vector<double> available_energies;
extern vector<vector<plotresponc>> plotresponce_mide;

extern vector<double> min_angle_resp, max_angle_resp;


bool Gaussi_rockin::getReflection(double angle, double tetabra, double lamda, bool type_crystal, bool poli_p){

    double p, dif, inte, inte1, inte2;

    dif = angle - tetabra;

    double energy = Convert_Ag_minusone_eV / lamda;

    int energy_resp_index;

    int index = 0;
    for(double energ : available_energies)
    {
        if (energ > energy)
        {
            energy_resp_index = index - 1;
            break;
        }
        index++;
    }

    double energy_min_angle_resp, energy_max_angle_resp;
    energy_min_angle_resp = max(min_angle_resp[energy_resp_index], min_angle_resp[energy_resp_index + 1]);
    energy_max_angle_resp = min(max_angle_resp[energy_resp_index], max_angle_resp[energy_resp_index + 1]);

    if(dif < energy_min_angle_resp){
        return false;
    }else{
        if(dif < energy_max_angle_resp){
            if(type_crystal and polarization_parameters.mka_poli){
                if(poli_p){
                    vector<double> x1, y1, y12;

                    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                        x1.push_back(plotresponce_mide[energy_resp_index][i].degree);
                        y1.push_back(plotresponce_mide[energy_resp_index][i].reflecti_total_p);
                        y12.push_back(plotresponce_mide[energy_resp_index][i].reflecti_two_deriv_p);
                    }

                    inte1 = splint_te(x1, y1, y12, dif);

                    vector<double> x2, y2, y22;

                    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                        x2.push_back(plotresponce_mide[energy_resp_index + 1][i].degree);
                        y2.push_back(plotresponce_mide[energy_resp_index + 1][i].reflecti_total_p);
                        y22.push_back(plotresponce_mide[energy_resp_index + 1][i].reflecti_two_deriv_p);
                    }

                    inte2 = splint_te(x2, y2, y22, dif);

                    inte = ((inte2 - inte1) / (available_energies[energy_resp_index + 1] - available_energies[energy_resp_index])) * (energy - available_energies[energy_resp_index]) + inte1;
                }else{
                    vector<double> x1, y1, y12;

                    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                        x1.push_back(plotresponce_mide[energy_resp_index][i].degree);
                        y1.push_back(plotresponce_mide[energy_resp_index][i].reflecti_total_s);
                        y12.push_back(plotresponce_mide[energy_resp_index][i].reflecti_two_deriv_s);
                    }

                    inte1 = splint_te(x1, y1, y12, dif);

                    vector<double> x2, y2, y22;

                    for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                        x2.push_back(plotresponce_mide[energy_resp_index + 1][i].degree);
                        y2.push_back(plotresponce_mide[energy_resp_index + 1][i].reflecti_total_s);
                        y22.push_back(plotresponce_mide[energy_resp_index + 1][i].reflecti_two_deriv_s);
                    }

                    inte2 = splint_te(x2, y2, y22, dif);

                    inte = ((inte2 - inte1) / (available_energies[energy_resp_index + 1] - available_energies[energy_resp_index])) * (energy - available_energies[energy_resp_index]) + inte1;
                }
            }else{
                vector<double> x1, y1, y12;

                for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                    x1.push_back(plotresponce_mide[energy_resp_index][i].degree);
                    y1.push_back(plotresponce_mide[energy_resp_index][i].reflecti_total);
                    y12.push_back(plotresponce_mide[energy_resp_index][i].reflecti_two_deriv);
                }

                inte1 = splint_te(x1, y1, y12, dif);

                vector<double> x2, y2, y22;

                for(unsigned int i = 0; i < plotresponce_mide.size(); i++){
                    x2.push_back(plotresponce_mide[energy_resp_index + 1][i].degree);
                    y2.push_back(plotresponce_mide[energy_resp_index + 1][i].reflecti_total);
                    y22.push_back(plotresponce_mide[energy_resp_index + 1][i].reflecti_two_deriv);
                }

                inte2 = splint_te(x2, y2, y22, dif);

                inte = ((inte2 - inte1) / (available_energies[energy_resp_index + 1] - available_energies[energy_resp_index])) * (energy - available_energies[energy_resp_index]) + inte1;
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

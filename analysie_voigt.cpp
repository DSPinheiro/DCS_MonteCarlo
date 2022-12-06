/*
 * Analysie_Voigt.cpp
 *
 *  Created on: Nov 4, 2020
 *      Author: danip
 */

#include "analysie_voigt.hh"

using namespace std;


void Analysie_Voigt::analyse(){
    
    double p_para, p_anti, dif, wave, energ_absorb, energy_exact, corre_30cel, energ, d_lat_2;
    double energy_obtain;

    p_para = a_para_voig[3];
    p_anti = a_anti_voig[3];

    if(GeometryInput.mode_bragg_geo)
        dif = (p_anti - p_para) / 2 + 90 - vert_div_corr - refra_corr + shape_corr;
    else
        dif = (p_anti - p_para) / 2 - vert_div_corr;



    dif *= M_PI / 180;

    energ_absorb = 0;

    corre_30cel = 0;

    if(FullEnergySpectrumInput.make_more_lines == 1)
        energy_exact = Convert_Ag_minusone_eV / picks[2].lamda;
    else if(FullEnergySpectrumInput.make_more_lines == 0)
        energy_exact = Convert_Ag_minusone_eV / picks[1].lamda;
    else
        energy_exact = middle_energy;


    d_lat_2 = Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_anti);

    wave = 2 * d_lat_2 * sin(dif);

    energ = Convert_Ag_minusone_eV / wave - energ_absorb;

    energy_obtain = energ;


    if(!AnalysiesCrystaltiltsInput.make_an_C1_ta){
        gener_out << endl;
        gener_out << "-----------------------------------------" << endl;
        gener_out << endl;
        gener_out << " Vertical correction: " << vert_div_corr << endl;
        gener_out << " Refraction correction: " << refra_corr << endl;
        gener_out << endl;
        gener_out << "Parameter analysies " << endl;
        gener_out << endl;
        gener_out << "ration widths anti/para: " << FWMH_V_anti / FWMH_V_para << endl;
        gener_out << endl;
        gener_out << "Angular difference: " << dif << endl;
        gener_out << "wavelength: " << wave << " A" << endl;
        gener_out << "Energy obtain: " << energ << " eV" << endl;

        gener_out << "Energy input: " << energy_exact << " eV" << endl;
        gener_out << "difference obtain and input: " << (energ - energy_exact) * 1000 << " MeV" << endl;


        if(FullEnergySpectrumInput.make_more_lines <= 1){
            cout << endl;
            cout << "-----------------------------------------" << endl;
            cout << endl;
            cout << "Parameter analysies " << endl;
            cout << endl;
            cout << "ration widths anti/para: " << FWMH_V_anti / FWMH_V_para << endl;
            cout << endl;
            cout << "Angular difference: " << dif << endl;
            cout << "wavelength: " << wave << " A" << endl;
            cout << "Energy obtain: " << energ << " eV" << endl;

            cout << "Energy input: " << energy_exact << " eV" << endl;
            cout << "difference obtain and input: " << (energ - energy_exact) * 1000 << " MeV" << endl;
        }else{
            cout << endl;
            cout << "-----------------------------------------" << endl;
            cout << endl;
            cout << "Parameter analysies (unreliable for an input energy spectrum)" << endl;
            cout << endl;
            cout << "ration widths anti/para: " << FWMH_V_anti / FWMH_V_para << endl;
            cout << endl;
            cout << "Angular difference: " << dif << endl;
            cout << "wavelength: " << wave << " A" << endl;
            cout << "Energy obtain: " << energ << " eV" << endl;

            cout << "Energy input: " << energy_exact << " eV" << endl;
            cout << "difference obtain and input: " << (energ - energy_exact) * 1000 << " MeV" << endl;
        }

        if(root_script){
            //old script writing
        }

    }

}

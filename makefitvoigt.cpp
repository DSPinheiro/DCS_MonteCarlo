/*
 * MakefitVoigt.cpp
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */


#include "makefitvoigt.hh"

using namespace std;



void MakefitVoigt::fit(bool Parallel){

    vector<double> x, y, sig;

    double gues[MA], a[MA], alamda, a_error_anti_voig[MA], FWMH_V, chisq, ochisq;

    vector<vector<double>> covar(MA, vector<double>(MA)), alpha(MA, vector<double>(MA));

    int NPT, ia[MA], npt_m_deg, k, itst;

    string name_varia[MA];

    NPT = PlotParametersInput.nubins;

    UserSettingsInput.TrueVoigt = true;

    if(!AnalysiesCrystaltiltsInput.make_an_C1_ta){
        gener_out << "------------------------------------" << endl;
        gener_out << endl;
        gener_out << " Fitting analysies" << endl;
        gener_out << endl;

        cout << "------------------------------------" << endl;
        cout << endl;
        cout << " Fitting analysies" << endl;
        cout << endl;

    }

    if(Parallel){
        for(int i = 0; i < PlotParametersInput.nubins; i++){
            x.push_back(Data_Fit_para[i].degree);
            y.push_back(Data_Fit_para[i].reflecti_total);
            sig.push_back(Data_Fit_para[i].reflecti_two_deriv);
        }

        gues[1] = amplitu_con_para;
        gues[4] = const_back_para;
        gues[3] = peak_posi_para;
        gues[0] = width_Gaus_para;
        gues[2] = width_Lore_para;

        ia[1] = do_amplitu_con_para;
        ia[4] = do_const_back_para;
        ia[3] = do_firstcryst;
        ia[0] = do_Gwidth_para;
        ia[2] = do_Lwidth_para;

        name_varia[1] = "Amplitude of parallel-";
        name_varia[4] = "Background of parallel-";
        name_varia[3] = "position of peak parallel-";
        name_varia[0] = "Gauss width of peak parallel-";
        name_varia[2] = "Lorentz with of peak parallel-";
    }else{
        for(int i = 0; i < PlotParametersInput.nubins; i++){
            x.push_back(Data_Fit_anti[i].degree);
            y.push_back(Data_Fit_anti[i].reflecti_total);
            sig.push_back(Data_Fit_anti[i].reflecti_two_deriv);
        }

        gues[1] = amplitu_con_anti;
        gues[4] = const_back_anti;
        gues[3] = peak_posi_anti;
        gues[0] = width_Gaus_anti;
        gues[2] = width_Lore_anti;

        ia[1] = do_amplitu_con_anti;
        ia[4] = do_const_back_anti;
        ia[3] = do_firstcryst_anti;
        ia[0] = do_Gwidth_anti;
        ia[2] = do_Lwidth_anti;

        name_varia[1] = "Amplitude of antiparallel-";
        name_varia[4] = "Background of antiparallel-";
        name_varia[3] = "position of peak antiparallel-";
        name_varia[0] = "Gauss width of peak antiparallel-";
        name_varia[2] = "Lorentz with of peak antiparallel-";
    }

    for(int i = 0; i < MA; i++){
        a[i] = gues[i];
        npt_m_deg = NPT - ia[i];
    }

    k = 1;
    alamda = -1;

    if(UserSettingsInput.TrueVoigt){
        mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Voig, alamda);
    }else{
        mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Pseud, alamda);
    }

    itst = 0;

    while(itst < 6){
        k++;

        ochisq = chisq;

        if(UserSettingsInput.TrueVoigt){
            mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Voig, alamda);
        }else{
            mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Pseud, alamda);
        }

        if(chisq > ochisq)
            itst = 0;
        else if(abs((ochisq - chisq) / chisq) < 10E-6)
            itst++;

        if(abs(ochisq - chisq) > 10)
            itst = 0;
    }

    alamda = 0;

    if(UserSettingsInput.TrueVoigt){
        mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Voig, alamda);
    }else{
        mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Pseud, alamda);
    }
    cout << "skgnkdbgnkdg" << endl;
    for(int i = 0; i < MA; i++){
        gener_out << name_varia[i] << "\t" << a[i] << endl;
        cout << name_varia[i] << "\t" << a[i] << endl;
        a_error_anti_voig[i] = sqrt(covar[i][i] * npt_m_deg);
    }

    FWMH_V = c1 * a[2] + sqrt(c2 * pow(a[2], 2) + pow(a[0], 2));
    if(UserSettingsInput.TrueVoigt){
        if(Parallel)
            FWMH_V_para = FWMH_V;
        else
            FWMH_V_anti = FWMH_V;
    }else{
        if(Parallel)
            FWMH_V_para = a[0];
        else
            FWMH_V_anti = a[0];
    }

    gener_out << "FWMH-\t" << FWMH_V << endl;
    gener_out << endl;


    cout << "FWMH-\t" << FWMH_V << endl;
    cout << endl;



    gener_out << endl;
    gener_out << "Reduced Chi-squared:\t" << chisq / npt_m_deg << endl;
    gener_out << endl;
    gener_out << "Uncertainties:" << endl;
    gener_out << sqrt(covar[0][0]) << "\t" << MA << endl;


    if(Parallel)
        copy(begin(a), end(a), begin(a_para_voig));
    else
        copy(begin(a), end(a), begin(a_anti_voig));


}

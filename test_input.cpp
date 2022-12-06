/*
 * Test_input.cpp
 *
 *  Created on: Oct 31, 2020
 *      Author: danip
 */

#include "test_input.hh"

using namespace std;



void Test_input::test_In(){

    double b_anti_pick;
    
    b_anti_pick = - 2 * (-teta_crys1 + theta_chk);

    double tetaref, teta_table_rad, xsi_rad, c1, C2_para, C2_anti, tan_e2, cos_e2, cos_e, esti_para, esti_anti, angle_dif, energy_exact, wave, energ;
    double low_bound_angl_para, high_bound_angl_para, low_bound_angl_anti, high_bound_angl_anti, Maxi_angl_para, Mini_angl_para, Maxi_angl_anti, Mini_angl_anti, var_temp;

    tetaref = 90 - teta_crys1;
    tetaref *= convrad;
    teta_table_rad = GeoParametersInput.teta_table * convrad;
    xsi_rad = GeoParametersInput.xsi * convrad;

    if(!UserSettingsInput.Simple_simu){
        xsi_rad = -atan((GeolengthelementsInput.S_shi_ver_A - GeolengthelementsInput.S_shi_ver_B) / GeoParapathlengthsInput.LT_aper);
        GeoParametersInput.xsi = xsi_rad * convdeg;
    }

    if(CurveVerticalTiltInput.make_CurveTilt){
        c1 = ObtainVertical::ObtainVert(1, 0);

        C2_para = ObtainVertical::ObtainVert(2, teta_crys1);
        C2_anti = ObtainVertical::ObtainVert(2, -teta_crys1);
    }else{
        c1 = GeoParametersInput.tilt_C1 * convrad;
        C2_para = GeoParametersInput.tilt_C2 * convrad;
        C2_anti = GeoParametersInput.tilt_C2 * convrad;
    }


    tan_e2 = tan(tetaref) / 2;
    cos_e2 = 2 * cos(2 * tetaref);
    cos_e = cos(tetaref);


    esti_para = convdeg * tan_e2 * (pow(C2_para, 2) + 4 * C2_para * c1 + 3 * pow(c1, 2)) - convdeg * xsi_rad * (C2_para + c1) / cos_e;
    esti_anti = convdeg * tan_e2 * (pow(C2_anti, 2) + 4 * C2_anti * c1 + pow(c1, 2) * (1 - 2 * cos_e2)) + convdeg * tan_e2 * (2 * pow(xsi_rad, 2)) - convdeg * xsi_rad * (C2_anti + c1 * (1 - cos_e2)) / cos_e;

    if(!UserSettingsInput.Simple_simu){
        esti_para -= convdeg * (2 * tetaref - teta_table_rad);
        esti_anti += convdeg * (2 * tetaref - teta_table_rad);
    }

    esti_anti += b_anti_pick + 2 * refra_corr + 2 * vert_div_corr;

    if(UserSettingsInput.see_para){
        gener_out << "The estimated angle of peak" << endl;
        gener_out << "taking into account crystal tilts and effective misalignement" << endl;

        cout << "The estimated angle of peak" << endl;
        cout << "taking into account crystal tilts and effective misalignement" << endl;

        ang_para_pre = teta_crys1 - esti_para;

        gener_out << " in parallel is:					" << ang_para_pre << " deg" << endl;
        gener_out << " or, in turn of reference angle:	" << - esti_para << " deg" << endl;
        gener_out << endl;

        cout << " in parallel is:					" << ang_para_pre << " deg" << endl;
        cout << " or, in turn of reference angle:	" << - esti_para << " deg" << endl;
        cout << endl;

    }

    if(UserSettingsInput.see_anti){
        gener_out << "The estimated angle of peak" << endl;
        gener_out << "taking into account crystal tilts and effective misalignement" << endl;

        cout << "The estimated angle of peak" << endl;
        cout << "taking into account crystal tilts and effective misalignement" << endl;

        ang_anti_pre = esti_anti - teta_crys1;

        gener_out << " in antiparallel is:				" << ang_anti_pre << " deg" << endl;
        gener_out << " or, in turn of reference angle:	" << - esti_anti << " deg" << endl;
        gener_out << endl;

        cout << " in antiparallel is:				" << ang_anti_pre << " deg" << endl;
        cout << " or, in turn of reference angle:	" << - esti_anti << " deg" << endl;
        cout << endl;

    }


    if(UserSettingsInput.see_para && UserSettingsInput.see_anti){
        angle_dif = (ang_anti_pre - ang_para_pre) / 2 + 90 - refra_corr - vert_div_corr;

        angle_dif *= convrad;
        energy_exact = Convert_Ag_minusone_eV / picks[1].lamda;
        wave = 2 * d_lat * sin(angle_dif);
        energ = Convert_Ag_minusone_eV / wave;

        gener_out << endl;
        gener_out << " Estimated energy" << endl;
        gener_out << " without crystal tilts and effective misalignement: " << energy_exact << endl;
        gener_out << endl;
        gener_out << " with crystal tilts and effective misalignement: " << energ << endl;
        gener_out << endl;
        gener_out << "difference: " << (energ - energy_exact) * 1000 << "MeV" << endl;
    }


    if(ang_para_pre > 0){
        low_bound_angl_para = ang_para_pre - PlotParametersInput.delta_angl;
        high_bound_angl_para = ang_para_pre + PlotParametersInput.delta_angl;
        low_bound_angl_anti = ang_anti_pre - PlotParametersInput.delta_angl;
        high_bound_angl_anti = ang_anti_pre + PlotParametersInput.delta_angl;

        cout << "With current setting, range to observe parallel peak: " << low_bound_angl_para << " deg; " << high_bound_angl_para << " deg" << endl;
        cout << "With current setting, range to observe antiparallel peak: " << low_bound_angl_anti << " deg; " << high_bound_angl_anti << " deg" << endl;

        cout << "Estimated antiparallel angle: " << esti_anti << " teta_crys1 = " << teta_crys1 << endl;
        cout << "Estimated parallel angle: " << esti_para << endl;

        Maxi_angl_anti = high_bound_angl_anti - teta_crys1;
        Mini_angl_anti = low_bound_angl_anti - teta_crys1;

        Maxi_angl_para = high_bound_angl_para - teta_crys1;
        Mini_angl_para = low_bound_angl_para - teta_crys1;


        Maxi_angl = Maxi_angl_para + PlotParametersInput.shift_disp_window;
        Mini_angl = Mini_angl_para + PlotParametersInput.shift_disp_window;

        cout << " Maxi_angl_anti: " << Maxi_angl_anti << endl;
        cout << " Mini_angl_anti: " << Mini_angl_anti << endl;
        cout << " Maxi_angl_para: " << Maxi_angl_para << endl;
        cout << " Mini_angl_para: " << Mini_angl_para << endl;
        cout << " *** Mini angl = " << Mini_angl << " Maxi_angl = " << Maxi_angl << endl;

        low_bound_angl_para = teta_crys1 + Mini_angl;
        high_bound_angl_para = teta_crys1 + Maxi_angl;
        low_bound_angl_anti = -teta_crys1 - Mini_angl;
        high_bound_angl_anti = -teta_crys1 - Maxi_angl;

        cout << " Checking common range to observe parallel peak -> " << low_bound_angl_para << " deg; " << high_bound_angl_para << " deg; peak: " << ang_para_pre << " deg" << endl;
        cout << " Checking common range to observe antiparallel peak -> " << low_bound_angl_anti << " deg; " << high_bound_angl_anti << " deg; peak: " << ang_anti_pre << " deg" << endl;
    }else
        throw runtime_error(" Error in test_input: ang_para_pre <= 0");


    gener_out << endl;
    gener_out << "------------------------------" << endl;
    gener_out << endl;
    gener_out << " Estimated analysis based on analytical expressions" << endl;
    gener_out << endl;


    var_temp = Maxi_angl - Mini_angl;

    if(var_temp < 0)
        throw runtime_error("*** Error in test_input: min_angle grater than max_angle");
    else if(var_temp > 4)
        throw runtime_error("*** Error in test_input: maximum angle variation > 4 deg, too large");


    if(UserSettingsInput.see_para){
        if(UserSettingsInput.see_anti){
            cout << "input range set up for observing both parallel and antiparallel" << endl;

            cout << Mini_angl << "\t" << Maxi_angl << endl;

            if(abs(Mini_angl) > 4 || abs(Maxi_angl) > 4)
                throw runtime_error("*** Error in test_input: minimum or maximum range > 4 deg, too large");

            if(FullEnergySpectrumInput.make_more_lines < 2){
                if(Maxi_angl < esti_anti || Mini_angl > esti_anti){
                    cout << "In this range the antiparallel peak will not be observed" << endl;
                    cout << "Mini_angl = " << Mini_angl - teta_crys1 << " Maxi_angl = " << Maxi_angl - teta_crys1 << endl;
                    cout << "Mini_angl(ref) = " << Mini_angl << " Maxi_angl(ref) = " << Maxi_angl << " estimated antiparallel angle " << esti_anti << endl;
                    cout << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << endl;
                    cout << " setting maximum number of x rays in simulation to 2011 to check image" << endl;

                    NumberRaysInput.nbeams = 2011;
                }
            }
        }else{
            Mini_angl -= teta_crys1;
            Maxi_angl -= teta_crys1;
        }


        if(Maxi_angl < -esti_para || Mini_angl > -esti_para){
            cout << "In this range the parallel peak will not be observed" << endl;
            cout << "Mini_angl = " << Mini_angl - teta_crys1 << " Maxi_angl = " << Maxi_angl - teta_crys1 << endl;
            cout << "Mini_angl(ref) = " << Mini_angl << " Maxi_angl(ref) = " << Maxi_angl << " estimated parallel angle " << esti_para << endl;
            cout << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << endl;
            cout << " setting maximum number of x rays in simulation to 2011 to check image" << endl;

            NumberRaysInput.nbeams = 2011;
        }

    }else{
        if(UserSettingsInput.see_anti){
            Mini_angl += teta_crys1;
            Maxi_angl += teta_crys1;

            if(Maxi_angl < esti_anti || Mini_angl > esti_anti){
                cout << "In this range the antiparallel peak will not be observed" << endl;
                cout << "Mini_angl = " << Mini_angl - teta_crys1 << " Maxi_angl = " << Maxi_angl - teta_crys1 << endl;
                cout << "Mini_angl(ref) = " << Mini_angl << " Maxi_angl(ref) = " << Maxi_angl << " estimated antiparallel angle " << esti_anti << endl;
                cout << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << endl;
                cout << " setting maximum number of x rays in simulation to 2011 to check image" << endl;

                NumberRaysInput.nbeams = 2011;
            }
        }else{
            cout << "No peak selected" << endl;
            cout << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << endl;
            cout << " setting maximum number of x rays in simulation to 2011 to check image" << endl;

            NumberRaysInput.nbeams = 2011;
        }
    }


    if(GeoParapathlengthsInput.type_source == "P"){
        cout << " Evaluation set for a point source" << endl;
        cout << endl;
    }else if(GeoParapathlengthsInput.type_source == "UC"){
        cout << " Evaluation set for an uniform circular source" << endl;
        cout << endl;
    }else if(GeoParapathlengthsInput.type_source == "UR"){
        cout << " Evaluation set for an uniform rectangular source" << endl;
        cout << endl;
    }else if(GeoParapathlengthsInput.type_source == "G"){
        cout << " Evaluation set for a gaussian source" << endl;
        cout << endl;
    }else{
        cout << "Bad input in the type_source" << endl;
        cout << endl;
    }

    if(!UserSettingsInput.Make_Vertical){
        cout << "Evaluation with rays inside central plane" << endl;
        cout << endl;
    }

    gener_out << "----------------------------------------" << endl;

}

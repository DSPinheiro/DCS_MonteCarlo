#include "CheckInputSpectrum.hh"

using namespace std;



bool CheckInputSpectrum::CheckSpectrum(string unit){
    
    double tetaref, tetabrag_ref, sin_e;
    double tilt_C1_temp, cos_tilt_C1;
    double n1x;
    double delta_para, start1_para, tw_d1_para, d_lat1_para;

    double min_angle_resp = -1E-3;
    double max_angle_resp = 1E-3;

    d_lat1_para = Latice_temp(d_lat, TemperatureParametersInput.T_crystal_1_para);

    tw_d1_para = 2 * d_lat1_para;

    if (UserSettingsInput.Simple_simu) {
        if (GeoParametersInput.Exp_crys1 < 0) {
            throw runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be greater than 0. Exp_crys1 = 90 - tetabragg, tetabragg is the physical glancing angle of the first crystal to the x axis.");
        }

        teta_crys1 = GeoParametersInput.Exp_crys1;
    }
    else {
        if (GeoParametersInput.Exp_crys1 > 0) {
            throw runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be less than 0. Exp_crys1 = - 90 - teta, teta is the physical angle of the table.");
        }

        teta_crys1 = -GeoParametersInput.teta_table - GeoParametersInput.Exp_crys1 + GeoParametersInput.OffsetRotCry1;
    }

    tetaref = (90 - teta_crys1) * M_PI / 180.0;
    sin_e = sin(tetaref);

    if(CurveVerticalTiltInput.make_CurveTilt){
        tilt_C1_temp = ObtainVertical::ObtainVert(1, 0);
    }else{
        tilt_C1_temp = GeoParametersInput.tilt_C1 * M_PI / 180.0;
    }

    cos_tilt_C1 = cos(tilt_C1_temp);
    n1x = - cos_tilt_C1 * sin_e;

    tetabrag_ref = asin(-n1x);

    if(unit == "eV"){
        delta_para = (Convert_Ag_minusone_eV / (sin(tetabrag_ref + min_angle_resp) * tw_d1_para) - Convert_Ag_minusone_eV / (sin(tetabrag_ref + max_angle_resp) * tw_d1_para)) * 1.315;
        start1_para = Convert_Ag_minusone_eV / (sin(tetabrag_ref + 0.6 * max_angle_resp) * tw_d1_para);

        cout << "Energy start: " << start1_para << endl;
        cout << "Energy delta: " << delta_para << endl;
    }else if(unit == "A"){
        delta_para = ((sin(tetabrag_ref + min_angle_resp) * tw_d1_para) - (sin(tetabrag_ref + max_angle_resp) * tw_d1_para));
        start1_para = (sin(tetabrag_ref + max_angle_resp) * tw_d1_para);

        //cout << "Wavelength start: " << start1_para << endl;
        //cout << "Wavelength delta: " << delta_para << endl;
    }else{
        throw runtime_error("Error in CheckInputSpectrum: bad energy unit input");
    }

    cout << Energy_spec[0].lamda << "\t" << Energy_spec[Energy_spec.size() - 1].lamda << endl;

    if(Energy_spec[0].lamda <= start1_para){
        if(Energy_spec[Energy_spec.size() - 1].lamda >= start1_para + delta_para){
            return true;
        }
    }

    return false;
}

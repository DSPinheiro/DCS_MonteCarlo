/*
 * Mask_test_c2.cpp
 *
 *  Created on: Nov 4, 2020
 *      Author: danip
 */

#include "mask_test_c2.hh"
#include <Util.h>

using namespace std;


void Mask_test_c2::test_c2(){

    double inte_mask_down_para, inte_mask_down_anti, ratio_para, ratio_anti, inte_mask_up_para, inte_mask_up_anti;

    UserSettingsInput.mask_C2 = 1;

    Source_complex c;

    c.run_Source(nullptr);

    if(UserSettingsInput.fitting){
        if(UserSettingsInput.see_para)
            Util::fit(true);
        if(UserSettingsInput.see_anti)
            Util::fit(false);
        if(UserSettingsInput.see_para && UserSettingsInput.see_anti){}
            Util::analyse();
    }

    inte_mask_down_para = a_para_voig[1];
    inte_mask_down_anti = a_anti_voig[1];

    UserSettingsInput.mask_C2 = 2;

    c.run_Source(nullptr);

    if(UserSettingsInput.fitting){
        if(UserSettingsInput.see_para)
            Util::fit(true);
        if(UserSettingsInput.see_anti)
            Util::fit(false);
        if(UserSettingsInput.see_para && UserSettingsInput.see_anti){}
            Util::analyse();
    }

    inte_mask_up_para = a_para_voig[1];
    inte_mask_up_anti = a_anti_voig[1];


    ratio_para = inte_mask_down_para / inte_mask_up_para;
    ratio_anti = inte_mask_down_anti / inte_mask_up_anti;

    gener_out << endl;
    gener_out << " Results for the mask test" << endl;
    gener_out << endl;
    gener_out << " For parallel" << endl;
    gener_out << endl;
    gener_out << "Intensity of parallel peak mask down: " << inte_mask_down_para << endl;
    gener_out << "Intensity of parallel peak mask up: " << inte_mask_up_para << endl;
    gener_out << endl;
    gener_out << "Ratio between them: " << ratio_para << endl;
    gener_out << endl;
    gener_out << " For antiparallel" << endl;
    gener_out << endl;
    gener_out << "Intensity of antiparallel peak mask down: " << inte_mask_down_anti << endl;
    gener_out << "Intensity of antiparallel peak mask up: " << inte_mask_up_anti << endl;
    gener_out << endl;
    gener_out << "Ratio between them: " << ratio_anti << endl;

}

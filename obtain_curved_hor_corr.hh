/*
 * Obtain_Curved_Hor_Corr.hh
 *
 *  Created on: Nov 2, 2020
 *      Author: danip
 */

#ifndef OBTAIN_CURVED_HOR_CORR_HH_
#define OBTAIN_CURVED_HOR_CORR_HH_


#include "simuGlobals.hh"


class Obtain_Curved_Hor_Corr{
    public:
        static std::vector<double> horCorr(double y_pro_C1, double y_max_C1, double z_pro_C1, double z_max_C1, bool type_c);
};


#endif /* OBTAIN_CURVED_HOR_CORR_HH_ */

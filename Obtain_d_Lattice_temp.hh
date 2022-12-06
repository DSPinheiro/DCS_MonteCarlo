#pragma once

/*
 * Obtain_d_Latice_temp.hh
 *
 *  Created on: Oct 29, 2020
 *      Author: danip
 */

#ifndef OBTAIN_D_LATICE_TEMP_HH_
#define OBTAIN_D_LATICE_TEMP_HH_

static double Latice_temp(double d_lat, double T_crystal){
        double C1, C2, C3, C4, t0, Temp, a;
        const double a22 = 1.000054702395071;

        t0 = 273.15;

        C1 = 3.725E-6;
        C2 = 5.88E-3;
        C3 = C2 * 124.0;
        C4 = 2.774E-10;

        Temp = T_crystal + t0;

        a = (1.0 + (Temp - t0) * C1 + (pow(Temp, 2) - pow(t0, 2)) * C4 + (exp(-C2 * Temp) - exp(-C2 * t0)) * exp(C3) * C1 / C2);

        a /= a22;

        double d_lat_t = d_lat * a;

        d_lat_t = d_lat * (1 + (Temp - 295.65) * (2.56E-6));

        return d_lat_t;
}


#endif /* OBTAIN_D_LATICE_TEMP_HH_ */

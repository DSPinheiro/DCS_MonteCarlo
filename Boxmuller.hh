/*
 * Boxmuller.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef BOXMULLER_HH_
#define BOXMULLER_HH_

#include <math.h>

static double Box(double sta_dev, double mean){
        double fac, rsq, v1, v2;

        while(true){
                v1 = 2 * ((double)rand() / RAND_MAX) - 1;
                v2 = 2 * ((double)rand() / RAND_MAX) - 1;
                rsq = pow(v1, 2) + pow(v2, 2);

                if(not (rsq >= 1 or rsq == 0))
                        break;

        }

        fac = sqrt(-2 * log(rsq) / rsq);

        return mean + sta_dev * v2 * fac;

}



#endif /* BOXMULLER_HH_ */

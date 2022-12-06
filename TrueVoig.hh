#pragma once

/*
 * TrueVoig.hh
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */

#ifndef TRUEVOIG_HH_
#define TRUEVOIG_HH_

#include "cw.hh"

static void Voig(double &x, double* a, double &y, double* dyda, int na){

    int nam;

    double xw, yw, a0, aky, ak, al, DERKX, DERKY, DERI1, DY, DEKYY;

    const double c1 = 1.665109222315395, c2 = c1 / 2, c3 = 1.128379167095513;

    y = a[na - 1];
    std::fill(&dyda[0], &dyda[0] + 20, -1);
    nam = na - 1;

    DERI1 = 0;

    for(int j = 1; j < nam; j += 3){
        xw = (x - a[j + 2]) * c1 / a[0];
        yw = a[j + 1] * c2 / a[0];

        cw(0, yw, aky, a0);

        DEKYY = 2 * yw * aky - c3;

        cw(xw, yw, ak, al);

        DERKX = 2 * (yw * al - xw * ak);
        DERKY = 2 * (xw * al + yw * ak) - c3;
        DY = (DERKY - DEKYY * ak / aky) / aky;
        DERI1 += ((xw * DERKX / aky + yw * DY) * a[j]) / a[0];
        dyda[j] = ak / aky;
        dyda[j + 1] = c2 * a[j] * DY / a[0];
        dyda[j + 2] = - c1 * a[j] * DERKX / (a[0] * aky);
        y += a[j] * ak / aky;
    }

    dyda[0] = - DERI1;
    dyda[na - 1] = 1;

}



#endif /* TRUEVOIG_HH_ */

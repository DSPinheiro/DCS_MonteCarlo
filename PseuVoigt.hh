/*
 * PseudoVoig.hh
 *
 *  Created on: Nov 4, 2020
 *      Author: danip
 */

#ifndef PSEUDOVOIG_HH_
#define PSEUDOVOIG_HH_

#include "cw.hh"

static void Pseud(double &x, double* a, double &y, double* dyda, int na){

        double Lorenterm, DerLorenterm, const1, const2, term1, term2, Gaussterm, DerGausserm;

        const1 = sqrt(4 * log(2)) / (sqrt(M_PI) * a[0]);
        const2 = 4 * log(2) / pow(a[0], 2);

        Lorenterm = 2 * a[0] / (M_PI * (4 * pow((x - a[3]), 2) + pow(a[0], 2)));
        Gaussterm = const1 * exp(-const2 * pow((x - a[3]), 2));

        DerLorenterm = Lorenterm * 8 * (x - a[3]) / (4 * pow((x - a[3]), 2) + pow(a[0], 2));
        DerGausserm = 2 * const2 * (x - a[3]) * Gaussterm;

        term1 = (4 * pow((x - a[3]), 2) - a[0] * a[0]) / (pow((4 * pow((x - a[3]), 2) + pow(a[0], 2)), 2));
        term2 = (1 - 2 * const2 * pow((x - a[3]), 2)) / a[0];

        dyda[1] = a[2] * Lorenterm + (1 - a[2]) * Gaussterm;

        dyda[4] = 1;
        dyda[3] = a[1] * (a[2] * DerLorenterm + (1 - a[2]) * DerGausserm);

        dyda[0] = a[1] * (a[2] * 2 * term1 / M_PI + (1 - a[2]) * term2 * Gaussterm);
        dyda[2] = a[1] * (Lorenterm - Gaussterm);

        y = a[4] + a[1] * (a[2] * Lorenterm + (1 - a[2]) * Gaussterm);

}



#endif /* PSEUDOVOIG_HH_ */

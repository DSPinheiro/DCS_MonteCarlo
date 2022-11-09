/*
 * splint.hh
 *
 *  Created on: Nov 1, 2020
 *      Author: danip
 */

#ifndef SPLINT_HH_
#define SPLINT_HH_

#include <stdexcept>
#include <vector>
#include <math.h>

static double splint_te(std::vector<double> xa, std::vector<double> ya, std::vector<double> y2a, double x){
        int k, khi, klo;
        double a, b, h;

        klo = 1;
        khi = xa.size();

        while(khi - klo > 1){
                k = (khi + klo) / 2;
                if(xa[k - 1] > x)
                        khi = k;
                else
                        klo = k;
        }

        h = xa[khi - 1] - xa[klo - 1];
        if(h == 0)
                throw std::runtime_error("bad xa input in splint");

        a = (xa[khi - 1] - x) / h;
        b = (x - xa[klo - 1]) / h;

        return a * ya[klo - 1] + b * ya[khi - 1] + ((pow(a, 3) - a) * y2a[klo - 1] + (pow(b, 3) - b) * y2a[khi - 1]) * pow(h, 2) / 6.0;
}



#endif /* SPLINT_HH_ */

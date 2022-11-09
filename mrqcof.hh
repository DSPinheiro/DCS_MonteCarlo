/*
 * mrqcof.hh
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */

#ifndef MRQCOF_HH_
#define MRQCOF_HH_

#include <vector>
#include <iostream>

static void mrq_cof(std::vector<double> x, std::vector<double> y, std::vector<double> sig, int ndata, double* a, int* ia, int ma, std::vector<std::vector<double>> alpha, double* beta, int nalp, double &chisq, void (funcs(double&, double*, double&, double*, int))){

    const int MMAX = 20;

    double ymod, dyda[MMAX], sig2i, dy, wt;

    int mfit, j, k;

    mfit = 0;

    for(j = 0; j < ma; j++){
        if(ia[j] != 0)
            mfit++;
    }

    for(j = 0; j < mfit; j++){
        for(k = 0; k <= j; k++){
            alpha[j][k] = 0;
        }
        beta[j] = 0;
    }

    chisq = 0;

    for(int i = 0; i < ndata; i ++){
        funcs(x.at(i), a, ymod, dyda, ma);

        sig2i = 1 / (sig.at(i)*sig.at(i));
        dy = y.at(i) - ymod;

        j = 0;

        for(int l = 0; l < ma; l++){
            if(ia[l] != 0){
                wt = dyda[l] * sig2i;
                k = 0;

                for(int m = 0; m <= l; m++){
                    if(ia[m] != 0){
                        alpha.at(j).at(k) += wt*dyda[m];
                        k++;
                    }
                }
                beta[j] += dy*wt;

                j++;
            }
        }

        chisq += dy*dy*sig2i;

    }


    for(j = 1; j < mfit; j++){
        for(k = 0; k < j; k++){
            alpha.at(k).at(j) = alpha.at(j).at(k);
        }
    }


}


#endif /* MRQCOF_HH_ */

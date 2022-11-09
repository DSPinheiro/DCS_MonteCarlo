/*
 * mrqmin.hh
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */

#ifndef MRQMIN_HH_
#define MRQMIN_HH_

#include <vector>

#include "mrqcof.hh"
#include "gaussj.hh"
#include "covsrt.hh"

static void mrq_min(std::vector<double> x, std::vector<double> y, std::vector<double> sig, int ndata, double* a, int* ia, int ma, std::vector<std::vector<double>> covar, std::vector<std::vector<double>> alpha, int nca, double& chisq, void (funcs(double&, double*, double&, double*, int)), double& alamda){

    int mfit, j, k;

    const int MMAX = 20;

    double ochisq, atry[MMAX], beta[MMAX], da[MMAX];

    if(alamda < 0){
        mfit = 0;

        for(j = 0; j < ma; j++){
            if(ia[j] != 0)
                mfit++;
        }

        alamda = 0.001;

        mrq_cof(x, y, sig, ndata, a, ia, ma, alpha, beta, nca, chisq, funcs);
        ochisq = chisq;

        for(j = 0; j < ma; j++){
            atry[j] = a[j];
        }
    }

    j = 0;

    for(int l = 0; l < ma; l++){
        if(ia[l] != 0){
            k = 0;

            for(int m = 0; m < ma; m++){
                if(ia[m] != 0){
                    covar[j][k] = alpha[j][k];
                    k++;
                }
            }

            covar[j][j] = alpha[j][j] * (1 + alamda);
            da[j] = beta[j];
            j++;
        }
    }

    //da is declared as 1d array, but gauss_j expects 2d array as the parameter... the original code is also like this i dont even know how it compiled....
    //gauss_j(covar, mfit, nca, da, 1, 1);

    if(alamda == 0){
        //this resets the covar to 0 in a wierdly complicated way....
        mfit = ma;
        cov_srt(covar, nca, ma, ia, mfit);
        return;
    }

    j = 0;

    for(int l = 0; l < ma; l++){

        if(ia[l] != 0){
            atry[l] = a[l] + da[j];
            j++;
        }
    }

    mrq_cof(x, y, sig, ndata, atry, ia, ma, covar, da, nca, chisq, funcs);

    if(chisq < ochisq){

        alamda *= 0.1;
        ochisq = chisq;
        j = 0;

        for(int l = 0; l < ma; l++){
            if(ia[l] != 0){
                k = 0;

                for(int m = 0; m < ma; m++){
                    if(ia[m] != 0){
                        alpha[j][k] = covar[j][k];
                        k++;
                    }
                }

                beta[j] = da[j];
                a[l] = atry[l];
                j++;
            }
        }
    }else{
        alamda *= 10;
        chisq = ochisq;
    }

    return;

}


#endif /* MRQMIN_HH_ */

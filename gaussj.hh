#pragma once

/*
 * gaussj.hh
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */

#ifndef GAUSSJ_HH_
#define GAUSSJ_HH_

#include <iostream>

#include <vector>

static void gauss_j(std::vector<std::vector<double>> a, int n, int np, std::vector<std::vector<double>> b, int m, int mp){

    const int NMAX = 50;

    int ipiv[NMAX], indxr[NMAX], indxc[NMAX], irow, icol;

    double big, dum, pivinv;

    for(int j = 0; j < n; j++){
        ipiv[j] = 0;
    }

    for(int i = 0; i < n; i++){
        big = 0;
        for(int j = 0; j <n; j++){
            if(ipiv[j] != 1){
                for(int k = 0; k < n; k++){
                    if(ipiv[k] == 0){
                        if(a.at(j).at(k) >= big){
                            big = std::abs(a.at(j).at(k));
                            irow = j;
                            icol = k;
                        }
                    }else if(ipiv[k] > 1){
                        system("pause");
                    }
                }
            }
        }

        ipiv[icol]++;
        if(irow != icol){
            for(int l = 0; l < n; l++){
                dum = a.at(irow).at(l);
                a.at(irow).at(l) = a.at(icol).at(l);
                a.at(icol).at(l) = dum;
            }

            for(int l = 0; l < m; l++){
                dum = b.at(irow).at(l);
                b.at(irow).at(l) = b.at(icol).at(l);
                b.at(icol).at(l) = dum;
            }
        }

        indxr[i] = irow;
        indxc[i] = icol;
        if(a.at(icol).at(icol) == 0){
            system("pause");
        }

        pivinv = 1 / a.at(icol).at(icol);
        a.at(icol).at(icol) = 1;

        for(int l = 0; l < n; l++)
            a.at(icol).at(l) *=pivinv;

        for(int l = 0; l < m; l++)
            b.at(icol).at(l) *= pivinv;

        for(int ll = 0; ll < n; ll++){
            if(ll != icol){
                dum = a.at(ll).at(icol);
                a.at(ll).at(icol) = 0;
                for(int l = 0; l < n; l++)
                    a.at(ll).at(l) -= a.at(icol).at(l)*dum;
                for(int l = 0; l < n; l++)
                    b.at(ll).at(l) -= b.at(icol).at(l)*dum;
            }
        }
    }

    for(int l = n - 1; l >= 0; l--){
        if(indxr[l] != indxc[l]){
            for(int k = 0; k < n; k++){
                dum = a.at(k).at(indxr[l]);
                a.at(k).at(indxr[l]) = a.at(k).at(indxc[l]);
                a.at(k).at(indxc[l]) = dum;
            }
        }
    }

}


#endif /* GAUSSJ_HH_ */

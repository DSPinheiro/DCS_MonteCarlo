#pragma once

/*
 * cw.hh
 *
 *  Created on: Nov 3, 2020
 *      Author: danip
 */

#ifndef CW_HH_
#define CW_HH_

#include <complex>
#include <cmath>


static void cw(double xw, double yw, double &ak, double &al){

        const std::complex<double> C1(0.5641896, 0), C2(1.12837917, 0), CZ1(0.4613135,0), CZ2(0.1901635,0), CZ3(0.09999216,0), CZ4(1.7844927,0), CZ5(0.002883994,0), CZ6(5.5253437,0);
        const std::complex<double> CZZ1(0.55124242,0), CZZ2(2.752551,0), CZZ3(0.05176536,0), CZZ4(2.724745,0), ZI(0,1);

        std::complex<double> CWZ;

        const int ITMAX=400,ITPOL=40;

        const double EPS=1E-09;


        std::complex<double> Z (xw, yw);

        std::complex<double> Z2 = Z * Z;

        if(std::abs(xw) > 3 || std::abs(yw) > 3.9){
                CWZ = Z * ZI * (CZ1 / (Z2 - CZ2) + CZ3 / (Z2 - CZ4) + CZ5 / (Z2 - CZ6));
        }else if(std::abs(yw) > 0.5){
                std::complex<double> Cgold(0, 0);
                std::complex<double> Ca0(0, 0);
                std::complex<double> Ca1(1, 0);
                std::complex<double> Cb0 = Ca1;
                std::complex<double> Cb1 = Z;

                for(int n = 0; n < ITMAX; n++){

                        std::complex<double> CN((double)-n / 2, 0);
                        std::complex<double> Ca2(0, 0);
                        std::complex<double> Cb2(0, 0);
                        Ca0 = Ca1;
                        Ca1 = Ca2;
                        Cb0 = Cb1;
                        Cb1 = Cb2;

                        if(Ca2 == std::complex<double>(0, 0))
                                continue;

                        CWZ = Ca2 / Cb2;

                        if(abs((CWZ - Cgold) / CWZ) < EPS)
                                break;

                        Cgold = CWZ;

                }

                CWZ = CWZ * C1 * ZI;

        }else{

                Z = -Z * ZI;
                std::complex<double> Z21 = Z * Z;
                CWZ = 1 / (double)(2 * ITPOL + 1);

                for(int i = ITPOL; i >= 2; i--)
                        CWZ = 1 / (double)(2 * i - 1) - Z21 * CWZ / (double)i;

                CWZ = (std::complex<double>(1, 0) - Z21 * CWZ) * Z;
                CWZ = exp(-Z2) * (std::complex<double>(1, 0) - C2 * CWZ);
        }

        ak = real(CWZ);
        al = imag(CWZ);

}


#endif /* CW_HH_ */

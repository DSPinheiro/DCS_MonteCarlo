#define _USE_MATH_DEFINES


#include "../include/Util.h"

#include <cstdlib>
#include <math.h>

#include <complex>
#include <cmath>

#include <vector>
#include <iostream>

#include <stdexcept>

#include "../include/simuGlobals.hh"
#include <stdio.h>
#include <algorithm>

#include <string>

#ifdef OPENMP
#include <omp.h>
#endif

using namespace Util;

/// <summary>
/// Function to generate a random radius value according to a circular gaussian distribution.
/// Looks ineficient, might be able to improve it.
/// </summary>
/// <param name="sta_dev">
/// Standard deviation of the distribution.
/// </param>
/// <param name="mean">
/// Mean value of the distribution
/// </param>
/// <returns>
/// Randomized radius value from the circle center.
/// </returns>
double Util::GaussianBox(double sta_dev, double mean, bool box_muller) {
    
    unsigned seed;

    static thread_local std::mt19937 generator;

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    #ifdef OPENMP
    // random seed for this event if we have openMP enabled
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() + 17 * omp_get_thread_num();
    #else
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    #endif

    generator.seed(seed);
    
    double v1, v2;

    if(box_muller)
    {
        double fac, rsq;

        std::uniform_int_distribution<uint32_t> uniform(0, RAND_MAX);

        while (true) {
            v1 = 2 * ((double)uniform(generator) / RAND_MAX) - 1;
            v2 = 2 * ((double)uniform(generator) / RAND_MAX) - 1;
            rsq = pow(v1, 2) + pow(v2, 2);

            if (!(rsq >= 1 || rsq == 0))
                break;

        }

        fac = sqrt(-2 * log(rsq) / rsq);

        return mean + sta_dev * v2 * fac;
    }
    else
    {
        std::normal_distribution<double> normal(mean, sta_dev);

        v1 = normal(generator);
        v2 = normal(generator);

        return sqrt(v1 * v1 + v2 * v2);
    }

}


/// <summary>
/// Function that makes a matrix operation on the covar matrix.
/// I'm not sure what operation is being done here. It's something for the fitting algorithm.
/// </summary>
/// <param name="covar"></param>
/// <param name="npc"></param>
/// <param name="ma"></param>
/// <param name="ia"></param>
/// <param name="mfit"></param>
void Util::cov_srt(std::vector<std::vector<double>> covar, int npc, int ma, int* ia, int mfit) {

    double swap;

    int k;

    for (int i = mfit - 1; i < ma; i++) {
        for (int j = 0; j <= i; j++) {
            covar.at(i).at(j) = 0;
            covar.at(j).at(i) = 0;
        }
    }

    k = mfit;

    for (int j = ma - 1; j >= 0; j--) {
        if (ia[j] != 0) {
            k--;
            for (int i = 0; i < ma; i++) {
                swap = covar.at(i).at(k);
                covar.at(i).at(k) = covar.at(i).at(j);
                covar.at(i).at(j) = swap;
            }

            for (int i = 0; i < ma; i++) {
                swap = covar.at(k).at(i);
                covar.at(k).at(i) = covar.at(j).at(i);
                covar.at(j).at(i) = swap;
            }
        }
    }


}


/// <summary>
/// Helper function to calculate the Voigt profile.
/// The implemented Voigt profile is a true Voigt so we need this complex function
/// </summary>
/// <param name="xw"></param>
/// <param name="yw"></param>
/// <param name="ak"></param>
/// <param name="al"></param>
void Util::cw(double xw, double yw, double& ak, double& al) {

    const std::complex<double> c1(0.5641896, 0), C2(1.12837917, 0), CZ1(0.4613135, 0), CZ2(0.1901635, 0), CZ3(0.09999216, 0), CZ4(1.7844927, 0), CZ5(0.002883994, 0), CZ6(5.5253437, 0);
    const std::complex<double> CZZ1(0.55124242, 0), CZZ2(2.752551, 0), CZZ3(0.05176536, 0), CZZ4(2.724745, 0), ZI(0, 1);

    std::complex<double> CWZ;

    const int ITMAX = 400, ITPOL = 40;

    const double EPS = 1E-09;


    std::complex<double> Z(xw, yw);

    std::complex<double> Z2 = Z * Z;

    if (std::abs(xw) > 3 || std::abs(yw) > 3.9) {
        CWZ = Z * ZI * (CZ1 / (Z2 - CZ2) + CZ3 / (Z2 - CZ4) + CZ5 / (Z2 - CZ6));
    }
    else if (std::abs(yw) > 0.5) {
        std::complex<double> Cgold(0, 0);
        std::complex<double> Ca0(0, 0);
        std::complex<double> Ca1(1, 0);
        std::complex<double> Cb0 = Ca1;
        std::complex<double> Cb1 = Z;

        for (int n = 0; n < ITMAX; n++) {

            std::complex<double> CN((double)-n / 2, 0);
            std::complex<double> Ca2(0, 0);
            std::complex<double> Cb2(0, 0);
            Ca0 = Ca1;
            Ca1 = Ca2;
            Cb0 = Cb1;
            Cb1 = Cb2;

            if (Ca2 == std::complex<double>(0, 0))
                continue;

            CWZ = Ca2 / Cb2;

            if (abs((CWZ - Cgold) / CWZ) < EPS)
                break;

            Cgold = CWZ;

        }

        CWZ = CWZ * c1 * ZI;

    }
    else {

        Z = -Z * ZI;
        std::complex<double> Z21 = Z * Z;
        CWZ = 1 / (double)(2 * ITPOL + 1);

        for (int i = ITPOL; i >= 2; i--)
            CWZ = 1 / (double)(2 * i - 1) - Z21 * CWZ / (double)i;

        CWZ = (std::complex<double>(1, 0) - Z21 * CWZ) * Z;
        CWZ = exp(-Z2) * (std::complex<double>(1, 0) - C2 * CWZ);
    }

    ak = real(CWZ);
    al = imag(CWZ);

}


/// <summary>
/// Function to find the position of an element in the input array.
/// </summary>
/// <param name="array">
/// Array to search.
/// </param>
/// <param name="value">
/// Value to search for.
/// </param>
/// <returns>
/// Position of value in the array.
/// For now it throws an error if the value is not found but it should also return -1.
/// </returns>
int Util::FindLoc(std::vector<double> array, double value) {
    for (unsigned int i = 0; i < array.size(); i++) {
        if (array[i] == value) {
            return i;
        }
    }

    throw std::runtime_error("Value not found in array.");

}


/*

    FIRST CRYSTAL ANGLE FUNCTIONS

*/


/// <summary>
/// Function to get the first approximation to the angle.
/// Used in the simple simulation mode.
/// </summary>
/// <param name="tetaref">
/// Angle of the radiation.
/// </param>
/// <param name="tetadir">
/// Angle of the crystal.
/// </param>
/// <param name="sin_fi">
/// Sin of the vertical angle
/// </param>
/// <param name="cos_fi">
/// Cos of the vertical angle
/// </param>
/// <param name="tilt_C1">
/// Angle of the crystal systematic tilt
/// </param>
/// <param name="squa_tilt1"></param>
/// <returns>
/// Exit angle of the radiation.
/// </returns>
double Util::getFirstApproxAngle(double tetaref, double tetadir, double sin_fi, double cos_fi, double tilt_C1, double squa_tilt1) {

    double temp_sin, sinte;

    temp_sin = sin(tetadir + tetaref) * cos_fi;
    sinte = temp_sin * (1 - squa_tilt1) + sin_fi * tilt_C1;

    return asin(sinte);
}


/// <summary>
/// Function to get the full approximation to the angle.
/// Used in the simple simulation mode.
/// </summary>
/// <param name="tetaref">
/// Angle of the radiation.
/// </param>
/// <param name="tetadir">
/// Angle of the crystal.
/// </param>
/// <param name="cos_e"></param>
/// <param name="tan_e"></param>
/// <param name="fidir"></param>
/// <param name="tilt_C1">
/// Angle of the crystal systematic tilt
/// </param>
/// <returns>
/// Exit angle of the radiation.
/// </returns>
double Util::getFullApproximationAngle(double tetaref, double tetadir, double cos_e, double tan_e, double fidir, double tilt_C1) {
    return tetaref + tetadir - (pow(fidir, 2) + pow(tilt_C1, 2)) * tan_e + fidir * tilt_C1 / cos_e;
}


/// <summary>
/// Function to get the full geometric angle.
/// Used in the full simulation mode.
/// </summary>
/// <param name="r1x">
/// x component of the director vector for the radiation.
/// </param>
/// <param name="r1y">
/// y component of the director vector for the radiation.
/// </param>
/// <param name="r1z">
/// z component of the director vector for the radiation.
/// </param>
/// <param name="n1x">
/// x component of the director vector for the crystal.
/// </param>
/// <param name="n1y">
/// y component of the director vector for the crystal.
/// </param>
/// <param name="n1z">
/// z component of the director vector for the crystal.
/// </param>
/// <returns>
/// Vector with 4 elements:
/// Total spatial exit angle,
/// x component of the director vector for the radiation.
/// y component of the director vector for the radiation.
/// z component of the director vector for the radiation.
/// </returns>
///TODO: optimize
std::vector<double> Util::getFullAngle(double r1x, double r1y, double r1z, double n1x, double n1y, double n1z) {
    double inter_pro, angle, r2x, r2y, r2z;

    inter_pro = r1x * n1x + r1y * n1y + r1z * n1z;

    angle = asin(-inter_pro);
    r2x = r1x - 2 * inter_pro * n1x;
    r2y = r1y - 2 * inter_pro * n1y;
    r2z = r1z - 2 * inter_pro * n1z;

    std::vector<double> res;
    res.push_back(angle);
    res.push_back(r2x);
    res.push_back(r2y);
    res.push_back(r2z);

    return res;
}


/// <summary>
/// Some sort of gauss numerical recipe.
/// It's used in the minimization algorithm but I'm not sure what it is supposed to do.
/// </summary>
/// <param name="a"></param>
/// <param name="n"></param>
/// <param name="np"></param>
/// <param name="b"></param>
/// <param name="m"></param>
/// <param name="mp"></param>
void Util::gauss_j(std::vector<std::vector<double>> a, int n, int np, std::vector<std::vector<double>> b, int m, int mp) {

    const int NMAX = 50;

    int ipiv[NMAX], indxr[NMAX], indxc[NMAX], irow, icol;

    double big, dum, pivinv;

    for (int j = 0; j < n; j++) {
        ipiv[j] = 0;
    }

    for (int i = 0; i < n; i++) {
        big = 0;
        for (int j = 0; j < n; j++) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < n; k++) {
                    if (ipiv[k] == 0) {
                        if (a.at(j).at(k) >= big) {
                            big = std::abs(a.at(j).at(k));
                            irow = j;
                            icol = k;
                        }
                    }
                    else if (ipiv[k] > 1) {
                        system("pause");
                    }
                }
            }
        }

        ipiv[icol]++;
        if (irow != icol) {
            for (int l = 0; l < n; l++) {
                dum = a.at(irow).at(l);
                a.at(irow).at(l) = a.at(icol).at(l);
                a.at(icol).at(l) = dum;
            }

            for (int l = 0; l < m; l++) {
                dum = b.at(irow).at(l);
                b.at(irow).at(l) = b.at(icol).at(l);
                b.at(icol).at(l) = dum;
            }
        }

        indxr[i] = irow;
        indxc[i] = icol;
        if (a.at(icol).at(icol) == 0) {
            system("pause");
        }

        pivinv = 1 / a.at(icol).at(icol);
        a.at(icol).at(icol) = 1;

        for (int l = 0; l < n; l++)
            a.at(icol).at(l) *= pivinv;

        for (int l = 0; l < m; l++)
            b.at(icol).at(l) *= pivinv;

        for (int ll = 0; ll < n; ll++) {
            if (ll != icol) {
                dum = a.at(ll).at(icol);
                a.at(ll).at(icol) = 0;
                for (int l = 0; l < n; l++)
                    a.at(ll).at(l) -= a.at(icol).at(l) * dum;
                for (int l = 0; l < n; l++)
                    b.at(ll).at(l) -= b.at(icol).at(l) * dum;
            }
        }
    }

    for (int l = n - 1; l >= 0; l--) {
        if (indxr[l] != indxc[l]) {
            for (int k = 0; k < n; k++) {
                dum = a.at(k).at(indxr[l]);
                a.at(k).at(indxr[l]) = a.at(k).at(indxc[l]);
                a.at(k).at(indxc[l]) = dum;
            }
        }
    }

}


/// <summary>
/// Part of the fitting minimization algorithm.
/// This calculates the reduced chi^2.
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <param name="sig"></param>
/// <param name="ndata"></param>
/// <param name="a"></param>
/// <param name="ia"></param>
/// <param name="ma"></param>
/// <param name="alpha"></param>
/// <param name="beta"></param>
/// <param name="nalp"></param>
/// <param name="chisq"></param>
/// <param name="funcs"></param>
void Util::mrq_cof(std::vector<double> x, std::vector<double> y, std::vector<double> sig, int ndata, double* a, int* ia, int ma, std::vector<std::vector<double>> alpha, double* beta, int nalp, double& chisq, void(funcs(double&, double*, double&, double*, int))) {

    const int MMAX = 20;

    double ymod, dyda[MMAX], sig2i, dy, wt;

    int mfit, j, k;

    mfit = 0;

    for (j = 0; j < ma; j++) {
        if (ia[j] != 0)
            mfit++;
    }

    for (j = 0; j < mfit; j++) {
        for (k = 0; k <= j; k++) {
            alpha[j][k] = 0;
        }
        beta[j] = 0;
    }

    chisq = 0;

    for (int i = 0; i < ndata; i++) {
        funcs(x.at(i), a, ymod, dyda, ma);

        sig2i = 1 / (sig.at(i) * sig.at(i));
        dy = y.at(i) - ymod;

        j = 0;

        for (int l = 0; l < ma; l++) {
            if (ia[l] != 0) {
                wt = dyda[l] * sig2i;
                k = 0;

                for (int m = 0; m <= l; m++) {
                    if (ia[m] != 0) {
                        alpha.at(j).at(k) += wt * dyda[m];
                        k++;
                    }
                }
                beta[j] += dy * wt;

                j++;
            }
        }

        chisq += dy * dy * sig2i;

    }


    for (j = 1; j < mfit; j++) {
        for (k = 0; k < j; k++) {
            alpha.at(k).at(j) = alpha.at(j).at(k);
        }
    }


}


/// <summary>
/// Minimization algorithm for fitting.
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <param name="sig"></param>
/// <param name="ndata"></param>
/// <param name="a"></param>
/// <param name="ia"></param>
/// <param name="ma"></param>
/// <param name="covar"></param>
/// <param name="alpha"></param>
/// <param name="nca"></param>
/// <param name="chisq"></param>
/// <param name="funcs"></param>
/// <param name="alamda"></param>
void Util::mrq_min(std::vector<double> x, std::vector<double> y, std::vector<double> sig, int ndata, double* a, int* ia, int ma, std::vector<std::vector<double>> covar, std::vector<std::vector<double>> alpha, int nca, double& chisq, void(funcs(double&, double*, double&, double*, int)), double& alamda) {

    int mfit, j, k;

    const int MMAX = 20;

    double atry[MMAX], beta[MMAX], da[MMAX];

    if (alamda < 0) {
        mfit = 0;

        for (j = 0; j < ma; j++) {
            if (ia[j] != 0)
                mfit++;
        }

        alamda = 0.001;

        mrq_cof(x, y, sig, ndata, a, ia, ma, alpha, beta, nca, chisq, funcs);
        ochisq = chisq;

        for (j = 0; j < ma; j++) {
            atry[j] = a[j];
        }
    }

    j = 0;

    for (int l = 0; l < ma; l++) {
        if (ia[l] != 0) {
            k = 0;

            for (int m = 0; m < ma; m++) {
                if (ia[m] != 0) {
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

    if (alamda == 0) {
        //this resets the covar to 0 in a wierdly complicated way....
        mfit = ma;
        cov_srt(covar, nca, ma, ia, mfit);
        return;
    }

    j = 0;

    for (int l = 0; l < ma; l++) {

        if (ia[l] != 0) {
            atry[l] = a[l] + da[j];
            j++;
        }
    }

    mrq_cof(x, y, sig, ndata, atry, ia, ma, covar, da, nca, chisq, funcs);

    if (chisq < ochisq) {

        alamda *= 0.1;
        ochisq = chisq;
        j = 0;

        for (int l = 0; l < ma; l++) {
            if (ia[l] != 0) {
                k = 0;

                for (int m = 0; m < ma; m++) {
                    if (ia[m] != 0) {
                        alpha[j][k] = covar[j][k];
                        k++;
                    }
                }

                beta[j] = da[j];
                a[l] = atry[l];
                j++;
            }
        }
    }
    else {
        alamda *= 10;
        chisq = ochisq;
    }

    return;

}


/// <summary>
/// Function to calculate the crystals' latice spacing given a temperature.
/// </summary>
/// <param name="d_lat">
/// Reference latice spacing.
/// </param>
/// <param name="T_crystal">
/// Crystal temperature.
/// </param>
/// <returns>
/// Latice spacing for the temperature.
/// </returns>
double Util::Latice_temp(double d_lat, double T_crystal) {
    double c1, C2, C3, C4, t0, Temp, a;
    const double a22 = 1.000054702395071;

    t0 = 273.15;

    c1 = 3.725E-6;
    C2 = 5.88E-3;
    C3 = C2 * 124.0;
    C4 = 2.774E-10;

    Temp = T_crystal + t0;

    a = (1.0 + (Temp - t0) * c1 + (pow(Temp, 2) - pow(t0, 2)) * C4 + (exp(-C2 * Temp) - exp(-C2 * t0)) * exp(C3) * c1 / C2);

    a /= a22;

    double d_lat_t = d_lat * a;

    d_lat_t = d_lat * (1 + (Temp - 295.65) * (2.56E-6));

    return d_lat_t;
}

//TODO: optimize
std::vector<double> Util::getYZ(double r_temp, double sin_tetap_temp, double cos_tetap_temp, double tan_tetadir_temp, double tan_fidir_temp, double L_temp) {
    std::vector<double> res;

    res.push_back(r_temp * cos_tetap_temp + tan_tetadir_temp * L_temp);
    res.push_back(r_temp * sin_tetap_temp + tan_fidir_temp * L_temp);

    return res;
}


void Util::Pseud(double& x, double* a, double& y, double* dyda, int na) {

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


bool Util::Reached(double z, double y, double tetadir_temp, double fidir_temp, double L_temp, double z_max, double z_min, double y_max, double y_min) {
    double z_temp, y_temp;

    z_temp = z + tan(fidir_temp) * L_temp;
    y_temp = y + tan(tetadir_temp) * L_temp;

    if (z_temp < z_max && z_temp > z_min && y_temp < y_max && y_temp > y_min)
        return true;
    else
        return false;
}

/*

    SECOND CRYSTAL ANGLE FUNCTIONS

*/

/// <summary>
/// Function to get the first approximation to the angle.
/// Used in the simple simulation mode.
/// </summary>
/// <param name="tetaref">
/// Angle of the radiation.
/// </param>
/// <param name="tetadir">
/// Angle of the crystal.
/// </param>
/// <param name="sin_fi">
/// Sin of the vertical angle
/// </param>
/// <param name="cos_fi">
/// Cos of the vertical angle
/// </param>
/// <param name="tilt_C1">
/// Angle of the crystal systematic tilt
/// </param>
/// <param name="squa_tilt1"></param>
/// <returns>
/// Exit angle of the radiation.
/// </returns>
double Util::getFirstApproxAngle2(double tetaref, double tetadir, double delrot, double sin_fi, double cos_fi, double squa_tilt2, double cosdel, double cosdel_othe, double cosdel_teta, double cosdel_teta_othe, double sin_teref_tedi, bool Parallel) {

    double temp_sin, sinte;

    if (Parallel) {
        temp_sin = sin(tetadir + tetaref - delrot) * cos_fi;
        sinte = temp_sin * (1 - squa_tilt2) + cosdel * sin_fi - cosdel_othe * cos_fi * sin_teref_tedi;
        return asin(sinte);
    }
    else {
        temp_sin = sin(-tetadir + tetaref + delrot) * cos_fi;
        sinte = temp_sin * (1 - squa_tilt2) + cosdel * sin_fi - cosdel_teta_othe * cos_fi * sin_teref_tedi;
        return asin(sinte);
    }

}


/// <summary>
/// Function to get the full approximation to the angle.
/// Used in the simple simulation mode.
/// </summary>
/// <param name="tetaref">
/// Angle of the radiation.
/// </param>
/// <param name="tetadir">
/// Angle of the crystal.
/// </param>
/// <param name="cos_e"></param>
/// <param name="tan_e"></param>
/// <param name="fidir"></param>
/// <param name="tilt_C1">
/// Angle of the crystal systematic tilt
/// </param>
/// <returns>
/// Exit angle of the radiation.
/// </returns>
double Util::getFullApproximationAngle2(double tetaref, double tetadir, double delrot, double cos_e, double tan_e, double cos2_e, double fidir, double tilt_C1, double tilt_C2, bool Parallel) {

    if (Parallel)
        return tetaref + tetadir - delrot - tan_e * (pow(fidir, 2) + pow(tilt_C2, 2) + 4 * tilt_C1 * (tilt_C1 + tilt_C2)) + fidir * (tilt_C2 + 2 * tilt_C1) / cos_e;
    else
        return tetaref - tetadir + delrot - tan_e * (pow(fidir, 2) + pow(tilt_C2, 2) + 4 * tilt_C1 * tilt_C2 - 4 * cos2_e * pow(tilt_C1, 2)) + fidir * (tilt_C2 - 2 * cos2_e * tilt_C1) / cos_e;

}


/// <summary>
/// Function to get the full geometric angle.
/// Used in the full simulation mode.
/// </summary>
/// <param name="r1x">
/// x component of the director vector for the radiation.
/// </param>
/// <param name="r1y">
/// y component of the director vector for the radiation.
/// </param>
/// <param name="r1z">
/// z component of the director vector for the radiation.
/// </param>
/// <param name="n1x">
/// x component of the director vector for the crystal.
/// </param>
/// <param name="n1y">
/// y component of the director vector for the crystal.
/// </param>
/// <param name="n1z">
/// z component of the director vector for the crystal.
/// </param>
/// <returns>
/// Vector with 4 elements:
/// Total spatial exit angle,
/// x component of the director vector for the radiation.
/// y component of the director vector for the radiation.
/// z component of the director vector for the radiation.
/// </returns>
///TODO: optimize
std::vector<double> Util::getFullAngle2(double r2x, double r2y, double r2z, double n2x, double n2y, double n2z) {

    double inter_pro, angle, r3x, r3y, r3z;

    inter_pro = r2x * n2x + r2y * n2y + r2z * n2z;

    angle = asin(-inter_pro);
    r3x = r2x - 2 * inter_pro * n2x;
    r3y = r2y - 2 * inter_pro * n2y;
    r3z = r2z - 2 * inter_pro * n2z;

    std::vector<double> res;
    res.push_back(angle);
    res.push_back(r3x);
    res.push_back(r3y);
    res.push_back(r3z);

    return res;

}


/// <summary>
/// Function to calculate spline interpolation.
/// </summary>
/// <param name="x">
/// Vector of x values.
/// </param>
/// <param name="y">
/// Vector of y values.
/// </param>
/// <param name="yp1">
/// Value of the first derivative at the first point.
/// </param>
/// <param name="ypn">
/// Value of the first derivative at the last point.
/// </param>
/// <returns>
/// Vector with the values of the second derivative at each point.
/// </returns>
std::vector<double> Util::spline(std::vector<double> x, std::vector<double> y, double yp1, double ypn) {

    std::vector<double> y2, u;

    double p, qn, sig, un;

    size_t n = x.size();

    if (yp1 > 0.99E30) {
        y2.push_back(0);
        u.push_back(0);
    }
    else {
        y2.push_back(0.5);
        u.push_back((3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1));
    }

    for (int i = 1; i < n - 1; i++) {
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        p = sig * y2[i - 1] + 2;

        y2.push_back((sig - 1) / p);
        u.push_back((6 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])) / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p);
    }

    if (ypn > 0.99E30) {
        qn = 0;
        un = 0;
    }
    else {
        qn = 0.5;
        un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
    }

    y2.push_back((un - qn * u[n - 2]) / (qn * y2[n - 2] + 1));

    for (int k = n - 2; k >= 0; k--) {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }

    return y2;
}


/// <summary>
/// Function to calculate the value of the spline interpolation at an x value.
/// </summary>
/// <param name="xa">
/// Interpolated x values.
/// </param>
/// <param name="ya">
/// Interpolated y values.
/// </param>
/// <param name="y2a">
/// Values of the second derivative for the interpolation at each point.
/// </param>
/// <param name="x">
/// x value to interpolate.
/// </param>
/// <returns>
/// y value given by the spline interpolation.
/// </returns>
double Util::splint_te(std::vector<double> xa, std::vector<double> ya, std::vector<double> y2a, double x) {
    int k, klo;
    double a, b, h;

    klo = 1;
    size_t khi = xa.size();

    while (khi - klo > 1) {
        k = (khi + klo) / 2;
        if (xa[k - 1] > x)
            khi = k;
        else
            klo = k;
    }

    h = xa[khi - 1] - xa[klo - 1];
    if (h == 0)
        throw std::runtime_error("bad xa input in splint");

    a = (xa[khi - 1] - x) / h;
    b = (x - xa[klo - 1]) / h;

    return a * ya[klo - 1] + b * ya[khi - 1] + ((pow(a, 3) - a) * y2a[klo - 1] + (pow(b, 3) - b) * y2a[khi - 1]) * pow(h, 2) / 6.0;
}


/// <summary>
/// Function to calculate the Voigt profile.
/// </summary>
/// <param name="x">
/// Mean value.
/// </param>
/// <param name="a"></param>
/// <param name="y"></param>
/// <param name="dyda"></param>
/// <param name="na">
/// Number of points to calculate the profile.
/// </param>
void Util::Voig(double& x, double* a, double& y, double* dyda, int na) {

    int nam;

    double xw, yw, a0, aky, ak, al, DERKX, DERKY, DERI1, DY, DEKYY;

    const double c1 = 1.665109222315395, c2 = c1 / 2, c3 = 1.128379167095513;

    y = a[na - 1];
    std::fill(&dyda[0], &dyda[0] + 20, -1);
    nam = na - 1;

    DERI1 = 0;

    for (int j = 1; j < nam; j += 3) {
        xw = (x - a[j + 2]) * c1 / a[0];
        yw = a[j + 1] * c2 / a[0];

        Util::cw(0, yw, aky, a0);

        DEKYY = 2 * yw * aky - c3;

        Util::cw(xw, yw, ak, al);

        DERKX = 2 * (yw * al - xw * ak);
        DERKY = 2 * (xw * al + yw * ak) - c3;
        DY = (DERKY - DEKYY * ak / aky) / aky;
        DERI1 += ((xw * DERKX / aky + yw * DY) * a[j]) / a[0];
        dyda[j] = ak / aky;
        dyda[j + 1] = c2 * a[j] * DY / a[0];
        dyda[j + 2] = -c1 * a[j] * DERKX / (a[0] * aky);
        y += a[j] * ak / aky;
    }

    dyda[0] = -DERI1;
    dyda[na - 1] = 1;

}


/// <summary>
/// Function to analyse the simulated spectra.
/// The analysis is outputed to the console and the general output.
/// </summary>
void Util::analyse() {

    std::stringstream logString;

    double p_para, p_anti, dif, wave, energ_absorb, energy_exact, corre_30cel, energ, d_lat_2;
    double energy_obtain;

    p_para = a_para_voig[3];
    p_anti = a_anti_voig[3];

    if (GeometryInput.mode_bragg_geo)
        dif = (p_anti - p_para) / 2 + 90 - vert_div_corr - refra_corr + shape_corr;
    else
        dif = (p_anti - p_para) / 2 - vert_div_corr;



    dif *= M_PI / 180;

    energ_absorb = 0;

    corre_30cel = 0;

    if (FullEnergySpectrumInput.make_more_lines == 1)
        energy_exact = Convert_Ag_minusone_eV / picks[2].lamda;
    else if (FullEnergySpectrumInput.make_more_lines == 0)
        energy_exact = Convert_Ag_minusone_eV / picks[0].lamda;
    else
        energy_exact = middle_energy;


    d_lat_2 = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_2_anti);

    wave = 2 * d_lat_2 * sin(dif);

    energ = Convert_Ag_minusone_eV / wave - energ_absorb;

    energy_obtain = energ;


    gener_out << std::endl;
    gener_out << "-----------------------------------------" << std::endl;
    gener_out << std::endl;
    gener_out << " Vertical correction: " << vert_div_corr << std::endl;
    gener_out << " Refraction correction: " << refra_corr << std::endl;
    gener_out << std::endl;
    gener_out << "Parameter analysies " << std::endl;
    gener_out << std::endl;
    gener_out << "ration widths anti/para: " << FWMH_V_anti / FWMH_V_para << std::endl;
    gener_out << std::endl;
    gener_out << "Angular difference: " << dif << std::endl;
    gener_out << "wavelength: " << wave << " A" << std::endl;
    gener_out << "Energy obtain: " << energ << " eV" << std::endl;

    gener_out << "Energy input: " << energy_exact << " eV" << std::endl;
    gener_out << "difference obtain and input: " << (energ - energy_exact) * 1000 << " MeV" << std::endl;


    if (FullEnergySpectrumInput.make_more_lines <= 1) {
        logString << std::endl;
        logString << "-----------------------------------------" << std::endl;
        logString << std::endl;
        logString << "Parameter analysies " << std::endl;
        logString << std::endl;
        logString << "ration widths anti/para: " << FWMH_V_anti / FWMH_V_para << std::endl;
        logString << std::endl;
        logString << "Angular difference: " << dif << std::endl;
        logString << "wavelength: " << wave << " A" << std::endl;
        logString << "Energy obtain: " << energ << " eV" << std::endl;

        logString << "Energy input: " << energy_exact << " eV" << std::endl;
        logString << "difference obtain and input: " << (energ - energy_exact) * 1000 << " MeV" << std::endl;
    }
    else {
        logString << std::endl;
        logString << "-----------------------------------------" << std::endl;
        logString << std::endl;
        logString << "Parameter analysies (unreliable for an input energy spectrum)" << std::endl;
        logString << std::endl;
        logString << "ration widths anti/para: " << FWMH_V_anti / FWMH_V_para << std::endl;
        logString << std::endl;
        logString << "Angular difference: " << dif << std::endl;
        logString << "wavelength: " << wave << " A" << std::endl;
        logString << "Energy obtain: " << energ << " eV" << std::endl;

        logString << "Energy input: " << energy_exact << " eV" << std::endl;
        logString << "difference obtain and input: " << (energ - energy_exact) * 1000 << " MeV" << std::endl;
        
    }
    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #else
    std::cout << logString.str();
    #endif

    if (root_script) {
        //old script writing
    }

}


/// <summary>
/// Function to correct the vertical angle due to the apperture.
/// </summary>
void Util::geo_corre() {

    double Dis_total, teta_ref, tan_e, con_deg, con_rad;

    con_deg = 180 / M_PI;
    con_rad = M_PI / 180;

    Dis_total = GeoParapathlengthsInput.LT_aper + GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det;
    teta_ref = M_PI / 2 - teta_crys1 * con_rad;
    tan_e = tan(teta_ref);

    if (UserSettingsInput.Make_Vertical) {
        if (GeoParapathlengthsInput.type_source == "UC")
            vert_div_corr = tan_e * (pow(GeolengthelementsInput.S_aper, 2) + pow(GeolengthelementsInput.zdetc, 2) / (24 * pow(Dis_total, 2)));
        else if (UserSettingsInput.Make_Vertical)
            vert_div_corr = tan_e * (pow(GeolengthelementsInput.z_aper, 2) + pow(GeolengthelementsInput.zdetc, 2) / (24 * pow(Dis_total, 2)));

        vert_div_corr *= con_deg;

    }
    else
        vert_div_corr = 0;

}


/// <summary>
/// Function to calculate the effective vertical tilt crystal angle.
/// </summary>
/// <param name="crystal">
/// Index to identify the crystal to calculate.
/// </param>
/// <param name="angle">
/// Current second crystal angle.
/// </param>
/// <returns>
/// Effective vertical tilt.
/// </returns>
double Util::ObtainVert(int crystal, double angle) {

    double angle_temp, temp, rad, offsettilt, phase_temp, consttilt;

    rad = M_PI / 180.0;

    if (crystal == 1) {
        angle_temp = GeoParametersInput.Exp_crys1 * rad;
        phase_temp = CurveVerticalTiltInput.phas_tilt1 * rad;
        offsettilt = CurveVerticalTiltInput.offsettilt1 * rad;
        consttilt = CurveVerticalTiltInput.consttilt1 * rad;
    }
    else {
        angle_temp = angle * rad;
        phase_temp = CurveVerticalTiltInput.phas_tilt2 * rad;
        offsettilt = CurveVerticalTiltInput.offsettilt2 * rad;
        consttilt = CurveVerticalTiltInput.consttilt2 * rad;
    }

    temp = cos(angle_temp + phase_temp);

    double tilt;

    if (temp > 0) {
        tilt = -acos(cos(offsettilt) * pow(cos(angle_temp + phase_temp), 2) + pow(sin(angle_temp + phase_temp), 2));
    }
    else {
        tilt = acos(cos(offsettilt) * pow(cos(angle_temp + phase_temp), 2) + pow(sin(angle_temp + phase_temp), 2));
    }

    return tilt + consttilt;
}


/// <summary>
/// Function to check if the source energy spectrum is going to be visible in the detector.
/// It's not a full geometric calculation but is a good enough approximation where we
/// know if the spectrum is going to appear in the output or not.
/// </summary>
/// <param name="unit">
/// Energy unit for the source spectrum
/// </param>
/// <returns>
/// True or False, if the spectrum will be visible or not.
/// </returns>
bool Util::CheckSpectrum(std::string unit) {

    std::stringstream logString;

    double tetaref, tetabrag_ref, sin_e;
    double tilt_C1_temp, cos_tilt_C1;
    double n1x;
    double delta_para, start1_para, tw_d1_para, d_lat1_para;

    double min_angle_resp = -1E-3;
    double max_angle_resp = 1E-3;

    d_lat1_para = Util::Latice_temp(d_lat, TemperatureParametersInput.T_crystal_1_para);

    tw_d1_para = 2 * d_lat1_para;

    if (UserSettingsInput.Simple_simu) {
        if (GeoParametersInput.Exp_crys1 < 0) {
            throw std::runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be greater than 0. Exp_crys1 = 90 - tetabragg, tetabragg is the physical glancing angle of the first crystal to the x axis.");
        }

        teta_crys1 = GeoParametersInput.Exp_crys1;
    }
    else {
        if (GeoParametersInput.Exp_crys1 > 0) {
            throw std::runtime_error("Bad input ofr Exp_crys1. For a simple simulation it has to be less than 0. Exp_crys1 = - 90 - teta, teta is the physical angle of the table.");
        }

        teta_crys1 = -GeoParametersInput.teta_table - GeoParametersInput.Exp_crys1 + GeoParametersInput.OffsetRotCry1;
    }

    tetaref = (90 - teta_crys1) * M_PI / 180.0;
    sin_e = sin(tetaref);

    if (CurveVerticalTiltInput.make_CurveTilt) {
        tilt_C1_temp = Util::ObtainVert(1, 0);
    }
    else {
        tilt_C1_temp = GeoParametersInput.tilt_C1 * M_PI / 180.0;
    }

    cos_tilt_C1 = cos(tilt_C1_temp);
    n1x = -cos_tilt_C1 * sin_e;

    tetabrag_ref = asin(-n1x);

    if (unit == "eV") {
        delta_para = (Convert_Ag_minusone_eV / (sin(tetabrag_ref + min_angle_resp) * tw_d1_para) - Convert_Ag_minusone_eV / (sin(tetabrag_ref + max_angle_resp) * tw_d1_para)) * 1.315;
        start1_para = Convert_Ag_minusone_eV / (sin(tetabrag_ref + 0.6 * max_angle_resp) * tw_d1_para);

        logString << "Energy start: " << start1_para << std::endl;
        logString << "Energy delta: " << delta_para << std::endl;
    }
    else if (unit == "A") {
        delta_para = ((sin(tetabrag_ref + min_angle_resp) * tw_d1_para) - (sin(tetabrag_ref + max_angle_resp) * tw_d1_para));
        start1_para = (sin(tetabrag_ref + max_angle_resp) * tw_d1_para);

        logString << "Wavelength start: " << start1_para << std::endl;
        logString << "Wavelength delta: " << delta_para << std::endl;
    }
    else {
        throw std::runtime_error("Error in CheckInputSpectrum: bad energy unit input");
    }

    logString << Energy_spec[0].lamda << "\t" << Energy_spec[Energy_spec.size() - 1].lamda << std::endl;
    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #else
    std::cout << logString.str();
    #endif

    if (Energy_spec[0].lamda <= start1_para) {
        if (Energy_spec[Energy_spec.size() - 1].lamda >= start1_para + delta_para) {
            return true;
        }
    }

    return false;
}


/// <summary>
/// Function to check if the radiation is going to be reflected.
/// This is done using the crystal responce functions as probability and
/// calculating a random value.
/// </summary>
/// <param name="angle">
/// Radiation incident angle.
/// </param>
/// <param name="tetabra">
/// Crystal angle.
/// </param>
/// <param name="lamda">
/// Radiation energy
/// </param>
/// <param name="type_crystal">
/// Curved or flat crystal (true, false).
/// </param>
/// <param name="poli_p">
/// Use p-polarization or s-polarization (true, false).
/// </param>
/// <returns>
/// True or False, if the radiation was reflected or not
/// </returns>
bool Util::getReflection(double angle, double tetabra, double lamda, bool type_crystal, bool poli_p) {

    double p, dif, inte, inte1, inte2;

    dif = angle - tetabra;

    double energy = Convert_Ag_minusone_eV / lamda;

    int energy_resp_index;

    int index = 0;
    for (double energ : available_energies)
    {
        if (energ > energy)
        {
            energy_resp_index = index - 1;
            break;
        }
        index++;
    }

    unsigned seed;

    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<uint32_t> uniform(0, RAND_MAX);

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    #ifdef OPENMP
    // random seed for this event if we have openMP enabled
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() + 17 * omp_get_thread_num();
    #else
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    #endif

    generator.seed(seed);


    double energy_min_angle_resp, energy_max_angle_resp;
    energy_min_angle_resp = std::max(min_angle_resp[energy_resp_index], min_angle_resp[energy_resp_index + 1]);
    energy_max_angle_resp = std::min(max_angle_resp[energy_resp_index], max_angle_resp[energy_resp_index + 1]);

    if (dif < energy_min_angle_resp) {
        return false;
    }
    else {
        if (dif < energy_max_angle_resp) {
            if (type_crystal && PolarizationParametersInput.mka_poli) {
                if (poli_p) {
                    inte1 = Util::splint_te(Crystal_Responces[energy_resp_index].degrees,
                                            Crystal_Responces[energy_resp_index].reflecti_total_ps,
                                            Crystal_Responces[energy_resp_index].reflecti_two_deriv_ps, dif);

                    inte2 = Util::splint_te(Crystal_Responces[energy_resp_index + 1].degrees,
                                            Crystal_Responces[energy_resp_index + 1].reflecti_total_ps,
                                            Crystal_Responces[energy_resp_index + 1].reflecti_total_ps, dif);

                    inte = ((inte2 - inte1) / (available_energies[energy_resp_index + 1] - available_energies[energy_resp_index])) * (energy - available_energies[energy_resp_index]) + inte1;
                }
                else {
                    inte1 = Util::splint_te(Crystal_Responces[energy_resp_index].degrees,
                                            Crystal_Responces[energy_resp_index].reflecti_total_ss,
                                            Crystal_Responces[energy_resp_index].reflecti_two_deriv_ss, dif);

                    inte2 = Util::splint_te(Crystal_Responces[energy_resp_index + 1].degrees,
                                            Crystal_Responces[energy_resp_index + 1].reflecti_total_ss,
                                            Crystal_Responces[energy_resp_index + 1].reflecti_two_deriv_ss, dif);

                    inte = ((inte2 - inte1) / (available_energies[energy_resp_index + 1] - available_energies[energy_resp_index])) * (energy - available_energies[energy_resp_index]) + inte1;
                }
            }
            else {
                inte1 = Util::splint_te(Crystal_Responces[energy_resp_index].degrees,
                                        Crystal_Responces[energy_resp_index].reflecti_totals,
                                        Crystal_Responces[energy_resp_index].reflecti_two_derivs, dif);

                inte2 = Util::splint_te(Crystal_Responces[energy_resp_index + 1].degrees,
                                        Crystal_Responces[energy_resp_index + 1].reflecti_totals,
                                        Crystal_Responces[energy_resp_index + 1].reflecti_two_derivs, dif);

                inte = ((inte2 - inte1) / (available_energies[energy_resp_index + 1] - available_energies[energy_resp_index])) * (energy - available_energies[energy_resp_index]) + inte1;
            }

            p = ((double)uniform(generator) / RAND_MAX);

            if (p < inte)
                return true;
            else
                return false;

        }
        else {
            return false;
        }
    }

}


/// <summary>
/// Function to calculate the next bin's temperature shift due to random fluctuations.
/// A bin corresponds to a rotation angle.
/// </summary>
/// <param name="bin_tem">
/// Bin to calculate the new temperature shift.
/// </param>
/// <param name="bin_fas">
/// How many bins since the temperature was updated.
/// </param>
/// <param name="pha_tem">
/// Random temperature shift modifier.
/// </param>
/// <returns>
/// New crystal temperature shift.
/// </returns>
double Util::getNewTemp(int bin_tem, int& bin_fas, double& pha_tem) {

    std::stringstream logString;

    unsigned seed;

    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<uint32_t> uniform(0, RAND_MAX);

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    #ifdef OPENMP
    // random seed for this event if we have openMP enabled
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() + 17 * omp_get_thread_num();
    #else
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    #endif

    generator.seed(seed);

    if (bin_fas > TemperatureParametersInput.TT_tempera) {
        pha_tem = 2 * M_PI * ((double)uniform(generator) / RAND_MAX);
        bin_fas = 0;
    }
    else
        bin_fas++;


    if (TemperatureParametersInput.TT_tempera == 0) {
        logString << "Warning: Value for TT_tempera is 0" << std::endl;
        #ifdef QT_EXISTS
        if(logBox != NULL)
            logBox->appendPlainText(logString.str().c_str());
        #else
        std::cout << logString.str();
        #endif
    }

    return TemperatureParametersInput.AA_tempera * cos(2 * M_PI * bin_tem / TemperatureParametersInput.TT_tempera + pha_tem);

}


/// <summary>
/// Function to calculate the source energy for this event.
/// </summary>
/// <param name="a_lamds_uni">
/// Effective latice spacing for the first crystal.
/// </param>
/// <param name="db_lamds_uni">
/// Effective latice spacing for the second crystal.
/// </param>
/// <param name="tw_d">
/// 2 * total latice spacing for the first crystal.
/// </param>
/// <returns>
/// Wavelength of the generated event.
/// </returns>
double Util::getEnergy(double a_lamds_uni, double db_lamds_uni, double tw_d) {

    double p1, p2, natur_li, pm1, pm2, pm3, pm4, hit, rnd_inten, energy_t;
    int I_picks;

    unsigned seed;

    static thread_local std::mt19937 generator;
    std::uniform_int_distribution<uint32_t> uniform(0, RAND_MAX);

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    #ifdef OPENMP
    // random seed for this event if we have openMP enabled
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() + 17 * omp_get_thread_num();
    #else
    seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    #endif

    generator.seed(seed);


    if (FullEnergySpectrumInput.make_more_lines == 0)
        I_picks = 2;
    else if (FullEnergySpectrumInput.make_more_lines == 1) {
        if (FullEnergySpectrumInput.Do_background) {
            pm1 = 0.1875;
            pm2 = 0.34375;
            pm3 = 0.421875;
            pm4 = 0.5;
        }
        else {
            pm1 = FullEnergySpectrumInput.p1_ener;
            pm2 = pm1 + FullEnergySpectrumInput.p2_ener;
            pm3 = pm2 + FullEnergySpectrumInput.p3_ener;
            pm4 = 1;
        }


        p1 = ((double)uniform(generator) / RAND_MAX);


        if (p1 < pm1)
            I_picks = 1;
        else {
            if (p1 < pm2)
                I_picks = 2;
            else {
                if (p1 < pm3)
                    I_picks = 3;
                else {
                    if (p1 < pm4)
                        I_picks = 4;
                    else
                        I_picks = 5;
                }
            }
        }
    }
    else {
        rnd_inten = (double)uniform(generator) / RAND_MAX;
        
        energy_t = Util::splint_te(Energy_Spectrum_Vectors.cum_ints, Energy_Spectrum_Vectors.lamdas, Energy_Spectrum_Vectors.lamda_two_derivs, rnd_inten);

        return Convert_Ag_minusone_eV / energy_t;

    }

    if (FullEnergySpectrumInput.make_more_lines == 0 || FullEnergySpectrumInput.make_more_lines == 1) {
        if (I_picks == 5) {
            p2 = ((double)uniform(generator) / RAND_MAX);
            return a_lamds_uni + db_lamds_uni * p2;
        }
        else {
            hit = -1;

            while (hit < 0 || hit > tw_d) {
                p1 = ((double)uniform(generator) / RAND_MAX) * M_PI;

                natur_li = picks[I_picks - 1].natural_varia;

                hit = picks[I_picks - 1].lamda + natur_li * tan(p1);
            }
        }

        hit = Util::GaussianBox(PhysicalParametersInput.gauss_Doop, hit);
        return hit;

    }
    else {
        throw std::runtime_error("Error in intensity_source: energy could not be generated from input");
    }

}


/// <summary>
/// Function to calculate the horizontal angular limits for the source radiation.
/// </summary>
/// <param name="tetaref">
/// Radiation angle.
/// </param>
/// <param name="delrot_min">
/// Minimum angle for the second crystal rotation.
/// </param>
/// <param name="delrot_max">
/// Maximum angle for the second crystal rotation.
/// </param>
/// <param name="fi_max"></param>
/// <param name="teta_max"></param>
/// <param name="teta_min"></param>
/// <returns>
/// Vector with the minimum and maximum angles.
/// </returns>
std::vector<double> Util::getLims(double tetaref, double delrot_min, double delrot_max, double fi_max, double teta_max, double teta_min) {

    double tan_e, cos_e, teta_min_temp, corre_term1, corre_term2, teta_max_temp;

    double teta_max_out, teta_min_out;

    tan_e = tan(tetaref) / 2;
    cos_e = cos(tetaref);

    corre_term1 = tan_e * (pow(fi_max, 2) + pow(GeoParametersInput.tilt_C1, 2));
    corre_term2 = fi_max * abs(GeoParametersInput.tilt_C1) / cos_e;

    teta_min_temp = *max_element(min_angle_resp.begin(), min_angle_resp.end()) + delrot_min + corre_term1 - corre_term2;

    if (teta_min_temp > teta_min)
        teta_min_out = teta_min_temp;

    teta_max_temp = *min_element(max_angle_resp.begin(), max_angle_resp.end()) + delrot_max + corre_term1 + corre_term2;

    if (teta_max_temp < teta_max)
        teta_max_out = teta_max_temp;

    std::vector<double> res;
    res.push_back(teta_min_out);
    res.push_back(teta_max_out);

    return res;
}


/// <summary>
/// Function to initialize the plot parameters.
/// </summary>
void Util::initPlates() {

    int int_plot[6];

    double ratio_xy[6];

    double y_detc_2, z_detc_2;

    y_detc_2 = GeolengthelementsInput.ydetc / 2;
    z_detc_2 = GeolengthelementsInput.zdetc / 2;


    int_plot[0] = NumberRaysInput.nbeams - 1;
    int_plot[1] = NumberRaysInput.nbeams + 1;
    int_plot[2] = (int)(NumberRaysInput.nbeams / 100) + 1;
    int_plot[3] = 10;
    int_plot[4] = (int)(NumberRaysInput.nbeams / 100) + 1;
    int_plot[5] = 10;

    if (GraphOptionsInput.make_imageC1_After_refle) {
        int_plot[1] = 500;
        int_plot[2] = 1;
        int_plot[3] = 10;
        int_plot[4] = 1;
        int_plot[5] = 10;
    }

    if (GeoParapathlengthsInput.type_source == "UR") {
        max_plot_x[0] = GeolengthelementsInput.y_aper;
        max_plot_x[1] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[2] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[3] = y_detc_2;
        max_plot_x[4] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[5] = y_detc_2;

        max_plot_y[0] = 3 * GeolengthelementsInput.z_aper;
        max_plot_y[1] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[2] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[3] = z_detc_2;
        max_plot_y[4] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[5] = z_detc_2;
    }
    else if (GeoParapathlengthsInput.type_source == "UC") {
        max_plot_x[0] = GeolengthelementsInput.S_aper;
        max_plot_x[1] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[2] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[3] = y_detc_2;
        max_plot_x[4] = GeolengthelementsInput.y_first_crys / 2;
        max_plot_x[5] = y_detc_2;

        max_plot_y[0] = GeolengthelementsInput.S_aper;
        max_plot_y[1] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[2] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[3] = z_detc_2;
        max_plot_y[4] = GeolengthelementsInput.z_first_crys / 2;
        max_plot_y[5] = z_detc_2;
    }


    ratio_xy[0] = max_plot_x[0] / max_plot_y[0];
    ratio_xy[1] = max_plot_x[1] / max_plot_y[1];
    ratio_xy[2] = max_plot_x[2] / max_plot_y[2];
    ratio_xy[3] = max_plot_x[3] / max_plot_y[3];
    ratio_xy[4] = max_plot_x[4] / max_plot_y[4];
    ratio_xy[5] = max_plot_x[5] / max_plot_y[5];


    //TODO
    //more gui stuff


}


/// <summary>
/// Function to calculate the position of the event in the image plates.
/// </summary>
/// <param name="crystal">
/// Image plate number.
/// 1 - Source
/// 2 - First crystal
/// 3 - Second crystal Parallel
/// 4 - Detector Parallel
/// 5 - Second crystal Antiparallel
/// 6 - Detector Antiparallel
/// </param>
/// <param name="y">
/// y value of the projected event position.
/// </param>
/// <param name="z">
/// z value of the projected event position.
/// </param>
void Util::Make(int crystal, double y, double z,
                int &counts_sour,
                int &counts_C1,
                int &counts_C2_para,
                int &counts_detc_para,
                int &counts_C2_anti,
                int &counts_detc_anti) {

    double max_plot_x_temp, max_plot_y_temp;
    int nx, ny;

    
    max_plot_x_temp = max_plot_x[crystal - 1];
    max_plot_y_temp = max_plot_y[crystal - 1];

    nx = (int)(nm2 * y / max_plot_x_temp + np2);
    ny = (int)(nm2 * z / max_plot_y_temp + np2);


    if (crystal == 1) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            hist_image_plate_source[i]++;
            counts_sour++;

            if (max_hist[crystal - 1] < hist_image_plate_source[i])
                max_hist[crystal - 1] = hist_image_plate_source[i];
        }
    }
    else if (crystal == 2) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            hist_image_plate_crystal1[i]++;
            counts_C1++;

            if (max_hist[crystal - 1] < hist_image_plate_crystal1[i])
                max_hist[crystal - 1] = hist_image_plate_crystal1[i];
        }
    }
    else if (crystal == 3) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            hist_image_plate_crystal2_para[i]++;
            counts_C2_para++;

            if (max_hist[crystal - 1] < hist_image_plate_crystal2_para[i])
                max_hist[crystal - 1] = hist_image_plate_crystal2_para[i];
        }
    }
    else if (crystal == 4) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            hist_image_plate_detc_para[i]++;
            counts_detc_para++;

            if (max_hist[crystal - 1] < hist_image_plate_detc_para[i])
                max_hist[crystal - 1] = hist_image_plate_detc_para[i];
        }
    }
    else if (crystal == 5) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            hist_image_plate_crystal2_anti[i]++;
            counts_C2_anti++;

            if (max_hist[crystal - 1] < hist_image_plate_crystal2_anti[i])
                max_hist[crystal - 1] = hist_image_plate_crystal2_anti[i];
        }
    }
    else if (crystal == 6) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            hist_image_plate_detc_anti[i]++;
            counts_detc_anti++;

            if (max_hist[crystal - 1] < hist_image_plate_detc_anti[i])
                max_hist[crystal - 1] = hist_image_plate_detc_anti[i];
        }
    }
}


/// <summary>
/// Function to fit the simulated spectrum with Voigt profiles.
/// </summary>
/// <param name="Parallel">
/// If we are fitting the Parallel or Antiparallel results.
/// </param>
void Util::fit(bool Parallel) {

    std::stringstream logString;

    std::vector<double> x, y, sig;

    double gues[MA], a[MA], alamda, a_error_anti_voig[MA], FWMH_V, chisq, ochisq;

    std::vector<std::vector<double>> covar(MA, std::vector<double>(MA)), alpha(MA, std::vector<double>(MA));

    int NPT, ia[MA], npt_m_deg, k, itst;

    std::string name_varia[MA];

    NPT = PlotParametersInput.nubins;

    UserSettingsInput.TrueVoigt = true;

    gener_out << "------------------------------------" << std::endl;
    gener_out << std::endl;
    gener_out << " Fitting analysies" << std::endl;
    gener_out << std::endl;

    logString << "------------------------------------" << std::endl;
    logString << std::endl;
    logString << " Fitting analysies" << std::endl;
    logString << std::endl;

    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #else
    std::cout << logString.str();
    #endif

    if (Parallel) {
        for (int i = 0; i < PlotParametersInput.nubins; i++) {
            x.push_back(Data_Fit_para[i].degree);
            y.push_back(Data_Fit_para[i].reflecti_total);
            sig.push_back(Data_Fit_para[i].reflecti_two_deriv);
        }

        gues[1] = amplitu_con_para;
        gues[4] = const_back_para;
        gues[3] = peak_posi_para;
        gues[0] = width_Gaus_para;
        gues[2] = width_Lore_para;

        ia[1] = do_amplitu_con_para;
        ia[4] = do_const_back_para;
        ia[3] = do_firstcryst;
        ia[0] = do_Gwidth_para;
        ia[2] = do_Lwidth_para;

        name_varia[1] = "Amplitude of parallel-";
        name_varia[4] = "Background of parallel-";
        name_varia[3] = "position of peak parallel-";
        name_varia[0] = "Gauss width of peak parallel-";
        name_varia[2] = "Lorentz with of peak parallel-";
    }
    else {
        for (int i = 0; i < PlotParametersInput.nubins; i++) {
            x.push_back(Data_Fit_anti[i].degree);
            y.push_back(Data_Fit_anti[i].reflecti_total);
            sig.push_back(Data_Fit_anti[i].reflecti_two_deriv);
        }

        gues[1] = amplitu_con_anti;
        gues[4] = const_back_anti;
        gues[3] = peak_posi_anti;
        gues[0] = width_Gaus_anti;
        gues[2] = width_Lore_anti;

        ia[1] = do_amplitu_con_anti;
        ia[4] = do_const_back_anti;
        ia[3] = do_firstcryst_anti;
        ia[0] = do_Gwidth_anti;
        ia[2] = do_Lwidth_anti;

        name_varia[1] = "Amplitude of antiparallel-";
        name_varia[4] = "Background of antiparallel-";
        name_varia[3] = "position of peak antiparallel-";
        name_varia[0] = "Gauss width of peak antiparallel-";
        name_varia[2] = "Lorentz with of peak antiparallel-";
    }

    for (int i = 0; i < MA; i++) {
        a[i] = gues[i];
        npt_m_deg = NPT - ia[i];
    }

    k = 1;
    alamda = -1;

    if (UserSettingsInput.TrueVoigt) {
        Util::mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Util::Voig, alamda);
    }
    else {
        Util::mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Util::Pseud, alamda);
    }

    itst = 0;

    while (itst < 6) {
        k++;

        ochisq = chisq;

        if (UserSettingsInput.TrueVoigt) {
            Util::mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Util::Voig, alamda);
        }
        else {
            Util::mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Util::Pseud, alamda);
        }

        if (chisq > ochisq)
            itst = 0;
        else if (abs((ochisq - chisq) / chisq) < 10E-6)
            itst++;

        if (abs(ochisq - chisq) > 10)
            itst = 0;
    }

    alamda = 0;

    if (UserSettingsInput.TrueVoigt) {
        Util::mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Util::Voig, alamda);
    }
    else {
        Util::mrq_min(x, y, sig, NPT, a, ia, MA, covar, alpha, MA, chisq, Util::Pseud, alamda);
    }
    
    
    logString.clear();
    for (int i = 0; i < MA; i++) {
        gener_out << name_varia[i] << "\t" << a[i] << std::endl;
        
        logString << name_varia[i] << "\t" << a[i] << std::endl;
        
        a_error_anti_voig[i] = sqrt(covar[i][i] * npt_m_deg);
    }
    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #else
    std::cout << logString.str();
    #endif

    FWMH_V = c1_coef * a[2] + sqrt(c2_coef * pow(a[2], 2) + pow(a[0], 2));
    if (UserSettingsInput.TrueVoigt) {
        if (Parallel)
            FWMH_V_para = FWMH_V;
        else
            FWMH_V_anti = FWMH_V;
    }
    else {
        if (Parallel)
            FWMH_V_para = a[0];
        else
            FWMH_V_anti = a[0];
    }

    gener_out << "FWMH-\t" << FWMH_V << std::endl;
    gener_out << std::endl;

    logString.clear();
    logString << "FWMH-\t" << FWMH_V << std::endl;
    logString << std::endl;
    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #else
    std::cout << logString.str();
    #endif

    gener_out << std::endl;
    gener_out << "Reduced Chi-squared:\t" << chisq / npt_m_deg << std::endl;
    gener_out << std::endl;
    gener_out << "Uncertainties:" << std::endl;
    gener_out << sqrt(covar[0][0]) << "\t" << MA << std::endl;


    if (Parallel)
        std::copy(std::begin(a), std::end(a), std::begin(a_para_voig));
    else
        std::copy(std::begin(a), std::end(a), std::begin(a_anti_voig));


}


/// <summary>
/// Function to read the crystal curve responces.
/// </summary>
void Util::Read_CurveResponce() {

    std::stringstream logString;

    std::vector<std::string> _available_energies;

    bool exist_file;
    double y1d, yud;// theta_dire;

    std::ifstream pathEnergies;
    std::ifstream pathFile;

    if (GeometryInput.mode_bragg_geo) {

        char inEnergies[1024] = "";
        strcat(inEnergies, File_simu);
        strcat(inEnergies, "\\Crystal_Profiles");
        strcat(inEnergies, "\\sorted_ens.txt");

        struct stat buffer;
        exist_file = (stat(inEnergies, &buffer) == 0);

        if (exist_file)
            pathEnergies.open(inEnergies);
        else {
            logString << "File \"sorted_ens.txt\" does not exist" << std::endl;

            throw std::runtime_error("A file with the available crystal profile energies is required.");
        }


        std::string ener;
        if (pathEnergies.is_open()) {
            while (pathEnergies >> ener) {
                _available_energies.push_back(ener);
                available_energies.push_back(stod(ener) * 1000);
            }
        }

    }
    else {

        char inFile[223] = "";
        strcat(inFile, File_simu);
        strcat(inFile, "\\Plot_crystall_responce_transm.txt");


        struct stat buffer;
        exist_file = (stat(inFile, &buffer) == 0);

        if (exist_file)
            pathFile.open(inFile);
        else {
            logString << "File \"Plot_crystal_responce_transm.txt\" does not exist" << std::endl;
            
            throw std::runtime_error("A file with crystall responce from XOP in Laue geometry is required");
        }

    }
    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #else
    std::cout << logString.str();
    #endif

    double cel_re1, cel_re2;

    int energyIndex = 0;

    for (std::string ener : _available_energies) {
        std::ifstream pathFile_p;

        char inFile_p[1024] = "";
        strcat(inFile_p, File_simu);
        strcat(inFile_p, "\\Crystal_Profiles");
        strcat(inFile_p, (std::string("\\") + ener + std::string("keV_p")).c_str());

        struct stat buffer1;
        exist_file = (stat(inFile_p, &buffer1) == 0);

        if (exist_file) {
            pathFile_p.open(inFile_p);
            //cout << "Reading file: " << ener << "keV_p" << endl;
        }
        else {
            logString.clear();
            logString << "File " << inFile_p << " does not exist" << std::endl;
            #ifdef QT_EXISTS
            if(logBox != NULL)
                logBox->appendPlainText(logString.str().c_str());
            #else
            std::cout << logString.str();
            #endif

            throw std::runtime_error("Expected a crystall responce from XOP in bragg geometry for energy" + ener);
        }

        std::vector<plotresponc> responce;
        if (pathFile_p.is_open()) {
            while (pathFile_p >> cel_re1 >> cel_re2) {
                plotresponc tmp;
                tmp.degree = cel_re1 / one_micro;
                tmp.reflecti_total = 0;
                tmp.reflecti_two_deriv = 0;
                tmp.reflecti_total_s = 0;
                tmp.reflecti_two_deriv_s = 0;
                tmp.reflecti_total_p = cel_re2;
                tmp.reflecti_two_deriv_p = 0;
                responce.push_back(tmp);
            }

            pathFile_p.close();
        }

        std::ifstream pathFile_s;

        char inFile_s[1024] = "";
        strcat(inFile_s, File_simu);
        strcat(inFile_s, "\\Crystal_Profiles");
        strcat(inFile_s, (std::string("\\") + ener + std::string("keV_s")).c_str());

        struct stat buffer2;
        exist_file = (stat(inFile_s, &buffer2) == 0);

        if (exist_file) {
            pathFile_s.open(inFile_s);
            //cout << "Reading file: " << ener << "keV_p" << endl;
        }
        else {
            logString.clear();
            logString << "File " << inFile_s << " does not exist" << std::endl;
            #ifdef QT_EXISTS
            if(logBox != NULL)
                logBox->appendPlainText(logString.str().c_str());
            #else
            std::cout << logString.str();
            #endif

            throw std::runtime_error("Expected a crystall responce from XOP in bragg geometry for energy" + ener);
        }

        int index = 0;
        if (pathFile_s.is_open()) {
            while (pathFile_s >> cel_re1 >> cel_re2) {
                responce[index].reflecti_total = responce[index].reflecti_total_p + cel_re2;
                responce[index].reflecti_total_s = cel_re2;

                index++;
            }

            pathFile_s.close();
        }

        plotresponce_mide.push_back(responce);

        min_angle_resp.push_back(plotresponce_mide[energyIndex][0].degree);
        max_angle_resp.push_back(plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree);

        y1d = (plotresponce_mide[energyIndex][1].reflecti_total - plotresponce_mide[energyIndex][0].reflecti_total) / (plotresponce_mide[energyIndex][1].degree - plotresponce_mide[energyIndex][0].degree);
        yud = (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].reflecti_total - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].reflecti_total) / (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].degree);

        //Total
        std::vector<double> x, y, y2;

        for (unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++) {
            x.push_back(plotresponce_mide[energyIndex][i].degree);
            y.push_back(plotresponce_mide[energyIndex][i].reflecti_total);
        }

        y2 = Util::spline(x, y, y1d, yud);

        for (unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++) {
            plotresponce_mide[energyIndex][i].reflecti_two_deriv = y2[i];
        }


        y1d = (plotresponce_mide[energyIndex][1].reflecti_total_s - plotresponce_mide[energyIndex][0].reflecti_total_s) / (plotresponce_mide[energyIndex][1].degree - plotresponce_mide[energyIndex][0].degree);
        yud = (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].reflecti_total_s - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].reflecti_total_s) / (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].degree);

        //S Polarization
        std::vector<double> x_s, y_s, y2_s;

        for (unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++) {
            x_s.push_back(plotresponce_mide[energyIndex][i].degree);
            y_s.push_back(plotresponce_mide[energyIndex][i].reflecti_total_s);
        }

        y2_s = Util::spline(x_s, y_s, y1d, yud);

        for (unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++) {
            plotresponce_mide[energyIndex][i].reflecti_two_deriv_s = y2_s[i];
        }


        y1d = (plotresponce_mide[energyIndex][1].reflecti_total_p - plotresponce_mide[energyIndex][0].reflecti_total_p) / (plotresponce_mide[energyIndex][1].degree - plotresponce_mide[energyIndex][0].degree);
        yud = (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].reflecti_total_p - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].reflecti_total_p) / (plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 1].degree - plotresponce_mide[energyIndex][plotresponce_mide[energyIndex].size() - 2].degree);

        //P Polarization
        std::vector<double> x_p, y_p, y2_p;

        for (unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++) {
            x_p.push_back(plotresponce_mide[energyIndex][i].degree);
            y_p.push_back(plotresponce_mide[energyIndex][i].reflecti_total_p);
        }

        y2_p = Util::spline(x_p, y_p, y1d, yud);

        for (unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++) {
            plotresponce_mide[energyIndex][i].reflecti_two_deriv_p = y2_p[i];
        }


        std::vector<double> degrees, reflecti_totals, reflecti_two_derivs, reflecti_total_ps, reflecti_two_deriv_ps, reflecti_total_ss, reflecti_two_deriv_ss;

        for (unsigned int i = 0; i < plotresponce_mide[energyIndex].size(); i++) {
            degrees.push_back(plotresponce_mide[energyIndex][i].degree);
            reflecti_totals.push_back(plotresponce_mide[energyIndex][i].reflecti_total);
            reflecti_two_derivs.push_back(plotresponce_mide[energyIndex][i].reflecti_two_deriv);
            reflecti_total_ps.push_back(plotresponce_mide[energyIndex][i].reflecti_total_p);
            reflecti_two_deriv_ps.push_back(plotresponce_mide[energyIndex][i].reflecti_two_deriv_p);
            reflecti_total_ss.push_back(plotresponce_mide[energyIndex][i].reflecti_total_s);
            reflecti_two_deriv_ss.push_back(plotresponce_mide[energyIndex][i].reflecti_two_deriv_s);
        }

        plotresponc_vecs responce_vecs = {
            degrees,
            reflecti_totals,
            reflecti_two_derivs,
            reflecti_total_ps,
            reflecti_two_deriv_ps,
            reflecti_total_ss,
            reflecti_two_deriv_ss
        };

        Crystal_Responces.push_back(responce_vecs);

        energyIndex++;
    }
}


/// <summary>
/// Function to calculate the horizontal angular correction.
/// </summary>
/// <param name="y_pro_C1">
/// Low limit for reflextion on the y axis.
/// </param>
/// <param name="y_max_C1">
/// High limit for reflextion on the y axis.
/// </param>
/// <param name="z_pro_C1">
/// Low limit for reflextion on the z axis.
/// </param>
/// <param name="z_max_C1">
/// High limit for reflextion on the z axis.
/// </param>
/// <param name="type_c">
/// If the correction is for the first or second crystal.
/// </param>
/// <returns>
/// Vector with the correction for the y and z axis.
/// </returns>
///TODO: optimize
std::vector<double> Util::horCorr(double y_pro_C1, double y_max_C1, double z_pro_C1, double z_max_C1, bool type_c) {

    double R_cur_crys_t;

    if (type_c)
        R_cur_crys_t = CurvedCrystalInput.R_cur_crys_1;
    else
        R_cur_crys_t = CurvedCrystalInput.R_cur_crys_2;

    std::vector<double> res;

    res.push_back(cos(2 * R_cur_crys_t * y_pro_C1 * (1 - pow((z_pro_C1 / z_max_C1), 2)) / pow(y_max_C1, 2)));
    res.push_back(sin(2 * R_cur_crys_t * y_pro_C1 * (1 - pow((z_pro_C1 / z_max_C1), 2)) / pow(y_max_C1, 2)));

    return res;

}


/// <summary>
/// Function to sort the results into variables to use in the fitting algorithm.
/// </summary>
/// <param name="numbins">
/// Result bin to sort.
/// </param>
/// <param name="angle_para">
/// Parallel second crystal angle.
/// </param>
/// <param name="toint_para">
/// Total number of events for the results in the parallel bin.
/// </param>
/// <param name="angle_anti">
/// Antiparallel second crystal angle.
/// </param>
/// <param name="toint_anti">
/// Total number of events for the results in the antiparallel bin.
/// </param>
void Util::FitData(int numbins, double angle_para, int toint_para, double angle_anti, int toint_anti) {

    if (Data_Fit_anti.size() < (unsigned int)numbins)
        Data_Fit_anti.resize(PlotParametersInput.nubins);
    if (Data_Fit_para.size() < (unsigned int)numbins)
        Data_Fit_para.resize(PlotParametersInput.nubins);

    Data_Fit_para.at(numbins - 1).degree = angle_para;
    Data_Fit_para.at(numbins - 1).reflecti_total = (double)(toint_para + 10);
    Data_Fit_para.at(numbins - 1).reflecti_two_deriv = sqrt((double)(toint_para + 10));

    Data_Fit_anti.at(numbins - 1).degree = angle_anti;
    Data_Fit_anti.at(numbins - 1).reflecti_total = (double)(toint_anti + 10);
    Data_Fit_anti.at(numbins - 1).reflecti_two_deriv = sqrt((double)(toint_anti + 10));
}


/// <summary>
/// Function to read the source energy spectrum.
/// The cumulative distribution for the spectrum is also calculated
/// to be used for the random energy generation.
/// </summary>
void Util::Read_EnergySpectrum(const std::string& filename) {

    bool exist_file;
    double y1d, yud;
    double x1d, xud;
    int mult;

    std::ifstream pathFile;
    const char* inFile = filename.c_str();

    struct stat buffer;
    exist_file = (stat(inFile, &buffer) == 0);

    if (exist_file) {
        pathFile.open(inFile);
    }
    else {
        throw std::runtime_error("input energy spectrum file \"" + filename + "\" not found");
    }

    if (Unit_energy == "keV")
        mult = 1000;
    else if (Unit_energy == "eV")
        mult = 1;
    else
        mult = 1;

    double cel_re1, cel_re2;
    double cum = 0;

    if (pathFile.is_open()) {
        while (pathFile >> cel_re1 >> cel_re2) {
            cum += cel_re2;

            energy_gen tmp;
            tmp.lamda = cel_re1 * mult;
            tmp.intensity = cel_re2;
            tmp.cum_int = cum;
            tmp.intensity_two_deriv = 0;
            tmp.lambda_two_deriv = 0;
            Energy_spec.push_back(tmp);
        }

        pathFile.close();
    }

    y1d = (Energy_spec[1].intensity - Energy_spec[0].intensity) / (Energy_spec[1].lamda - Energy_spec[0].lamda);
    yud = (Energy_spec[Energy_spec.size() - 1].intensity - Energy_spec[Energy_spec.size() - 2].intensity) / (Energy_spec[Energy_spec.size() - 1].lamda - Energy_spec[Energy_spec.size() - 2].lamda);

    std::vector<double> x, y, yc, ycn, x2, y2;

    for (unsigned int i = 0; i < Energy_spec.size(); i++) {
        x.push_back(Energy_spec[i].lamda);
        y.push_back(Energy_spec[i].intensity);
        yc.push_back(Energy_spec[i].cum_int);
    }

    double min_cum = *min_element(yc.begin(), yc.end());
    double max_cum = *max_element(yc.begin(), yc.end()) - min_cum;

    for (unsigned int i = 0; i < Energy_spec.size(); i++) {
        Energy_spec[i].cum_int -= min_cum;
        Energy_spec[i].cum_int /= max_cum;
        ycn.push_back(Energy_spec[i].cum_int);
    }

    x1d = (Energy_spec[1].lamda - Energy_spec[0].lamda) / (Energy_spec[1].cum_int - Energy_spec[0].cum_int);
    xud = (Energy_spec[Energy_spec.size() - 1].lamda - Energy_spec[Energy_spec.size() - 2].lamda) / (Energy_spec[Energy_spec.size() - 1].cum_int - Energy_spec[Energy_spec.size() - 2].cum_int);

    y2 = Util::spline(x, y, y1d, yud);
    x2 = Util::spline(ycn, x, x1d, xud);

    for (unsigned int i = 0; i < Energy_spec.size(); i++) {
        Energy_spec[i].intensity_two_deriv = y2[i];
        Energy_spec[i].lambda_two_deriv = x2[i];
    }

    std::vector<double> lamdas, intensities, cum_ints, intensity_two_derivs, lamda_two_derivs;

    for (unsigned int i = 0; i < Energy_spec.size(); i++) {
        lamdas.push_back(Energy_spec[i].lamda);
        intensities.push_back(Energy_spec[i].cum_int);
        cum_ints.push_back(Energy_spec[i].cum_int);
        intensity_two_derivs.push_back(Energy_spec[i].intensity_two_deriv);
        lamda_two_derivs.push_back(Energy_spec[i].lambda_two_deriv);
    }

    Energy_Spectrum_Vectors = {
        lamdas,
        intensities,
        cum_ints,
        intensity_two_derivs,
        lamda_two_derivs
    };

}


/// <summary>
/// Function to calculate the effective angular misalign from the source to the detector.
/// </summary>
/// <param name="Dis_total">
/// Total distance the radiation travels between the source and detector.
/// </param>
/// <returns>
/// Vector with the angular and z axis interaction limits.
/// </returns>
std::vector<double> Util::misalign(double Dis_total) {

    double dev, term_arc, Geo_para_matr[4][4], tetaref, fi_max, fi_min, z_max, z_min;

    //Old unused variables
    //double len, hei, div, fi_temp;

    fi_max = 10;
    fi_min = -10;

    z_max = GeolengthelementsInput.zdetc / 2;
    z_min = -GeolengthelementsInput.zdetc / 2;
    tetaref = M_PI / 2 - teta_crys1;

    Geo_para_matr[0][0] = GeolengthelementsInput.S_aper / 2;
    Geo_para_matr[0][1] = -GeolengthelementsInput.S_aper / 2;
    Geo_para_matr[0][2] = GeoParapathlengthsInput.dist_T_Cr1;
    Geo_para_matr[0][3] = GeoParametersInput.xsi;

    Geo_para_matr[1][0] = GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[1][1] = -GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[1][2] = GeoParapathlengthsInput.dist_T_Cr1 / 2;
    Geo_para_matr[1][3] = GeoParametersInput.xsi - 2 * GeoParametersInput.tilt_C1 * sin(tetaref) * GeoParapathlengthsInput.dist_Cr1_Cr2 / (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2);

    Geo_para_matr[2][0] = GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[2][1] = -GeolengthelementsInput.z_first_crys / 2;
    Geo_para_matr[2][2] = GeoParapathlengthsInput.dist_Cr2_Det;
    Geo_para_matr[2][3] = GeoParametersInput.xsi - 2 * GeoParametersInput.tilt_C1 * sin(tetaref) * (GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det) /
        (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det) - 2 * GeoParametersInput.tilt_C2 * sin(tetaref) *
        GeoParapathlengthsInput.dist_Cr2_Det / (GeoParapathlengthsInput.dist_T_Cr1 + GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det);

    Geo_para_matr[3][0] = z_max;
    Geo_para_matr[3][1] = -z_max;

    if (GeoParapathlengthsInput.type_source == "P")
        term_arc = z_max;
    else
        term_arc = (GeolengthelementsInput.S_aper + GeolengthelementsInput.zdetc) / 2;


    fi_max = atan(term_arc / Dis_total) + GeoParametersInput.xsi;
    fi_min = -atan(term_arc / Dis_total) + GeoParametersInput.xsi;

    dev = term_arc + Dis_total * tan(GeoParametersInput.xsi - atan(term_arc / Dis_total));

    z_max = GeolengthelementsInput.zdetc / 2 + dev;
    z_min = -GeolengthelementsInput.zdetc / 2 + dev;

    std::vector<double> res;
    res.push_back(fi_max);
    res.push_back(fi_min);
    res.push_back(z_max);
    res.push_back(z_min);

    return res;
}


/// <summary>
/// Function to convert angles from degree to radians.
/// </summary>
void Util::Set_angs() {

    if (never_set_angle) {
        //tetaref *= convrad;
        GeoParametersInput.xsi *= convrad;
        teta_crys1 *= convrad;
        GeoParametersInput.tilt_C1 *= convrad;
        GeoParametersInput.tilt_C2 *= convrad;

        //Old vars
        //stan_V_C1 *= convrad;
        //stan_H_C1 *= convrad;
        //stan_V_C2 *= convrad;
        //stan_H_C2 *= convrad;

        GeoParametersInput.OffsetRotCry1 *= convrad;

        GeoParametersInput.teta_table *= convrad;
        GeoParametersInput.table_resolution *= convrad;
        
        GeoParametersInput.teta_detec_para *= convrad;
        GeoParametersInput.teta_detec_anti *= convrad;

        //Old vars
        //y_first_crys_2 /= 2;
        //z_first_crys_2 /= 2;
        //step_x_hist_y_crys /= 5;
        //step_x_hist_z_crys /= 5;
        //S_aper_D_2 *= 2;

        never_set_angle = false;
    }
}


/// <summary>
/// Function to test the input parameters for the energy of the source radiation.
/// </summary>
void Util::test_In() {

    std::stringstream logString;

    double b_anti_pick;

    b_anti_pick = -2 * (-teta_crys1 + theta_chk);

    double tetaref, teta_table_rad, xsi_rad, c1, C2_para, C2_anti, tan_e2, cos_e2, cos_e, esti_para, esti_anti, angle_dif, energy_exact, wave, energ;
    double low_bound_angl_para, high_bound_angl_para, low_bound_angl_anti, high_bound_angl_anti, Maxi_angl_para, Mini_angl_para, Maxi_angl_anti, Mini_angl_anti, var_temp;

    tetaref = 90 - teta_crys1;
    tetaref *= convrad;
    teta_table_rad = GeoParametersInput.teta_table * convrad;
    xsi_rad = GeoParametersInput.xsi * convrad;

    if (!UserSettingsInput.Simple_simu) {
        xsi_rad = -atan((GeolengthelementsInput.S_shi_ver_A - GeolengthelementsInput.S_shi_ver_B) / GeoParapathlengthsInput.LT_aper);
        GeoParametersInput.xsi = xsi_rad * convdeg;
    }

    if (CurveVerticalTiltInput.make_CurveTilt) {
        c1 = Util::ObtainVert(1, 0);

        C2_para = Util::ObtainVert(2, teta_crys1);
        C2_anti = Util::ObtainVert(2, -teta_crys1);
    }
    else {
        c1 = GeoParametersInput.tilt_C1 * convrad;
        C2_para = GeoParametersInput.tilt_C2 * convrad;
        C2_anti = GeoParametersInput.tilt_C2 * convrad;
    }


    tan_e2 = tan(tetaref) / 2;
    cos_e2 = 2 * cos(2 * tetaref);
    cos_e = cos(tetaref);


    esti_para = convdeg * tan_e2 * (pow(C2_para, 2) + 4 * C2_para * c1 + 3 * pow(c1, 2)) - convdeg * xsi_rad * (C2_para + c1) / cos_e;
    esti_anti = convdeg * tan_e2 * (pow(C2_anti, 2) + 4 * C2_anti * c1 + pow(c1, 2) * (1 - 2 * cos_e2)) + convdeg * tan_e2 * (2 * pow(xsi_rad, 2)) - convdeg * xsi_rad * (C2_anti + c1 * (1 - cos_e2)) / cos_e;

    if (!UserSettingsInput.Simple_simu) {
        esti_para -= convdeg * (2 * tetaref - teta_table_rad);
        esti_anti += convdeg * (2 * tetaref - teta_table_rad);
    }

    esti_anti += b_anti_pick + 2 * refra_corr + 2 * vert_div_corr;

    if (UserSettingsInput.see_para) {
        gener_out << "The estimated angle of peak" << std::endl;
        gener_out << "taking into account crystal tilts and effective misalignement" << std::endl;

        logString << "The estimated angle of peak" << std::endl;
        logString << "taking into account crystal tilts and effective misalignement" << std::endl;

        ang_para_pre = teta_crys1 - esti_para;

        gener_out << " in parallel is:					" << ang_para_pre << " deg" << std::endl;
        gener_out << " or, in turn of reference angle:	" << -esti_para << " deg" << std::endl;
        gener_out << std::endl;

        logString << " in parallel is:					" << ang_para_pre << " deg" << std::endl;
        logString << " or, in turn of reference angle:	" << -esti_para << " deg" << std::endl;
        logString << std::endl;
        #ifdef QT_EXISTS
        if(logBox != NULL)
            logBox->appendPlainText(logString.str().c_str());
        #endif
    }

    if (UserSettingsInput.see_anti) {
        gener_out << "The estimated angle of peak" << std::endl;
        gener_out << "taking into account crystal tilts and effective misalignement" << std::endl;

        logString.clear();
        logString << "The estimated angle of peak" << std::endl;
        logString << "taking into account crystal tilts and effective misalignement" << std::endl;

        ang_anti_pre = esti_anti - teta_crys1;

        gener_out << " in antiparallel is:				" << ang_anti_pre << " deg" << std::endl;
        gener_out << " or, in turn of reference angle:	" << -esti_anti << " deg" << std::endl;
        gener_out << std::endl;


        logString << " in antiparallel is:				" << ang_anti_pre << " deg" << std::endl;
        logString << " or, in turn of reference angle:	" << -esti_anti << " deg" << std::endl;
        logString << std::endl;
        #ifdef QT_EXISTS
        if(logBox != NULL)
            logBox->appendPlainText(logString.str().c_str());
        #endif
    }


    if (UserSettingsInput.see_para && UserSettingsInput.see_anti) {
        angle_dif = (ang_anti_pre - ang_para_pre) / 2 + 90 - refra_corr - vert_div_corr;

        angle_dif *= convrad;
        energy_exact = Convert_Ag_minusone_eV / picks[0].lamda;
        wave = 2 * d_lat * sin(angle_dif);
        energ = Convert_Ag_minusone_eV / wave;

        gener_out << std::endl;
        gener_out << " Estimated energy" << std::endl;
        gener_out << " without crystal tilts and effective misalignement: " << energy_exact << std::endl;
        gener_out << std::endl;
        gener_out << " with crystal tilts and effective misalignement: " << energ << std::endl;
        gener_out << std::endl;
        gener_out << "difference: " << (energ - energy_exact) * 1000 << "MeV" << std::endl;
    }


    if (ang_para_pre > 0) {
        low_bound_angl_para = ang_para_pre - PlotParametersInput.delta_angl;
        high_bound_angl_para = ang_para_pre + PlotParametersInput.delta_angl;
        low_bound_angl_anti = ang_anti_pre - PlotParametersInput.delta_angl;
        high_bound_angl_anti = ang_anti_pre + PlotParametersInput.delta_angl;

        logString.clear();
        logString << "With current setting, range to observe parallel peak: " << low_bound_angl_para << " deg; " << high_bound_angl_para << " deg" << std::endl;
        logString << "With current setting, range to observe antiparallel peak: " << low_bound_angl_anti << " deg; " << high_bound_angl_anti << " deg" << std::endl;

        logString << "Estimated antiparallel angle: " << esti_anti << " teta_crys1 = " << teta_crys1 << std::endl;
        logString << "Estimated parallel angle: " << esti_para << std::endl;

        Maxi_angl_anti = high_bound_angl_anti - teta_crys1;
        Mini_angl_anti = low_bound_angl_anti - teta_crys1;

        Maxi_angl_para = high_bound_angl_para - teta_crys1;
        Mini_angl_para = low_bound_angl_para - teta_crys1;


        Maxi_angl = Maxi_angl_para + PlotParametersInput.shift_disp_window;
        Mini_angl = Mini_angl_para + PlotParametersInput.shift_disp_window;

        logString << " Maxi_angl_anti: " << Maxi_angl_anti << std::endl;
        logString << " Mini_angl_anti: " << Mini_angl_anti << std::endl;
        logString << " Maxi_angl_para: " << Maxi_angl_para << std::endl;
        logString << " Mini_angl_para: " << Mini_angl_para << std::endl;
        logString << " *** Mini angl = " << Mini_angl << " Maxi_angl = " << Maxi_angl << std::endl;

        low_bound_angl_para = teta_crys1 + Mini_angl;
        high_bound_angl_para = teta_crys1 + Maxi_angl;
        low_bound_angl_anti = -teta_crys1 - Mini_angl;
        high_bound_angl_anti = -teta_crys1 - Maxi_angl;

        logString << " Checking common range to observe parallel peak -> " << low_bound_angl_para << " deg; " << high_bound_angl_para << " deg; peak: " << ang_para_pre << " deg" << std::endl;
        logString << " Checking common range to observe antiparallel peak -> " << low_bound_angl_anti << " deg; " << high_bound_angl_anti << " deg; peak: " << ang_anti_pre << " deg" << std::endl;
        #ifdef QT_EXISTS
        if(logBox != NULL)
            logBox->appendPlainText(logString.str().c_str());
        #endif
    }
    else
        throw std::runtime_error(" Error in test_input: ang_para_pre <= 0");


    gener_out << std::endl;
    gener_out << "------------------------------" << std::endl;
    gener_out << std::endl;
    gener_out << " Estimated analysis based on analytical expressions" << std::endl;
    gener_out << std::endl;


    var_temp = Maxi_angl - Mini_angl;

    if (var_temp < 0)
        throw std::runtime_error("*** Error in test_input: min_angle grater than max_angle");
    else if (var_temp > 4)
        throw std::runtime_error("*** Error in test_input: maximum angle variation > 4 deg, too large");


    logString.clear();

    if (UserSettingsInput.see_para) {
        if (UserSettingsInput.see_anti) {
            logString << "input range set up for observing both parallel and antiparallel" << std::endl;

            logString << Mini_angl << "\t" << Maxi_angl << std::endl;

            if (abs(Mini_angl) > 4 || abs(Maxi_angl) > 4)
                throw std::runtime_error("*** Error in test_input: minimum or maximum range > 4 deg, too large");

            if (FullEnergySpectrumInput.make_more_lines < 2) {
                if (Maxi_angl < esti_anti || Mini_angl > esti_anti) {
                    logString << "In this range the antiparallel peak will not be observed" << std::endl;
                    logString << "Mini_angl = " << Mini_angl - teta_crys1 << " Maxi_angl = " << Maxi_angl - teta_crys1 << std::endl;
                    logString << "Mini_angl(ref) = " << Mini_angl << " Maxi_angl(ref) = " << Maxi_angl << " estimated antiparallel angle " << esti_anti << std::endl;
                    logString << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << std::endl;
                    logString << " setting maximum number of x rays in simulation to 2011 to check image" << std::endl;

                    NumberRaysInput.nbeams = 2011;
                }
            }
        }
        else {
            Mini_angl -= teta_crys1;
            Maxi_angl -= teta_crys1;
        }


        if (Maxi_angl < -esti_para || Mini_angl > -esti_para) {
            logString << "In this range the parallel peak will not be observed" << std::endl;
            logString << "Mini_angl = " << Mini_angl - teta_crys1 << " Maxi_angl = " << Maxi_angl - teta_crys1 << std::endl;
            logString << "Mini_angl(ref) = " << Mini_angl << " Maxi_angl(ref) = " << Maxi_angl << " estimated parallel angle " << esti_para << std::endl;
            logString << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << std::endl;
            logString << " setting maximum number of x rays in simulation to 2011 to check image" << std::endl;

            NumberRaysInput.nbeams = 2011;
        }

    }
    else {
        if (UserSettingsInput.see_anti) {
            Mini_angl += teta_crys1;
            Maxi_angl += teta_crys1;

            if (Maxi_angl < esti_anti || Mini_angl > esti_anti) {
                logString << "In this range the antiparallel peak will not be observed" << std::endl;
                logString << "Mini_angl = " << Mini_angl - teta_crys1 << " Maxi_angl = " << Maxi_angl - teta_crys1 << std::endl;
                logString << "Mini_angl(ref) = " << Mini_angl << " Maxi_angl(ref) = " << Maxi_angl << " estimated antiparallel angle " << esti_anti << std::endl;
                logString << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << std::endl;
                logString << " setting maximum number of x rays in simulation to 2011 to check image" << std::endl;

                NumberRaysInput.nbeams = 2011;
            }
        }
        else {
            logString << "No peak selected" << std::endl;
            logString << "*** In test_input, warning: bad input for Mini_angl and Maxi_angl, peak may not be observed ***" << std::endl;
            logString << " setting maximum number of x rays in simulation to 2011 to check image" << std::endl;

            NumberRaysInput.nbeams = 2011;
        }
    }
    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #endif

    logString.clear();
    if (GeoParapathlengthsInput.type_source == "P") {
        logString << " Evaluation set for a point source" << std::endl;
        logString << std::endl;
    }
    else if (GeoParapathlengthsInput.type_source == "UC") {
        logString << " Evaluation set for an uniform circular source" << std::endl;
        logString << std::endl;
    }
    else if (GeoParapathlengthsInput.type_source == "UR") {
        logString << " Evaluation set for an uniform rectangular source" << std::endl;
        logString << std::endl;
    }
    else if (GeoParapathlengthsInput.type_source == "G") {
        logString << " Evaluation set for a gaussian source" << std::endl;
        logString << std::endl;
    }
    else {
        logString << "Bad input in the type_source" << std::endl;
        logString << std::endl;
    }

    if (!UserSettingsInput.Make_Vertical) {
        logString << "Evaluation with rays inside central plane" << std::endl;
        logString << std::endl;
    }

    #ifdef QT_EXISTS
    if(logBox != NULL)
        logBox->appendPlainText(logString.str().c_str());
    #else
    std::cout << logString.str();
    #endif

    gener_out << "----------------------------------------" << std::endl;

}
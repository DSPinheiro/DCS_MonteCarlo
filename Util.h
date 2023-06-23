#pragma once

/*
 * Util.h
 *
 *  Created on: Feb 28, 2022
 *      Author: danip
 */

#ifndef UTIL_H_
#define UTIL_H_


#include <vector>
#include <string>


namespace Util {

    double GaussianBox(double sta_dev, double mean);
    void cov_srt(std::vector<std::vector<double>> covar, int npc, int ma, int* ia, int mfit);
    void cw(double xw, double yw, double& ak, double& al);
    int FindLoc(std::vector<double> array, double value);
    double getFirstApproxAngle(double tetaref, double tetadir, double sin_fi, double cos_fi, double tilt_C1, double squa_tilt1);
    double getFullApproximationAngle(double tetaref, double tetadir, double cos_e, double tan_e, double fidir, double tilt_C1);
    std::vector<double> getFullAngle(double r1x, double r1y, double r1z, double n1x, double n1y, double n1z);
    void gauss_j(std::vector<std::vector<double>> a, int n, int np, std::vector<std::vector<double>> b, int m, int mp);
    void mrq_cof(std::vector<double> x, std::vector<double> y, std::vector<double> sig, int ndata, double* a, int* ia, int ma, std::vector<std::vector<double>> alpha, double* beta, int nalp, double& chisq, void(funcs(double&, double*, double&, double*, int)));
    void mrq_min(std::vector<double> x, std::vector<double> y, std::vector<double> sig, int ndata, double* a, int* ia, int ma, std::vector<std::vector<double>> covar, std::vector<std::vector<double>> alpha, int nca, double& chisq, void(funcs(double&, double*, double&, double*, int)), double& alamda);
    double Latice_temp(double d_lat, double T_crystal);
    std::vector<double> getYZ(double r_temp, double sin_tetap_temp, double cos_tetap_temp, double tan_tetadir_temp, double tan_fidir_temp, double L_temp);
    void Pseud(double& x, double* a, double& y, double* dyda, int na);
    bool Reached(double z, double y, double tetadir_temp, double fidir_temp, double L_temp, double z_max, double z_min, double y_max, double y_min);
    double getFirstApproxAngle2(double tetaref, double tetadir, double delrot, double sin_fi, double cos_fi, double squa_tilt2, double cosdel, double cosdel_othe, double cosdel_teta, double cosdel_teta_othe, double sin_teref_tedi, bool Parallel);
    double getFullApproximationAngle2(double tetaref, double tetadir, double delrot, double cos_e, double tan_e, double cos2_e, double fidir, double tilt_C1, double tilt_C2, bool Parallel);
    std::vector<double> getFullAngle2(double r2x, double r2y, double r2z, double n2x, double n2y, double n2z);
    std::vector<double> spline(std::vector<double> x, std::vector<double> y, double yp1, double ypn);
    double splint_te(std::vector<double> xa, std::vector<double> ya, std::vector<double> y2a, double x);
    void Voig(double& x, double* a, double& y, double* dyda, int na);

    void analyse();
    void geo_corre();
    bool CheckSpectrum(std::string unit);
    double ObtainVert(int crystal, double angle);
    bool getReflection(double angle, double tetabra, double lamda, bool type_crystal, bool poli_p);
    double getNewTemp(int bin_tem, int& bin_fas, double& pha_tem);
    double getEnergy(double a_lamds_uni, double db_lamds_uni, double tw_d);
    std::vector<double> getLims(double tetaref, double delrot_min, double delrot_max, double fi_max, double teta_max, double teta_min);
    void initPlates();
    void Make(int crystal, double y, double z);
    void fit(bool Parallel);
    void Read_CurveResponce();
    std::vector<double> horCorr(double y_pro_C1, double y_max_C1, double z_pro_C1, double z_max_C1, bool type_c);
    void FitData(int numbins, double angle_para, int toint_para, double angle_anti, int toint_anti);
    void Read_EnergySpectrum(const std::string& filename);
    std::vector<double> misalign(double Dis_total);
    void Set_angs();
    void test_In();
}

#endif /* UTIL_H_ */
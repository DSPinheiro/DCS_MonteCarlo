#include <curand.h>
#include <curand_kernel.h>
#include <cuda/std/chrono>

#include <stdio.h> //DEBUG PURPOSES

#include "../include_cuda/Util_cuda.cuh"

using namespace Util_CUDA;


__device__ void Util_CUDA::setupRand(curandState *state, unsigned int seed, int block, int thread)
{
    curand_init(seed, block, thread, state);
}


__device__ double Util_CUDA::GaussianBox(curandState *state, double sta_dev, double mean, bool box_muller)
{
    double v1 = sta_dev * curand_normal_double(state) + mean;
    double v2 = sta_dev * curand_normal_double(state) + mean;

    return sqrt(v1 * v1 + v2 * v2);
}

__device__ double2 Util_CUDA::getYZ(double r_temp, double sin_tetap_temp, double cos_tetap_temp, double tan_tetadir_temp, double tan_fidir_temp, double L_temp)
{
    double2 res;

    res.x = (r_temp * cos_tetap_temp + tan_tetadir_temp * L_temp);
    res.y = (r_temp * sin_tetap_temp + tan_fidir_temp * L_temp);

    return res;
}


__device__ void Util_CUDA::Make(int crystal, double y, double z, MakeParameters *pars)
{

    double max_plot_x_temp, max_plot_y_temp;
    int nx, ny;

    max_plot_x_temp = pars->max_plot_x[crystal - 1];
    max_plot_y_temp = pars->max_plot_y[crystal - 1];

    nx = (int)(nm2 * y / max_plot_x_temp + np2);
    ny = (int)(nm2 * z / max_plot_y_temp + np2);


    if (crystal == 1) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            atomicAdd(&pars->hist_image_plate_source[i], 1);
            atomicAdd(&pars->counts_sour, 1);
            
            if (pars->max_hist[crystal - 1] < pars->hist_image_plate_source[i])
                atomicExch(&pars->max_hist[crystal - 1], pars->hist_image_plate_source[i]);
        }
    }
    else if (crystal == 2) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            atomicAdd(&pars->hist_image_plate_crystal1[i], 1);
            atomicAdd(&pars->counts_C1, 1);

            if (pars->max_hist[crystal - 1] < pars->hist_image_plate_crystal1[i])
                atomicExch(&pars->max_hist[crystal - 1], pars->hist_image_plate_crystal1[i]);
        }
    }
    else if (crystal == 3) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            atomicAdd(&pars->hist_image_plate_crystal2_para[i], 1);
            atomicAdd(&pars->counts_C2_para, 1);

            if (pars->max_hist[crystal - 1] < pars->hist_image_plate_crystal2_para[i])
                atomicExch(&pars->max_hist[crystal - 1], pars->hist_image_plate_crystal2_para[i]);
        }
    }
    else if (crystal == 4) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            atomicAdd(&pars->hist_image_plate_detc_para[i], 1);
            atomicAdd(&pars->counts_detc_para, 1);

            if (pars->max_hist[crystal - 1] < pars->hist_image_plate_detc_para[i])
                atomicExch(&pars->max_hist[crystal - 1], pars->hist_image_plate_detc_para[i]);
        }
    }
    else if (crystal == 5) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            atomicAdd(&pars->hist_image_plate_crystal2_anti[i], 1);
            atomicAdd(&pars->counts_C2_anti, 1);

            if (pars->max_hist[crystal - 1] < pars->hist_image_plate_crystal2_anti[i])
                atomicExch(&pars->max_hist[crystal - 1], pars->hist_image_plate_crystal2_anti[i]);
        }
    }
    else if (crystal == 6) {

        if (!(nx > n_his_ima || ny > n_his_ima || nx <= 0 || ny <= 0)) {
            int i = (nx - 1) * n_his_ima + (ny - 1);
            atomicAdd(&pars->hist_image_plate_detc_anti[i], 1);
            atomicAdd(&pars->counts_detc_anti, 1);

            if (pars->max_hist[crystal - 1] < pars->hist_image_plate_detc_anti[i])
                atomicExch(&pars->max_hist[crystal - 1], pars->hist_image_plate_detc_anti[i]);
        }
    }
}


__device__ double2 Util_CUDA::horCorr(double y_pro_C1, double y_max_C1, double z_pro_C1, double z_max_C1, bool type_c, double R_cur_crys_1, double R_cur_crys_2)
{

    double R_cur_crys_t;

    if (type_c)
        R_cur_crys_t = R_cur_crys_1;
    else
        R_cur_crys_t = R_cur_crys_2;

    double2 res;

    res.x = (cos(2 * R_cur_crys_t * y_pro_C1 * (1 - pow((z_pro_C1 / z_max_C1), 2)) / pow(y_max_C1, 2)));
    res.y = (sin(2 * R_cur_crys_t * y_pro_C1 * (1 - pow((z_pro_C1 / z_max_C1), 2)) / pow(y_max_C1, 2)));

    return res;
}


__device__ double4 Util_CUDA::getFullAngle(double r1x, double r1y, double r1z, double n1x, double n1y, double n1z)
{
    double inter_pro, angle, r2x, r2y, r2z;

    inter_pro = r1x * n1x + r1y * n1y + r1z * n1z;

    angle = asin(-inter_pro);
    r2x = r1x - 2 * inter_pro * n1x;
    r2y = r1y - 2 * inter_pro * n1y;
    r2z = r1z - 2 * inter_pro * n1z;

    double4 res;
    res.x = angle;
    res.y = r2x;
    res.z = r2y;
    res.w = r2z;

    return res;
}


__device__ double Util_CUDA::getEnergy(curandState *state, double a_lamds_uni, double db_lamds_uni, double tw_d, EnergyParameters pars)
{

    double p1, p2, natur_li, pm1, pm2, pm3, pm4, hit, rnd_inten, energy_t;
    int I_picks;


    if (pars.make_more_lines == 0)
        I_picks = 2;
    else if (pars.make_more_lines == 1) {
        if (pars.Do_background) {
            pm1 = 0.1875;
            pm2 = 0.34375;
            pm3 = 0.421875;
            pm4 = 0.5;
        }
        else {
            pm1 = pars.p1_ener;
            pm2 = pm1 + pars.p2_ener;
            pm3 = pm2 + pars.p3_ener;
            pm4 = 1;
        }


        p1 = curand_uniform_double(state);


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
        rnd_inten = curand_uniform_double(state);
        
        energy_t = Util_CUDA::splint_te(pars.Energy_spectrum_vectors->size, pars.Energy_spectrum_vectors->cum_ints, pars.Energy_spectrum_vectors->lamdas, pars.Energy_spectrum_vectors->lamda_two_derivs, rnd_inten);
        
        return Convert_Ag_minusone_eV / energy_t;
    }

    if (pars.make_more_lines == 0 || pars.make_more_lines == 1) {
        if (I_picks == 5) {
            p2 = curand_uniform_double(state);
            return a_lamds_uni + db_lamds_uni * p2;
        }
        else {
            hit = -1;

            while (hit < 0 || hit > tw_d) {
                p1 = curand_uniform_double(state) * CR_CUDART_PI;

                natur_li = pars.picks[I_picks - 1].natural_varia;

                hit = pars.picks[I_picks - 1].lamda + natur_li * tan(p1);
            }
        }

        hit = Util_CUDA::GaussianBox(state, pars.gauss_Doop, hit);
        return hit;

    }
}

//TODO: test if there is a faster alternative
__device__ double Util_CUDA::splint_te(int64_t size, double *xa, double *ya, double *y2a, double x)
{
    int k, klo;
    double a, b, h;

    klo = 1;
    size_t khi = size;
    
    while (khi - klo > 1) {
        k = (khi + klo) / 2;
        if (xa[k - 1] > x)
            khi = k;
        else
            klo = k;
    }

    h = xa[khi - 1] - xa[klo - 1];
    
    a = (xa[khi - 1] - x) / h;
    b = (x - xa[klo - 1]) / h;

    return a * ya[klo - 1] + b * ya[khi - 1] + ((pow(a, 3) - a) * y2a[klo - 1] + (pow(b, 3) - b) * y2a[khi - 1]) * pow(h, 2) / 6.0;
}


__device__ bool Util_CUDA::getReflection(curandState *state, double angle, double tetabra, double lamda, bool type_crystal, ReflectionParameters pars, bool poli_p)
{
    double p, dif, inte, inte1, inte2;

    dif = angle - tetabra;

    double energy = Convert_Ag_minusone_eV / lamda;

    int energy_resp_index;


    for (int i = 0; i < pars.size; i++)
    {
        double energ = pars.available_energies[i];

        if (energ > energy)
        {
            energy_resp_index = i - 1;
            break;
        }
    }

    double energy_min_angle_resp, energy_max_angle_resp;
    energy_min_angle_resp = max(pars.min_angle_resp[energy_resp_index], pars.min_angle_resp[energy_resp_index + 1]);
    energy_max_angle_resp = min(pars.max_angle_resp[energy_resp_index], pars.max_angle_resp[energy_resp_index + 1]);

    if (dif < energy_min_angle_resp) {
        return false;
    }
    else {
        if (dif < energy_max_angle_resp) {
            if (type_crystal && pars.mka_poli) {
                if (poli_p) {
                    inte1 = Util_CUDA::splint_te(pars.Crystal_Responces[energy_resp_index]->size,
                                                pars.Crystal_Responces[energy_resp_index]->degrees,
                                                pars.Crystal_Responces[energy_resp_index]->reflecti_total_ps,
                                                pars.Crystal_Responces[energy_resp_index]->reflecti_two_deriv_ps, dif);

                    inte2 = Util_CUDA::splint_te(pars.Crystal_Responces[energy_resp_index + 1]->size,
                                                pars.Crystal_Responces[energy_resp_index + 1]->degrees,
                                                pars.Crystal_Responces[energy_resp_index + 1]->reflecti_total_ps,
                                                pars.Crystal_Responces[energy_resp_index + 1]->reflecti_total_ps, dif);

                    inte = ((inte2 - inte1) / (pars.available_energies[energy_resp_index + 1] - pars.available_energies[energy_resp_index])) * (energy - pars.available_energies[energy_resp_index]) + inte1;
                }
                else {
                    inte1 = Util_CUDA::splint_te(pars.Crystal_Responces[energy_resp_index]->size,
                                                pars.Crystal_Responces[energy_resp_index]->degrees,
                                                pars.Crystal_Responces[energy_resp_index]->reflecti_total_ss,
                                                pars.Crystal_Responces[energy_resp_index]->reflecti_two_deriv_ss, dif);

                    inte2 = Util_CUDA::splint_te(pars.Crystal_Responces[energy_resp_index + 1]->size,
                                                pars.Crystal_Responces[energy_resp_index + 1]->degrees,
                                                pars.Crystal_Responces[energy_resp_index + 1]->reflecti_total_ss,
                                                pars.Crystal_Responces[energy_resp_index + 1]->reflecti_two_deriv_ss, dif);

                    inte = ((inte2 - inte1) / (pars.available_energies[energy_resp_index + 1] - pars.available_energies[energy_resp_index])) * (energy - pars.available_energies[energy_resp_index]) + inte1;
                }
            }
            else {
                inte1 = Util_CUDA::splint_te(pars.Crystal_Responces[energy_resp_index]->size,
                                            pars.Crystal_Responces[energy_resp_index]->degrees,
                                            pars.Crystal_Responces[energy_resp_index]->reflecti_totals,
                                            pars.Crystal_Responces[energy_resp_index]->reflecti_two_derivs, dif);

                inte2 = Util_CUDA::splint_te(pars.Crystal_Responces[energy_resp_index + 1]->size,
                                            pars.Crystal_Responces[energy_resp_index + 1]->degrees,
                                            pars.Crystal_Responces[energy_resp_index + 1]->reflecti_totals,
                                            pars.Crystal_Responces[energy_resp_index + 1]->reflecti_two_derivs, dif);

                inte = ((inte2 - inte1) / (pars.available_energies[energy_resp_index + 1] - pars.available_energies[energy_resp_index])) * (energy - pars.available_energies[energy_resp_index]) + inte1;
            }

            p = curand_uniform_double(state);

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


__device__ double4 Util_CUDA::getFullAngle2(double r2x, double r2y, double r2z, double n2x, double n2y, double n2z)
{

    double inter_pro, angle, r3x, r3y, r3z;

    inter_pro = r2x * n2x + r2y * n2y + r2z * n2z;

    angle = asin(-inter_pro);
    r3x = r2x - 2 * inter_pro * n2x;
    r3y = r2y - 2 * inter_pro * n2y;
    r3z = r2z - 2 * inter_pro * n2z;

    double4 res;
    res.x = angle;
    res.y = r3x;
    res.z = r3y;
    res.w = r3z;

    return res;

}

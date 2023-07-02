#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#include <cuda/std/chrono>
#include <vector>

#include "Util_cuda.cuh"

#include "parallel_bin.cuh"


void ParallelBin::makeBin(int totalBinEventCount, BinParameters *binIn, SetupParameters setup)
{
    int devId;
    cudaGetDevice(&devId);
    
    
    curandState *state;
    cudaMalloc(&state, sizeof(curandState));
    // distributions -> https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions
    Util_CUDA::setup_kernel<<<1,1>>>(state);
    
    BinParameters *binInCUDA;
    SetupParameters setupCUDA;

    cudaMalloc(&binInCUDA, sizeof(BinParameters));
    cudaMalloc((void **)(& setupCUDA), sizeof(SetupParameters));

    cudaMemcpy(binInCUDA, binIn, sizeof(BinParameters), cudaMemcpyHostToDevice);
    cudaMemcpy(&setupCUDA, &setup, sizeof(SetupParameters), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    makeBinKernel<<<32 * numSMs, blockSize>>>(totalBinEventCount, state, binInCUDA, setupCUDA);

    cudaMemcpy(binIn, binInCUDA, sizeof(BinParameters), cudaMemcpyDeviceToHost);
    
    // cudaDeviceSynchronize();
}


__global__ void makeBinKernel(int totalBinEventCount, curandState *randState, ParallelBin::BinParameters *binIn, ParallelBin::SetupParameters setup)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    double table_noise_width = (setup.table_resolution / 2.355) / 6.0;

    for (int i = idx; i < totalBinEventCount; i += stride)
    {
        //Temporary event to show in the 3D view
        //If we have less than maxEventNum we just append otherwise we see
        //this temporary event stored reaches the exit before appending.
        //Each triple of values is 1 point and each event will have 4 points.
        //Source - Crystal1 - Crystal2 - Detector
        std::vector<double> tmpEvent;

        if(setup.Make_Horizontal){
            binIn->p = setup.del_teta_L * curand_uniform_double(randState + idx + stride) + setup.teta_min_L;
            
            binIn->tetadir = binIn->p;
        }else
            binIn->tetadir = 0;


        binIn->sin_tetadir = sin(binIn->tetadir);
        binIn->cos_tetadir = cos(binIn->tetadir);
        binIn->tan_tetadir = binIn->sin_tetadir / binIn->cos_tetadir;

        if(setup.make_table_noise && setup.table_resolution >= 0.01)
        {
            binIn->teta_table_thread = setup.teta_table;
            binIn->teta_table_thread += table_noise_width * curand_normal_double(randState + idx + stride);
        }

        binIn->cos_tetadirCry1 = cos(binIn->tetadir + binIn->teta_table_thread * setup.convrad);


        if(setup.Make_Vertical){
            binIn->p = setup.del_fi_L * curand_uniform_double(randState + idx + stride) + setup.fi_min_L;
            binIn->fidir = binIn->p;
        }else
            binIn->fidir = setup.xsi * setup.convrad;


        binIn->sin_fidir = sin(binIn->fidir);
        binIn->cos_fidir = cos(binIn->fidir);
        binIn->tan_fidir = binIn->sin_fidir / binIn->cos_fidir;
        binIn->cos_fidirtilt = cos(binIn->fidir + setup.tilt_C1_temp);


        if(setup.type_source == "P"){
            binIn->z = setup.S_shi_ver_B;
            binIn->y = setup.S_shi_hor_B;
            binIn->r = sqrt(pow(binIn->z, 2) + pow(binIn->y, 2));

            if(setup.S_shi_ver_B == 0 && setup.S_shi_hor_B == 0){
                binIn->sin_tetap = 0;
                binIn->cos_tetap = 1;
            }else if(setup.S_shi_hor_B == 0){
                binIn->sin_tetap = 1;
                binIn->cos_tetap = 0;
            }else{
                binIn->var_temp = binIn->z / binIn->y;
                if(binIn->y > 0){
                    binIn->sin_tetap = binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                    binIn->cos_tetap = 1 / sqrt(1 + pow(binIn->var_temp, 2));
                }else{
                    binIn->sin_tetap = - binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                    binIn->cos_tetap = - 1 / sqrt(1 + pow(binIn->var_temp, 2));
                }
            }

        }else if(setup.type_source == "UC"){
            binIn->r = setup.S_sour_2 + 1;
            while(binIn->r > setup.S_sour_2){
                binIn->z = curand_uniform_double(randState + idx + stride) * setup.S_sour - setup.S_sour_2;
                binIn->y = curand_uniform_double(randState + idx + stride) * setup.S_sour - setup.S_sour_2;
                binIn->r = sqrt(pow(binIn->z, 2) + pow(binIn->y, 2));
            }


            binIn->z += setup.S_shi_ver_B;
            binIn->y += setup.S_shi_hor_B;

            if(binIn->y != 0)
                binIn->var_temp = binIn->z / binIn->y;
            else
                binIn->var_temp = 0;

            binIn->r = sqrt(pow(binIn->z, 2) + pow(binIn->y, 2));

            if(binIn->y > 0){
                binIn->sin_tetap = binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                binIn->cos_tetap = 1 / sqrt(1 + pow(binIn->var_temp, 2));
            }else{
                binIn->sin_tetap = - binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                binIn->cos_tetap = - 1 / sqrt(1 + pow(binIn->var_temp, 2));
            }
        }else if(setup.type_source == "G"){
            binIn->p = 2 * CR_CUDART_PI * curand_uniform_double(randState + idx + stride);
            binIn->tetap = binIn->p;
            binIn->sin_tetap = sin(binIn->tetap);
            binIn->cos_tetap = cos(binIn->tetap);
            binIn->r = setup.S_aper_R_2 + 1;

            while(binIn->r > setup.S_aper_R_2)
                binIn->r = Util_CUDA::GaussianBox(randState + idx + stride, setup.S_aper_var_2, 0);

            if(! (setup.S_shi_ver_B == 0 && setup.S_shi_hor_B == 0)){
                binIn->z = binIn->r * binIn->sin_tetap + setup.S_shi_ver_B;
                binIn->y = binIn->r * binIn->cos_tetap + setup.S_shi_hor_B;
                binIn->var_temp = binIn->z / binIn->y;
                binIn->r = sqrt(pow(binIn->z, 2) + pow(binIn->y, 2));

                if(binIn->y > 0){
                    binIn->sin_tetap = binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                    binIn->cos_tetap = 1 / sqrt(1 + pow(binIn->var_temp, 2));
                }else{
                    binIn->sin_tetap = - binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                    binIn->cos_tetap = - 1 / sqrt(1 + pow(binIn->var_temp, 2));
                }
            }

        }else if(setup.type_source == "UR"){
            binIn->z = curand_uniform_double(randState + idx + stride) * setup.z_sour - setup.z_sour_2;
            binIn->y = curand_uniform_double(randState + idx + stride) * setup.y_sour - setup.y_sour_2;

            binIn->z += setup.S_shi_ver_B;
            binIn->y += setup.S_shi_hor_B;
            binIn->var_temp = binIn->z / binIn->y;
            binIn->r = sqrt(pow(binIn->z, 2) + pow(binIn->y, 2));

            if(binIn->y > 0){
                binIn->sin_tetap = binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                binIn->cos_tetap = 1 / sqrt(1 + pow(binIn->var_temp, 2));
            }else{
                binIn->sin_tetap = - binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                binIn->cos_tetap = - 1 / sqrt(1 + pow(binIn->var_temp, 2));
            }
        }

        double2 yz = Util_CUDA::getYZ(binIn->r, binIn->sin_tetap, binIn->cos_tetap, binIn->tan_tetadir, binIn->tan_fidir, setup.LT_aper_Db);

        binIn->y = yz.x;
        binIn->z = yz.y;

        binIn->var_temp = pow(binIn->y - setup.S_shi_hor_A, 2) + pow(binIn->z - setup.S_shi_ver_A, 2);

        if(binIn->var_temp < setup.S_aper_sqr){

            if(setup.make_image_plates){
                Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                    binIn->max_plot_x,
                                    binIn->max_plot_y,
                                    binIn->hist_image_plate_source,
                                    binIn->hist_image_plate_crystal1,
                                    binIn->hist_image_plate_crystal2_para,
                                    binIn->hist_image_plate_crystal2_anti,
                                    binIn->hist_image_plate_detc_para,
                                    binIn->hist_image_plate_detc_anti,
                                    binIn->max_hist,
                                    binIn->counts_sour,
                                    binIn->counts_C1,
                                    binIn->counts_C2_para,
                                    binIn->counts_detc_para,
                                    binIn->counts_C2_anti,
                                    binIn->counts_detc_anti
                );

                Util_CUDA::Make(1, binIn->y, binIn->z, pars);

                //Event point at source
                tmpEvent.push_back(0); //X
                tmpEvent.push_back(binIn->y); //Y
                tmpEvent.push_back(binIn->z); //Z
            }


            binIn->r = sqrt(pow(binIn->y, 2) + pow(binIn->z, 2));

            double2 yz = Util_CUDA::getYZ(binIn->r, binIn->sin_tetap, binIn->cos_tetap, binIn->tan_tetadir, binIn->tan_fidir, setup.LT_aper_Db);

            binIn->y = yz.x;
            binIn->z = yz.y;


            binIn->y_pro_C1 = binIn->y * binIn->cos_tetadir / binIn->cos_tetadirCry1;
            binIn->z_pro_C1 = binIn->z * binIn->cos_fidir / binIn->cos_fidirtilt;

            if(binIn->y_pro_C1 < setup.y_max_C1 && binIn->y_pro_C1 > setup.y_min_C1 && binIn->z_pro_C1 < setup.z_max_C1 && binIn->z_pro_C1 > setup.z_min_C1){
                if(!setup.make_imageC1_After_refle){
                    if(setup.make_image_plates){
                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                    binIn->max_plot_x,
                                    binIn->max_plot_y,
                                    binIn->hist_image_plate_source,
                                    binIn->hist_image_plate_crystal1,
                                    binIn->hist_image_plate_crystal2_para,
                                    binIn->hist_image_plate_crystal2_anti,
                                    binIn->hist_image_plate_detc_para,
                                    binIn->hist_image_plate_detc_anti,
                                    binIn->max_hist,
                                    binIn->counts_sour,
                                    binIn->counts_C1,
                                    binIn->counts_C2_para,
                                    binIn->counts_detc_para,
                                    binIn->counts_C2_anti,
                                    binIn->counts_detc_anti
                        );
                        
                        Util_CUDA::Make(2, binIn->y_pro_C1, binIn->z_pro_C1, pars);
                        
                        //Event point at the first crystal
                        tmpEvent.push_back(0); //X
                        tmpEvent.push_back(binIn->y_pro_C1); //Y
                        tmpEvent.push_back(binIn->z_pro_C1); //Z
                    }
                }

                binIn->rx = binIn->cos_fidir * binIn->cos_tetadir;
                binIn->ry = binIn->cos_fidir * binIn->sin_tetadir;
                binIn->rz = binIn->sin_fidir;


                if(setup.Curve_crystall){
                    double2 corrRes = Util_CUDA::horCorr(
                        binIn->y_pro_C1,
                        setup.y_max_C1,
                        binIn->z_pro_C1,
                        setup.z_max_C1,
                        true,
                        setup.R_cur_crys_1,
                        setup.R_cur_crys_2);

                    binIn->Costeta_CHC = corrRes.x;
                    binIn->Sinteta_CHC = corrRes.y;

                    binIn->n1x = setup.n1x_temp * binIn->Costeta_CHC + setup.n1y_temp * binIn->Sinteta_CHC;
                    binIn->n1y = setup.n1y_temp * binIn->Costeta_CHC - setup.n1x_temp * binIn->Sinteta_CHC;
                }

                double4 angleRes = Util_CUDA::getFullAngle(binIn->rx, binIn->ry, binIn->rz, binIn->n1x, binIn->n1y, setup.n1z);
                binIn->angle = angleRes.x;
                binIn->r2x = angleRes.y;
                binIn->r2y = angleRes.z;
                binIn->r2z = angleRes.w;

                Util_CUDA::EnergyParameters pars = {
                    setup.make_more_lines,
                    setup.Do_background,
                    setup.p1_ener,
                    setup.p2_ener,
                    setup.p3_ener,
                    setup.Energy_spectrum_vectors,
                    setup.picks[5],
                    setup.gauss_Doop
                };

                binIn->lamda = Util_CUDA::getEnergy(randState + idx + stride, setup.a_lamds_uni, setup.b_lamds_uni, setup.tw_d1_para, pars);
                
                binIn->tetabra1 = asin(binIn->lamda / setup.tw_d1_para);

                bool poliP;

                if(setup.mka_poli)
                {
                    if(curand_uniform_double(randState + idx + stride) < setup.relationP_S)
                        poliP = true;
                    else
                        poliP = false;
                }
                else
                {
                    poliP = false;
                }

                Util_CUDA::ReflectionParameters parsR = {
                    setup.available_energies,
                    setup.min_angle_resp,
                    setup.max_angle_resp,
                    setup.mka_poli,
                    setup.Crystal_Responces
                };

                binIn->first_crystal_reach = Util_CUDA::getReflection(randState + idx + stride,
                        binIn->angle,
                        binIn->tetabra1,
                        binIn->lamda,
                        false,
                        parsR,
                        poliP);

                if(binIn->first_crystal_reach){
                    if(setup.make_imageC1_After_refle){
                        if(setup.make_image_plates){
                            Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                    binIn->max_plot_x,
                                    binIn->max_plot_y,
                                    binIn->hist_image_plate_source,
                                    binIn->hist_image_plate_crystal1,
                                    binIn->hist_image_plate_crystal2_para,
                                    binIn->hist_image_plate_crystal2_anti,
                                    binIn->hist_image_plate_detc_para,
                                    binIn->hist_image_plate_detc_anti,
                                    binIn->max_hist,
                                    binIn->counts_sour,
                                    binIn->counts_C1,
                                    binIn->counts_C2_para,
                                    binIn->counts_detc_para,
                                    binIn->counts_C2_anti,
                                    binIn->counts_detc_anti
                            );
                            
                            Util_CUDA::Make(2, binIn->y_pro_C1, binIn->z_pro_C1, pars);
                        }
                    }
                    
                    binIn->rx_rot = setup.cos_tetartab * binIn->r2x + setup.sin_tetartab * binIn->r2y;
                    binIn->ry_rot = -setup.sin_tetartab * binIn->r2x + setup.cos_tetartab * binIn->r2y;

                    binIn->var_temp = binIn->ry_rot / binIn->rx_rot;

                    binIn->tetadir = atan(binIn->var_temp);
                    binIn->tan_tetadir = binIn->var_temp;
                    binIn->cos_tetadir = cos(binIn->tetadir);


                    binIn->var_temp = binIn->r2z / sqrt(pow(binIn->rx_rot, 2) + pow(binIn->ry_rot, 2));
                    binIn->fidir = atan(binIn->var_temp);
                    binIn->tan_fidir = binIn->var_temp;
                    binIn->cos_fidir = cos(binIn->fidir);
                    binIn->cos_fidirtilt2_para = cos(binIn->fidir + setup.tilt_C2_para_temp);
                    binIn->cos_fidirtilt2_anti = cos(binIn->fidir + setup.tilt_C2_anti_temp);

                    binIn->corr_dis = binIn->y_pro_C1 * setup.cos_difteC1_Ta;

                    binIn->y_pro_C1 = -binIn->y_pro_C1 * setup.sin_difteC1_Ta;


                    if(binIn->y_pro_C1 == 0){
                        binIn->cos_tetap = 0;
                        if(binIn->z_pro_C1 > 0)
                            binIn->sin_tetap = 1;
                        else
                            binIn->sin_tetap = -1;
                    }else{
                        binIn->var_temp = binIn->z_pro_C1 / binIn->y_pro_C1;
                        binIn->r = sqrt(pow(binIn->z_pro_C1, 2) + pow(binIn->y_pro_C1, 2));

                        if(binIn->y_pro_C1 > 0){
                            binIn->sin_tetap = binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                            binIn->cos_tetap = 1 / sqrt(1 + pow(binIn->var_temp, 2));
                        }else{
                            binIn->sin_tetap = - binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                            binIn->cos_tetap = - 1 / sqrt(1 + pow(binIn->var_temp, 2));
                        }

                    }

                    double2 yz = Util_CUDA::getYZ(binIn->r, binIn->sin_tetap, binIn->cos_tetap, binIn->tan_tetadir, binIn->tan_fidir, setup.dist_Cr1_Cr2_Db - binIn->corr_dis);

                    binIn->y = yz.x;
                    binIn->z = yz.y;


                    if(setup.see_para){
                        binIn->sin_tetatab_del_dir = sin(setup.tetaref - binIn->delrot + binIn->tetadir);
                        binIn->y_pro_C1 = binIn->y * binIn->cos_tetadir / binIn->sin_tetatab_del_dir;
                        binIn->z_pro_C1 = binIn->z * binIn->cos_fidir / binIn->cos_fidirtilt2_para;
                        
                        if(binIn->y_pro_C1 < setup.y_max_C2 && binIn->y_pro_C1 > setup.y_min_C2 && binIn->z_pro_C1 < setup.z_max_C2 && binIn->z_pro_C1 > setup.z_min_C2){

                            if(setup.make_image_plates){
                                if(!setup.make_imageC2_After_refle){
                                    Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                binIn->max_plot_x,
                                                binIn->max_plot_y,
                                                binIn->hist_image_plate_source,
                                                binIn->hist_image_plate_crystal1,
                                                binIn->hist_image_plate_crystal2_para,
                                                binIn->hist_image_plate_crystal2_anti,
                                                binIn->hist_image_plate_detc_para,
                                                binIn->hist_image_plate_detc_anti,
                                                binIn->max_hist,
                                                binIn->counts_sour,
                                                binIn->counts_C1,
                                                binIn->counts_C2_para,
                                                binIn->counts_detc_para,
                                                binIn->counts_C2_anti,
                                                binIn->counts_detc_anti
                                    );
                                    
                                    Util_CUDA::Make(3, binIn->y_pro_C1, binIn->z_pro_C1, pars);

                                    //Event point at second crystal in parallel
                                    tmpEvent.push_back(0); //X
                                    tmpEvent.push_back(binIn->y_pro_C1); //Y
                                    tmpEvent.push_back(binIn->z_pro_C1); //Z
                                }
                            }

                            
                            binIn->tetabra2 = asin(binIn->lamda / setup.tw_d2_para);


                            if(setup.Curve_crystall){
                                double2 corrRes = Util_CUDA::horCorr(
                                    binIn->y_pro_C1,
                                    setup.y_max_C2,
                                    binIn->z_pro_C1,
                                    setup.z_max_C2,
                                    false,
                                    setup.R_cur_crys_1,
                                    setup.R_cur_crys_2);

                                binIn->Costeta_CHC = corrRes.x;
                                binIn->Sinteta_CHC = corrRes.y;

                                binIn->n2x_para = setup.n2x_para_temp * binIn->Costeta_CHC + setup.n2y_para_temp * binIn->Sinteta_CHC;
                                binIn->n2y_para = setup.n2y_para_temp * binIn->Costeta_CHC - setup.n2x_para_temp * binIn->Sinteta_CHC;
                            }


                            double4 angleRes = Util_CUDA::getFullAngle2(binIn->r2x, binIn->r2y, binIn->r2z, binIn->n2x_para, binIn->n2y_para, setup.n2z_para);
                            binIn->angle = angleRes.x;
                            binIn->r3x = angleRes.y;
                            binIn->r3y = angleRes.z;
                            binIn->r3z = angleRes.w;

                            
                            Util_CUDA::ReflectionParameters parsR = {
                                setup.available_energies,
                                setup.min_angle_resp,
                                setup.max_angle_resp,
                                setup.mka_poli,
                                setup.Crystal_Responces
                            };

                            binIn->sec_crystal_Parallel_reach = Util_CUDA::getReflection(randState + idx + stride,
                                    binIn->angle,
                                    binIn->tetabra2,
                                    binIn->lamda,
                                    false,
                                    parsR,
                                    poliP);
                            
                            if(binIn->sec_crystal_Parallel_reach){

                                if(setup.make_image_plates){
                                    if(setup.make_imageC2_After_refle){
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                binIn->max_plot_x,
                                                binIn->max_plot_y,
                                                binIn->hist_image_plate_source,
                                                binIn->hist_image_plate_crystal1,
                                                binIn->hist_image_plate_crystal2_para,
                                                binIn->hist_image_plate_crystal2_anti,
                                                binIn->hist_image_plate_detc_para,
                                                binIn->hist_image_plate_detc_anti,
                                                binIn->max_hist,
                                                binIn->counts_sour,
                                                binIn->counts_C1,
                                                binIn->counts_C2_para,
                                                binIn->counts_detc_para,
                                                binIn->counts_C2_anti,
                                                binIn->counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(3, binIn->y_pro_C1, binIn->z_pro_C1, pars);
                                    }
                                }


                                binIn->rx_rot_sec = setup.cos_tetartabdete_para * binIn->r3x + setup.sin_tetartabdete_para * binIn->r3y;
                                binIn->ry_rot_sec = -setup.sin_tetartabdete_para * binIn->r3x + setup.cos_tetartabdete_para * binIn->r3y;

                                binIn->var_temp = binIn->ry_rot_sec / binIn->rx_rot_sec;



                                binIn->tetadir_det = atan(binIn->var_temp);
                                binIn->tan_tetadir_det = binIn->var_temp;
                                binIn->cos_tetadir_det = cos(binIn->tetadir_det);


                                binIn->var_temp = binIn->rz / sqrt(pow(binIn->rx_rot, 2) + pow(binIn->ry_rot, 2));
                                binIn->fidir_det = atan(binIn->var_temp);
                                binIn->tan_fidir_det = binIn->var_temp;
                                binIn->cos_fidir_det = cos(binIn->fidir_det);
                                binIn->cos_fidirtilt2_para = cos(binIn->fidir + setup.tilt_C2_para_temp);

                                binIn->corr_dis_d_pa = -binIn->y_pro_C1 * setup.cos_difteC2_det_para;
                                binIn->y_pro_C1_d_pa = -binIn->y_pro_C1 * setup.sin_difteC2_det_para;


                                if(binIn->y_pro_C1_d_pa == 0){
                                    binIn->cos_tetap_det = 0;
                                    if(binIn->z_pro_C1 > 0)
                                        binIn->sin_tetap_det = 1;
                                    else
                                        binIn->sin_tetap_det = -1;
                                }else{
                                    binIn->var_temp = binIn->z_pro_C1 / binIn->y_pro_C1_d_pa;
                                    binIn->r_det = sqrt(pow(binIn->z_pro_C1, 2) + pow(binIn->y_pro_C1_d_pa, 2));

                                    if(binIn->y_pro_C1_d_pa > 0){
                                        binIn->sin_tetap_det = binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                                        binIn->cos_tetap_det = 1 / sqrt(1 + pow(binIn->var_temp, 2));
                                    }else{
                                        binIn->sin_tetap_det = - binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                                        binIn->cos_tetap_det = - 1 / sqrt(1 + pow(binIn->var_temp, 2));
                                    }
                                }

                                double2 yz = Util_CUDA::getYZ(binIn->r_det, binIn->sin_tetap_det, binIn->cos_tetap_det, binIn->tan_tetadir_det, binIn->tan_fidir_det, setup.dist_Cr2_det_Db - binIn->corr_dis_d_pa);

                                binIn->y_det = yz.x;
                                binIn->z_det = yz.y;

                                if(binIn->y_det < setup.ydetc_2_max && binIn->y_det > setup.ydetc_2_min && binIn->z_det < setup.zdetc_2_max && binIn->z_det > setup.zdetc_2_min){

                                    if(setup.make_image_plates){
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                binIn->max_plot_x,
                                                binIn->max_plot_y,
                                                binIn->hist_image_plate_source,
                                                binIn->hist_image_plate_crystal1,
                                                binIn->hist_image_plate_crystal2_para,
                                                binIn->hist_image_plate_crystal2_anti,
                                                binIn->hist_image_plate_detc_para,
                                                binIn->hist_image_plate_detc_anti,
                                                binIn->max_hist,
                                                binIn->counts_sour,
                                                binIn->counts_C1,
                                                binIn->counts_C2_para,
                                                binIn->counts_detc_para,
                                                binIn->counts_C2_anti,
                                                binIn->counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(4, binIn->y_det, binIn->z_det, pars);
                                        
                                        //Event point at detector in parallel
                                        tmpEvent.push_back(0); //X
                                        tmpEvent.push_back(binIn->y_det); //Y
                                        tmpEvent.push_back(binIn->z_det); //Z

                                        if (binIn->eventsToTrace_para.size() < setup.number_events) {
                                            binIn->eventsToTrace_para.push_back(tmpEvent);
                                        }
                                        else {
                                            binIn->eventsToTrace_para.erase(binIn->eventsToTrace_para.begin());
                                            binIn->eventsToTrace_para.push_back(tmpEvent);
                                        }
                                    }

                                    binIn->toint_para++;

                                    binIn->energy_sum_para_thread += setup.Convert_Ag_minusone_eV / binIn->lamda;
                                    // binIn->energy_sum_para[binIn->numbins - 1] += binIn->Convert_Ag_minusone_eV / binIn->lamda;
                                }
                                else {
                                    if (setup.make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        if (binIn->eventsToTrace_para.size() < setup.number_events && tmpEvent.size() >= 6) {
                                            binIn->eventsToTrace_para.push_back(tmpEvent);
                                        }
                                    }
                                }
                            }
                            else {
                                if (setup.make_image_plates) {
                                    //If the event does not reach the detector then only add when we have less than maxEventNum
                                    if (binIn->eventsToTrace_para.size() < setup.number_events && tmpEvent.size() >= 6) {
                                        binIn->eventsToTrace_para.push_back(tmpEvent);
                                    }
                                }
                            }
                        }
                        else {
                            if (setup.make_image_plates) {
                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                if (binIn->eventsToTrace_para.size() < setup.number_events && tmpEvent.size() >= 6) {
                                    binIn->eventsToTrace_para.push_back(tmpEvent);
                                }
                            }
                        }

                    }


                    if(setup.see_anti){


                        binIn->sin_tetatab_del_dir = sin(setup.tetaref + binIn->delrot - binIn->tetadir);

                        binIn->y_pro_C1 = binIn->y * binIn->cos_tetadir / binIn->sin_tetatab_del_dir;
                        binIn->z_pro_C1 = binIn->z * binIn->cos_fidir / binIn->cos_fidirtilt2_anti;

                        
                        if(binIn->y_pro_C1 < setup.y_max_C2 && binIn->y_pro_C1 > setup.y_min_C2 && binIn->z_pro_C1 < setup.z_max_C2 && binIn->z_pro_C1 > setup.z_min_C2){


                            if(setup.make_imageC2_After_refle){
                                if(setup.make_image_plates){
                                    Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                binIn->max_plot_x,
                                                binIn->max_plot_y,
                                                binIn->hist_image_plate_source,
                                                binIn->hist_image_plate_crystal1,
                                                binIn->hist_image_plate_crystal2_para,
                                                binIn->hist_image_plate_crystal2_anti,
                                                binIn->hist_image_plate_detc_para,
                                                binIn->hist_image_plate_detc_anti,
                                                binIn->max_hist,
                                                binIn->counts_sour,
                                                binIn->counts_C1,
                                                binIn->counts_C2_para,
                                                binIn->counts_detc_para,
                                                binIn->counts_C2_anti,
                                                binIn->counts_detc_anti
                                    );
                                    
                                    Util_CUDA::Make(5, binIn->y_pro_C1, binIn->z_pro_C1, pars);

                                    //Event point at second crystal in antiparallel
                                    tmpEvent.push_back(0); //X
                                    tmpEvent.push_back(binIn->y_pro_C1); //Y
                                    tmpEvent.push_back(binIn->z_pro_C1); //Z
                                }
                            }

                            
                            binIn->tetabra2 = asin(binIn->lamda / setup.tw_d2_anti);

                            if(setup.Curve_crystall){
                                double2 corrRes = Util_CUDA::horCorr(
                                    -binIn->y_pro_C1,
                                    setup.y_max_C2,
                                    binIn->z_pro_C1,
                                    setup.z_max_C2,
                                    false,
                                    setup.R_cur_crys_1,
                                    setup.R_cur_crys_2);

                                binIn->Costeta_CHC = corrRes.x;
                                binIn->Sinteta_CHC = corrRes.y;

                                binIn->n2x_anti = setup.n2x_anti_temp * binIn->Costeta_CHC + setup.n2y_anti_temp * binIn->Sinteta_CHC;
                                binIn->n2y_anti = setup.n2y_anti_temp * binIn->Costeta_CHC - setup.n2x_anti_temp * binIn->Sinteta_CHC;
                            }


                            double4 angleRes = Util_CUDA::getFullAngle2(binIn->r2x, binIn->r2y, binIn->r2z, binIn->n2x_anti, binIn->n2y_anti, setup.n2z_anti);
                            binIn->angle = angleRes.x;
                            binIn->r3x = angleRes.y;
                            binIn->r3y = angleRes.z;
                            binIn->r3z = angleRes.w;


                            Util_CUDA::ReflectionParameters parsR = {
                                setup.available_energies,
                                setup.min_angle_resp,
                                setup.max_angle_resp,
                                setup.mka_poli,
                                setup.Crystal_Responces
                            };

                            binIn->sec_crystal_Antiparallel_reach = Util_CUDA::getReflection(randState + idx + stride,
                                    binIn->angle,
                                    binIn->tetabra2,
                                    binIn->lamda,
                                    false,
                                    parsR,
                                    poliP);

                            if(binIn->sec_crystal_Antiparallel_reach){

                                if(setup.make_image_plates){
                                    if(setup.make_imageC2_After_refle){
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                binIn->max_plot_x,
                                                binIn->max_plot_y,
                                                binIn->hist_image_plate_source,
                                                binIn->hist_image_plate_crystal1,
                                                binIn->hist_image_plate_crystal2_para,
                                                binIn->hist_image_plate_crystal2_anti,
                                                binIn->hist_image_plate_detc_para,
                                                binIn->hist_image_plate_detc_anti,
                                                binIn->max_hist,
                                                binIn->counts_sour,
                                                binIn->counts_C1,
                                                binIn->counts_C2_para,
                                                binIn->counts_detc_para,
                                                binIn->counts_C2_anti,
                                                binIn->counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(5, binIn->y_pro_C1, binIn->z_pro_C1, pars);
                                    }
                                }


                                binIn->rx_rot_sec = setup.cos_tetartabdete_anti * binIn->r3x + setup.sin_tetartabdete_anti * binIn->r3y;
                                binIn->ry_rot_sec = - setup.sin_tetartabdete_anti * binIn->r3x + setup.cos_tetartabdete_anti * binIn->r3y;


                                binIn->var_temp = - binIn->ry_rot_sec / binIn->rx_rot_sec;
                                binIn->tetadir_det = atan(binIn->var_temp);
                                binIn->tan_tetadir_det = binIn->var_temp;
                                binIn->cos_tetadir_det = cos(binIn->tetadir_det);


                                binIn->var_temp = binIn->rz / sqrt(pow(binIn->rx_rot, 2) + pow(binIn->ry_rot, 2));
                                binIn->fidir_det = atan(binIn->var_temp);
                                binIn->tan_fidir_det = binIn->var_temp;
                                binIn->cos_fidir_det = cos(binIn->fidir_det);
                                binIn->cos_fidirtilt2_anti = cos(binIn->fidir + setup.tilt_C2_anti_temp);

                                binIn->corr_dis_d_pa = binIn->y_pro_C1 * setup.cos_difteC2_det_para;
                                binIn->y_pro_C1_d_pa = binIn->y_pro_C1 * setup.sin_difteC2_det_para;


                                if(binIn->y_pro_C1_d_pa == 0){
                                    binIn->cos_tetap_det = 0;
                                    if(binIn->z_pro_C1 > 0)
                                        binIn->sin_tetap_det = 1;
                                    else
                                        binIn->sin_tetap_det = -1;
                                }else{
                                    binIn->var_temp = binIn->z_pro_C1 / binIn->y_pro_C1_d_pa;
                                    binIn->r_det = sqrt(pow(binIn->z_pro_C1, 2) + pow(binIn->y_pro_C1_d_pa, 2));

                                    if(binIn->y_pro_C1_d_pa > 0){
                                        binIn->sin_tetap_det = binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                                        binIn->cos_tetap_det = 1 / sqrt(1 + pow(binIn->var_temp, 2));
                                    }else{
                                        binIn->sin_tetap_det = - binIn->var_temp / sqrt(1 + pow(binIn->var_temp, 2));
                                        binIn->cos_tetap_det = - 1 / sqrt(1 + pow(binIn->var_temp, 2));
                                    }
                                }


                                double2 yz = Util_CUDA::getYZ(binIn->r_det, binIn->sin_tetap_det, binIn->cos_tetap_det, binIn->tan_tetadir_det, binIn->tan_fidir_det, setup.dist_Cr2_det_Db - binIn->corr_dis_d_pa);

                                binIn->y_det = yz.x;
                                binIn->z_det = yz.y;

                                if(binIn->y_det < setup.ydetc_2_max && binIn->y_det > setup.ydetc_2_min && binIn->z_det < setup.zdetc_2_max && binIn->z_det > setup.zdetc_2_min){

                                    if (setup.make_image_plates) {
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                binIn->max_plot_x,
                                                binIn->max_plot_y,
                                                binIn->hist_image_plate_source,
                                                binIn->hist_image_plate_crystal1,
                                                binIn->hist_image_plate_crystal2_para,
                                                binIn->hist_image_plate_crystal2_anti,
                                                binIn->hist_image_plate_detc_para,
                                                binIn->hist_image_plate_detc_anti,
                                                binIn->max_hist,
                                                binIn->counts_sour,
                                                binIn->counts_C1,
                                                binIn->counts_C2_para,
                                                binIn->counts_detc_para,
                                                binIn->counts_C2_anti,
                                                binIn->counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(6, binIn->y_det, binIn->z_det, pars);

                                        //Event point at detector in antiparallel
                                        tmpEvent.push_back(0); //X
                                        tmpEvent.push_back(binIn->y_det); //Y
                                        tmpEvent.push_back(binIn->z_det); //Z

                                        if (binIn->eventsToTrace_anti.size() < setup.number_events) {
                                            binIn->eventsToTrace_anti.push_back(tmpEvent);
                                        }
                                        else {
                                            binIn->eventsToTrace_anti.erase(binIn->eventsToTrace_anti.begin());
                                            binIn->eventsToTrace_anti.push_back(tmpEvent);
                                        }
                                    }

                                    binIn->toint_anti++;

                                    binIn->energy_sum_anti_thread += setup.Convert_Ag_minusone_eV / binIn->lamda;
                                    // binIn->energy_sum_anti[binIn->numbins - 1] += binIn->Convert_Ag_minusone_eV / binIn->lamda;
                                }
                                else {
                                    if (setup.make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        if (binIn->eventsToTrace_anti.size() < setup.number_events && tmpEvent.size() >= 6) {
                                            binIn->eventsToTrace_anti.push_back(tmpEvent);
                                        }
                                    }
                                }

                            }
                            else {
                                if (setup.make_image_plates) {
                                    //If the event does not reach the detector then only add when we have less than maxEventNum
                                    if (binIn->eventsToTrace_anti.size() < setup.number_events && tmpEvent.size() >= 6) {
                                        binIn->eventsToTrace_anti.push_back(tmpEvent);
                                    }
                                }
                            }

                        }
                        else {
                            if (setup.make_image_plates) {
                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                if (binIn->eventsToTrace_anti.size() < setup.number_events && tmpEvent.size() >= 6) {
                                    binIn->eventsToTrace_anti.push_back(tmpEvent);
                                }
                            }
                        }

                    }

                }

            }

        }
    }
}
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#include <vector>

#include <cub/cub.cuh>

#include "Util_cuda.cuh"

#include "parallel_bin.cuh"


void ParallelBin::makeBin(int totalBinEventCount, const int Events3D, BinParameters *bin, SetupParameters *setup, ReductionVars *reduce)
{
    //Setup device specific values
    
    //Current CUDA device ID
    int devId;
    cudaGetDevice(&devId);

    //Size of the thread blocks
    int blockSize = 256;
    //Number of device multiprocessing units available
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
    //Effective size of the input reduction variables, depending on the number of events to simulate
    const int reduceSize = std::min({warpSize * numSMs * blockSize, totalBinEventCount});

    
    //State for the random generators 
    curandState *state;
    cudaMalloc(&state, sizeof(curandState));
    // distributions -> https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions
    //Setup the initial state based on the system clock
    
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();

    unsigned int seed = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

    Util_CUDA::setupRand<<<1,1>>>(state, seed);
    
    
    //Initialize device side pointers for the simulation variables

    //Bin variables
    BinParameters *binCUDA;
    //Setup variables
    SetupParameters *setupCUDA;
    //Summation reduction variables
    ReductionVars *reduceCUDA;


    //Allocate structures in the device
    cudaMalloc(&binCUDA, sizeof(BinParameters));
    cudaMalloc(&setupCUDA, sizeof(SetupParameters));
    cudaMalloc(&reduceCUDA, reduceSize * sizeof(ReductionVars));

    //Copy the initial values from the host
    cudaMemcpy(binCUDA, bin, sizeof(BinParameters), cudaMemcpyHostToDevice);
    cudaMemcpy(setupCUDA, setup, sizeof(SetupParameters), cudaMemcpyHostToDevice);
    
    for(int i = 0; i < reduceSize; i++)
    {
        cudaMemcpy(&reduceCUDA[i], reduce, sizeof(ReductionVars), cudaMemcpyHostToDevice);
    }

    
    //Execute the bin simulation
    makeBinKernel<<<warpSize * numSMs, blockSize>>>(totalBinEventCount, state, binCUDA, setupCUDA, reduceCUDA);

    //Synchronize the device to perform reduction
    cudaDeviceSynchronize();
    

    // Perform the reduction using CUB
    
    //Pointer where to store the reduce sum
    ReductionVars *d_sum;
    //allocate it
    cudaMalloc(&d_sum, sizeof(ReductionVars));
    
    
    // Determine temporary device storage requirements
    
    //Pointer to the temporary reduction storage
    void *d_temp_storage = NULL;
    //Size of the temporary reduction storage
    size_t temp_storage_bytes = 0;

    //Initialize the size of the temporary storage required
    // cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, reduceCUDA, d_sum, reduceSize);
    //Allocate the temporary storage
    // cudaMalloc(&d_temp_storage, temp_storage_bytes);
    //Perform the sum reduction
    // cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, reduceCUDA, d_sum, reduceSize);

    
    cudaMemcpy(bin, binCUDA, sizeof(BinParameters), cudaMemcpyDeviceToHost);
    cudaMemcpy(reduce, d_sum, sizeof(ReductionVars), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(binCUDA);
    cudaFree(reduceCUDA);
    cudaFree(setupCUDA);
    
    cudaFree(d_temp_storage);
    cudaFree(d_sum);
}


__global__ static void makeBinKernel(int totalBinEventCount, curandState *randState, 
                            ParallelBin::BinParameters *bin, ParallelBin::SetupParameters *setup, ParallelBin::ReductionVars *reduce)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int stride = blockDim.x * gridDim.x;


    double teta_table_thread = bin->teta_table_thread; double p = bin->p; double tetadir = bin->tetadir; double sin_tetadir = bin->sin_tetadir;
    double cos_tetadir = bin->cos_tetadir; double tan_tetadir = bin->tan_tetadir; double cos_tetadirCry1 = bin->cos_tetadirCry1;
    double fidir = bin->fidir; double sin_fidir = bin->sin_fidir; double cos_fidir = bin->cos_fidir; double tan_fidir = bin->tan_fidir;
    double cos_fidirtilt = bin->cos_fidirtilt; double z = bin->z; double y = bin->y; double r = bin->r; double tetap = bin->tetap;
    double sin_tetap = bin->sin_tetap; double cos_tetap = bin->cos_tetap; double var_temp = bin->var_temp; double y_pro_C1 = bin->y_pro_C1;
    double z_pro_C1 = bin->z_pro_C1; double rx = bin->rx; double ry = bin->ry; double rz = bin->rz; double Costeta_CHC = bin->Costeta_CHC;
    double Sinteta_CHC = bin->Sinteta_CHC; double n1x = bin->n1x; double n1y = bin->n1y; double angle = bin->angle; double r2x = bin->r2x; double r2y = bin->r2y;
    double r2z = bin->r2z; double lamda = bin->lamda; double tetabra1 = bin->tetabra1; bool first_crystal_reach = bin->first_crystal_reach; double rx_rot = bin->rx_rot;
    double ry_rot = bin->ry_rot; double cos_fidirtilt2_para = bin->cos_fidirtilt2_para; double cos_fidirtilt2_anti = bin->cos_fidirtilt2_anti; double corr_dis = bin->corr_dis;
    double sin_tetatab_del_dir = bin->sin_tetatab_del_dir; double tetabra2 = bin->tetabra2; double n2x_para = bin->n2x_para; double n2y_para = bin->n2y_para;
    double r3x = bin->r3x; double r3y = bin->r3y; double r3z = bin->r3z; bool sec_crystal_Parallel_reach = bin->sec_crystal_Parallel_reach; double rx_rot_sec = bin->rx_rot_sec;
    double ry_rot_sec = bin->ry_rot_sec; double tetadir_det = bin->tetadir_det; double tan_tetadir_det = bin->tan_tetadir_det; double fidir_det = bin->fidir_det;
    double tan_fidir_det = bin->tan_fidir_det; double corr_dis_d_pa = bin->corr_dis_d_pa; double y_pro_C1_d_pa = bin->y_pro_C1_d_pa; double cos_tetap_det = bin->cos_tetap_det;
    double sin_tetap_det = bin->sin_tetap_det; double r_det = bin->r_det; double y_det = bin->y_det; double z_det = bin->z_det; double n2x_anti = bin->n2x_anti;
    double n2y_anti = bin->n2y_anti; bool sec_crystal_Antiparallel_reach = bin->sec_crystal_Antiparallel_reach; int total_current_bins = bin->total_current_bins;
    double cos_tetadir_det = bin->cos_tetadir_det; double cos_fidir_det = bin->cos_fidir_det;

    double table_noise_width = (setup->table_resolution / 2.355) / 6.0;

    for (int i = idx; i < totalBinEventCount; i += stride)
    {
        //Temporary event to show in the 3D view
        //If we have less than maxEventNum we just append otherwise we see
        //this temporary event stored reaches the exit before appending.
        //Each triple of values is 1 point and each event will have 4 points.
        //Source - Crystal1 - Crystal2 - Detector
        double *tmpEvent_para = new double[12];
        double *tmpEvent_anti = new double[12];
        //current size of the temporary event (0, 3, 6, 9, 12)
        int tmpSize_para = 0;
        int tmpSize_anti = 0;

        if(setup->Make_Horizontal){
            p = setup->del_teta_L * curand_uniform_double(randState + idx + stride) + setup->teta_min_L;
            
            tetadir = p;
        }else
            tetadir = 0;


        sin_tetadir = sin(tetadir);
        cos_tetadir = cos(tetadir);
        tan_tetadir = sin_tetadir / cos_tetadir;

        if(setup->make_table_noise && setup->table_resolution >= 0.01)
        {
            teta_table_thread = setup->teta_table;
            teta_table_thread += table_noise_width * curand_normal_double(randState + idx + stride);
        }

        cos_tetadirCry1 = cos(tetadir + teta_table_thread * convrad);


        if(setup->Make_Vertical){
            p = setup->del_fi_L * curand_uniform_double(randState + idx + stride) + setup->fi_min_L;
            fidir = p;
        }else
            fidir = setup->xsi * convrad;


        sin_fidir = sin(fidir);
        cos_fidir = cos(fidir);
        tan_fidir = sin_fidir / cos_fidir;
        cos_fidirtilt = cos(fidir + setup->tilt_C1_temp);


        if(setup->type_source[0] == 'P'){
            z = setup->S_shi_ver_B;
            y = setup->S_shi_hor_B;
            r = sqrt(pow(z, 2) + pow(y, 2));

            if(setup->S_shi_ver_B == 0 && setup->S_shi_hor_B == 0){
                sin_tetap = 0;
                cos_tetap = 1;
            }else if(setup->S_shi_hor_B == 0){
                sin_tetap = 1;
                cos_tetap = 0;
            }else{
                var_temp = z / y;
                if(y > 0){
                    sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
                }else{
                    sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
                }
            }

        }else if(setup->type_source[0] == 'U' && setup->type_source[1] == 'C'){
            r = setup->S_sour_2 + 1;
            while(r > setup->S_sour_2){
                z = curand_uniform_double(randState + idx + stride) * setup->S_sour - setup->S_sour_2;
                y = curand_uniform_double(randState + idx + stride) * setup->S_sour - setup->S_sour_2;
                r = sqrt(pow(z, 2) + pow(y, 2));
            }


            z += setup->S_shi_ver_B;
            y += setup->S_shi_hor_B;

            if(y != 0)
                var_temp = z / y;
            else
                var_temp = 0;

            r = sqrt(pow(z, 2) + pow(y, 2));

            if(y > 0){
                sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
            }else{
                sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
            }
        }else if(setup->type_source[0] == 'G'){
            p = 2 * CR_CUDART_PI * curand_uniform_double(randState + idx + stride);
            tetap = p;
            sin_tetap = sin(tetap);
            cos_tetap = cos(tetap);
            r = setup->S_aper_R_2 + 1;

            while(r > setup->S_aper_R_2)
                r = Util_CUDA::GaussianBox(randState + idx + stride, setup->S_aper_var_2, 0);

            if(! (setup->S_shi_ver_B == 0 && setup->S_shi_hor_B == 0)){
                z = r * sin_tetap + setup->S_shi_ver_B;
                y = r * cos_tetap + setup->S_shi_hor_B;
                var_temp = z / y;
                r = sqrt(pow(z, 2) + pow(y, 2));

                if(y > 0){
                    sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
                }else{
                    sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                    cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
                }
            }

        }else if(setup->type_source[0] == 'U' && setup->type_source[0] == 'R'){
            z = curand_uniform_double(randState + idx + stride) * setup->z_sour - setup->z_sour_2;
            y = curand_uniform_double(randState + idx + stride) * setup->y_sour - setup->y_sour_2;

            z += setup->S_shi_ver_B;
            y += setup->S_shi_hor_B;
            var_temp = z / y;
            r = sqrt(pow(z, 2) + pow(y, 2));

            if(y > 0){
                sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
            }else{
                sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
            }
        }

        double2 yz = Util_CUDA::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, setup->LT_aper_Db);

        y = yz.x;
        z = yz.y;

        var_temp = pow(y - setup->S_shi_hor_A, 2) + pow(z - setup->S_shi_ver_A, 2);

        if(var_temp < setup->S_aper_sqr){

            if(setup->make_image_plates){
                Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                    reduce[idx].max_plot_x,
                                    reduce[idx].max_plot_y,
                                    reduce[idx].hist_image_plate_source,
                                    reduce[idx].hist_image_plate_crystal1,
                                    reduce[idx].hist_image_plate_crystal2_para,
                                    reduce[idx].hist_image_plate_crystal2_anti,
                                    reduce[idx].hist_image_plate_detc_para,
                                    reduce[idx].hist_image_plate_detc_anti,
                                    reduce[idx].max_hist,
                                    reduce[idx].counts_sour,
                                    reduce[idx].counts_C1,
                                    reduce[idx].counts_C2_para,
                                    reduce[idx].counts_detc_para,
                                    reduce[idx].counts_C2_anti,
                                    reduce[idx].counts_detc_anti
                );

                Util_CUDA::Make(1, y, z, pars);

                //Event point at source
                tmpEvent_para[0] = 0; //X
                tmpEvent_para[1] = y; //Y
                tmpEvent_para[2] = z; //Z
                tmpSize_para = 3;

                tmpEvent_anti[0] = 0; //X
                tmpEvent_anti[1] = y; //Y
                tmpEvent_anti[2] = z; //Z
                tmpSize_anti = 3;
            }


            r = sqrt(pow(y, 2) + pow(z, 2));

            double2 yz = Util_CUDA::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, setup->LT_aper_Db);

            y = yz.x;
            z = yz.y;


            y_pro_C1 = y * cos_tetadir / cos_tetadirCry1;
            z_pro_C1 = z * cos_fidir / cos_fidirtilt;

            if(y_pro_C1 < setup->y_max_C1 && y_pro_C1 > setup->y_min_C1 && z_pro_C1 < setup->z_max_C1 && z_pro_C1 > setup->z_min_C1){
                if(!setup->make_imageC1_After_refle){
                    if(setup->make_image_plates){
                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                    reduce[idx].max_plot_x,
                                    reduce[idx].max_plot_y,
                                    reduce[idx].hist_image_plate_source,
                                    reduce[idx].hist_image_plate_crystal1,
                                    reduce[idx].hist_image_plate_crystal2_para,
                                    reduce[idx].hist_image_plate_crystal2_anti,
                                    reduce[idx].hist_image_plate_detc_para,
                                    reduce[idx].hist_image_plate_detc_anti,
                                    reduce[idx].max_hist,
                                    reduce[idx].counts_sour,
                                    reduce[idx].counts_C1,
                                    reduce[idx].counts_C2_para,
                                    reduce[idx].counts_detc_para,
                                    reduce[idx].counts_C2_anti,
                                    reduce[idx].counts_detc_anti
                        );
                        
                        Util_CUDA::Make(2, y_pro_C1, z_pro_C1, pars);
                        
                        //Event point at the first crystal
                        tmpEvent_para[3] = 0; //X
                        tmpEvent_para[4] = y_pro_C1; //Y
                        tmpEvent_para[5] = z_pro_C1; //Z
                        tmpSize_para = 6;

                        tmpEvent_anti[3] = 0; //X
                        tmpEvent_anti[4] = y_pro_C1; //Y
                        tmpEvent_anti[5] = z_pro_C1; //Z
                        tmpSize_anti = 6;
                    }
                }

                rx = cos_fidir * cos_tetadir;
                ry = cos_fidir * sin_tetadir;
                rz = sin_fidir;


                if(setup->Curve_crystall){
                    double2 corrRes = Util_CUDA::horCorr(
                        y_pro_C1,
                        setup->y_max_C1,
                        z_pro_C1,
                        setup->z_max_C1,
                        true,
                        setup->R_cur_crys_1,
                        setup->R_cur_crys_2);

                    Costeta_CHC = corrRes.x;
                    Sinteta_CHC = corrRes.y;

                    n1x = setup->n1x_temp * Costeta_CHC + setup->n1y_temp * Sinteta_CHC;
                    n1y = setup->n1y_temp * Costeta_CHC - setup->n1x_temp * Sinteta_CHC;
                }

                double4 angleRes = Util_CUDA::getFullAngle(rx, ry, rz, n1x, n1y, setup->n1z);
                angle = angleRes.x;
                r2x = angleRes.y;
                r2y = angleRes.z;
                r2z = angleRes.w;

                Util_CUDA::EnergyParameters pars = {
                    setup->make_more_lines,
                    setup->Do_background,
                    setup->p1_ener,
                    setup->p2_ener,
                    setup->p3_ener,
                    setup->Energy_spectrum_vectors,
                    setup->picks[5],
                    setup->gauss_Doop
                };

                lamda = Util_CUDA::getEnergy(randState + idx + stride, setup->a_lamds_uni, setup->b_lamds_uni, setup->tw_d1_para, pars);
                
                tetabra1 = asin(lamda / setup->tw_d1_para);

                bool poliP;

                if(setup->mka_poli)
                {
                    if(curand_uniform_double(randState + idx + stride) < setup->relationP_S)
                        poliP = true;
                    else
                        poliP = false;
                }
                else
                {
                    poliP = false;
                }

                Util_CUDA::ReflectionParameters parsR = {
                    setup->available_energies,
                    setup->min_angle_resp,
                    setup->max_angle_resp,
                    setup->mka_poli,
                    setup->Crystal_Responces
                };

                first_crystal_reach = Util_CUDA::getReflection(randState + idx + stride,
                        angle,
                        tetabra1,
                        lamda,
                        false,
                        parsR,
                        poliP);

                if(first_crystal_reach){
                    if(setup->make_imageC1_After_refle){
                        if(setup->make_image_plates){
                            Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                    reduce[idx].max_plot_x,
                                    reduce[idx].max_plot_y,
                                    reduce[idx].hist_image_plate_source,
                                    reduce[idx].hist_image_plate_crystal1,
                                    reduce[idx].hist_image_plate_crystal2_para,
                                    reduce[idx].hist_image_plate_crystal2_anti,
                                    reduce[idx].hist_image_plate_detc_para,
                                    reduce[idx].hist_image_plate_detc_anti,
                                    reduce[idx].max_hist,
                                    reduce[idx].counts_sour,
                                    reduce[idx].counts_C1,
                                    reduce[idx].counts_C2_para,
                                    reduce[idx].counts_detc_para,
                                    reduce[idx].counts_C2_anti,
                                    reduce[idx].counts_detc_anti
                            );
                            
                            Util_CUDA::Make(2, y_pro_C1, z_pro_C1, pars);
                        }
                    }
                    
                    rx_rot = setup->cos_tetartab * r2x + setup->sin_tetartab * r2y;
                    ry_rot = -setup->sin_tetartab * r2x + setup->cos_tetartab * r2y;

                    var_temp = ry_rot / rx_rot;

                    tetadir = atan(var_temp);
                    tan_tetadir = var_temp;
                    cos_tetadir = cos(tetadir);


                    var_temp = r2z / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                    fidir = atan(var_temp);
                    tan_fidir = var_temp;
                    cos_fidir = cos(fidir);
                    cos_fidirtilt2_para = cos(fidir + setup->tilt_C2_para_temp);
                    cos_fidirtilt2_anti = cos(fidir + setup->tilt_C2_anti_temp);

                    corr_dis = y_pro_C1 * setup->cos_difteC1_Ta;

                    y_pro_C1 = -y_pro_C1 * setup->sin_difteC1_Ta;


                    if(y_pro_C1 == 0){
                        cos_tetap = 0;
                        if(z_pro_C1 > 0)
                            sin_tetap = 1;
                        else
                            sin_tetap = -1;
                    }else{
                        var_temp = z_pro_C1 / y_pro_C1;
                        r = sqrt(pow(z_pro_C1, 2) + pow(y_pro_C1, 2));

                        if(y_pro_C1 > 0){
                            sin_tetap = var_temp / sqrt(1 + pow(var_temp, 2));
                            cos_tetap = 1 / sqrt(1 + pow(var_temp, 2));
                        }else{
                            sin_tetap = - var_temp / sqrt(1 + pow(var_temp, 2));
                            cos_tetap = - 1 / sqrt(1 + pow(var_temp, 2));
                        }

                    }

                    double2 yz = Util_CUDA::getYZ(r, sin_tetap, cos_tetap, tan_tetadir, tan_fidir, setup->dist_Cr1_Cr2_Db - corr_dis);

                    y = yz.x;
                    z = yz.y;


                    if(setup->see_para){
                        sin_tetatab_del_dir = sin(setup->tetaref - setup->delrot + tetadir);
                        y_pro_C1 = y * cos_tetadir / sin_tetatab_del_dir;
                        z_pro_C1 = z * cos_fidir / cos_fidirtilt2_para;
                        
                        if(y_pro_C1 < setup->y_max_C2 && y_pro_C1 > setup->y_min_C2 && z_pro_C1 < setup->z_max_C2 && z_pro_C1 > setup->z_min_C2){

                            if(setup->make_image_plates){
                                if(!setup->make_imageC2_After_refle){
                                    Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                reduce[idx].max_plot_x,
                                                reduce[idx].max_plot_y,
                                                reduce[idx].hist_image_plate_source,
                                                reduce[idx].hist_image_plate_crystal1,
                                                reduce[idx].hist_image_plate_crystal2_para,
                                                reduce[idx].hist_image_plate_crystal2_anti,
                                                reduce[idx].hist_image_plate_detc_para,
                                                reduce[idx].hist_image_plate_detc_anti,
                                                reduce[idx].max_hist,
                                                reduce[idx].counts_sour,
                                                reduce[idx].counts_C1,
                                                reduce[idx].counts_C2_para,
                                                reduce[idx].counts_detc_para,
                                                reduce[idx].counts_C2_anti,
                                                reduce[idx].counts_detc_anti
                                    );
                                    
                                    Util_CUDA::Make(3, y_pro_C1, z_pro_C1, pars);

                                    //Event point at second crystal in parallel
                                    tmpEvent_para[6] = 0; //X
                                    tmpEvent_para[7] = y_pro_C1; //Y
                                    tmpEvent_para[8] = z_pro_C1; //Z
                                    tmpSize_para = 9;
                                }
                            }

                            
                            tetabra2 = asin(lamda / setup->tw_d2_para);


                            if(setup->Curve_crystall){
                                double2 corrRes = Util_CUDA::horCorr(
                                    y_pro_C1,
                                    setup->y_max_C2,
                                    z_pro_C1,
                                    setup->z_max_C2,
                                    false,
                                    setup->R_cur_crys_1,
                                    setup->R_cur_crys_2);

                                Costeta_CHC = corrRes.x;
                                Sinteta_CHC = corrRes.y;

                                n2x_para = setup->n2x_para_temp * Costeta_CHC + setup->n2y_para_temp * Sinteta_CHC;
                                n2y_para = setup->n2y_para_temp * Costeta_CHC - setup->n2x_para_temp * Sinteta_CHC;
                            }


                            double4 angleRes = Util_CUDA::getFullAngle2(r2x, r2y, r2z, n2x_para, n2y_para, setup->n2z_para);
                            angle = angleRes.x;
                            r3x = angleRes.y;
                            r3y = angleRes.z;
                            r3z = angleRes.w;

                            
                            Util_CUDA::ReflectionParameters parsR = {
                                setup->available_energies,
                                setup->min_angle_resp,
                                setup->max_angle_resp,
                                setup->mka_poli,
                                setup->Crystal_Responces
                            };

                            sec_crystal_Parallel_reach = Util_CUDA::getReflection(randState + idx + stride,
                                    angle,
                                    tetabra2,
                                    lamda,
                                    false,
                                    parsR,
                                    poliP);
                            
                            if(sec_crystal_Parallel_reach){

                                if(setup->make_image_plates){
                                    if(setup->make_imageC2_After_refle){
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                reduce[idx].max_plot_x,
                                                reduce[idx].max_plot_y,
                                                reduce[idx].hist_image_plate_source,
                                                reduce[idx].hist_image_plate_crystal1,
                                                reduce[idx].hist_image_plate_crystal2_para,
                                                reduce[idx].hist_image_plate_crystal2_anti,
                                                reduce[idx].hist_image_plate_detc_para,
                                                reduce[idx].hist_image_plate_detc_anti,
                                                reduce[idx].max_hist,
                                                reduce[idx].counts_sour,
                                                reduce[idx].counts_C1,
                                                reduce[idx].counts_C2_para,
                                                reduce[idx].counts_detc_para,
                                                reduce[idx].counts_C2_anti,
                                                reduce[idx].counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(3, y_pro_C1, z_pro_C1, pars);
                                    }
                                }


                                rx_rot_sec = setup->cos_tetartabdete_para * r3x + setup->sin_tetartabdete_para * r3y;
                                ry_rot_sec = -setup->sin_tetartabdete_para * r3x + setup->cos_tetartabdete_para * r3y;

                                var_temp = ry_rot_sec / rx_rot_sec;



                                tetadir_det = atan(var_temp);
                                tan_tetadir_det = var_temp;
                                cos_tetadir_det = cos(tetadir_det);


                                var_temp = rz / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                                fidir_det = atan(var_temp);
                                tan_fidir_det = var_temp;
                                cos_fidir_det = cos(fidir_det);
                                cos_fidirtilt2_para = cos(fidir + setup->tilt_C2_para_temp);

                                corr_dis_d_pa = -y_pro_C1 * setup->cos_difteC2_det_para;
                                y_pro_C1_d_pa = -y_pro_C1 * setup->sin_difteC2_det_para;


                                if(y_pro_C1_d_pa == 0){
                                    cos_tetap_det = 0;
                                    if(z_pro_C1 > 0)
                                        sin_tetap_det = 1;
                                    else
                                        sin_tetap_det = -1;
                                }else{
                                    var_temp = z_pro_C1 / y_pro_C1_d_pa;
                                    r_det = sqrt(pow(z_pro_C1, 2) + pow(y_pro_C1_d_pa, 2));

                                    if(y_pro_C1_d_pa > 0){
                                        sin_tetap_det = var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = 1 / sqrt(1 + pow(var_temp, 2));
                                    }else{
                                        sin_tetap_det = - var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = - 1 / sqrt(1 + pow(var_temp, 2));
                                    }
                                }

                                double2 yz = Util_CUDA::getYZ(r_det, sin_tetap_det, cos_tetap_det, tan_tetadir_det, tan_fidir_det, setup->dist_Cr2_det_Db - corr_dis_d_pa);

                                y_det = yz.x;
                                z_det = yz.y;

                                if(y_det < setup->ydetc_2_max && y_det > setup->ydetc_2_min && z_det < setup->zdetc_2_max && z_det > setup->zdetc_2_min){

                                    if(setup->make_image_plates){
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                reduce[idx].max_plot_x,
                                                reduce[idx].max_plot_y,
                                                reduce[idx].hist_image_plate_source,
                                                reduce[idx].hist_image_plate_crystal1,
                                                reduce[idx].hist_image_plate_crystal2_para,
                                                reduce[idx].hist_image_plate_crystal2_anti,
                                                reduce[idx].hist_image_plate_detc_para,
                                                reduce[idx].hist_image_plate_detc_anti,
                                                reduce[idx].max_hist,
                                                reduce[idx].counts_sour,
                                                reduce[idx].counts_C1,
                                                reduce[idx].counts_C2_para,
                                                reduce[idx].counts_detc_para,
                                                reduce[idx].counts_C2_anti,
                                                reduce[idx].counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(4, y_det, z_det, pars);
                                        
                                        //Event point at detector in parallel
                                        tmpEvent_para[9] = 0; //X
                                        tmpEvent_para[10] = y_det; //Y
                                        tmpEvent_para[11] = z_det; //Z
                                        tmpSize_para = 12;

                                        if (bin->curr3DEventCount_para <= setup->number_events) {
                                            reduce[idx].eventsToTrace_paraCUDA[bin->curr3DEventCount_para] = tmpEvent_para;
                                            atomicAdd(&(bin->curr3DEventCount_para), 1);
                                        }
                                        else {
                                            for (int i = 0; i < bin->curr3DEventCount_para; ++i)
                                                reduce[idx].eventsToTrace_paraCUDA[i] = reduce[idx].eventsToTrace_paraCUDA[i + 1];
                                            
                                            reduce[idx].eventsToTrace_paraCUDA[bin->curr3DEventCount_para] = tmpEvent_para;
                                        }
                                    }

                                    reduce[idx].toint_para++;

                                    reduce[idx].energy_sum_para_thread += Convert_Ag_minusone_eV / lamda;
                                }
                                else {
                                    if (setup->make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        if (bin->curr3DEventCount_para <= setup->number_events && tmpSize_para >= 6) {
                                            reduce[idx].eventsToTrace_paraCUDA[bin->curr3DEventCount_para] = tmpEvent_para;
                                            atomicAdd(&(bin->curr3DEventCount_para), 1);
                                        }
                                    }
                                }
                            }
                            else {
                                if (setup->make_image_plates) {
                                    //If the event does not reach the detector then only add when we have less than maxEventNum
                                    if (bin->curr3DEventCount_para < setup->number_events && tmpSize_para >= 6) {
                                        reduce[idx].eventsToTrace_paraCUDA[bin->curr3DEventCount_para] = tmpEvent_para;
                                        atomicAdd(&(bin->curr3DEventCount_para), 1);
                                    }
                                }
                            }
                        }
                        else {
                            if (setup->make_image_plates) {
                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                if (bin->curr3DEventCount_para < setup->number_events && tmpSize_para >= 6) {
                                    reduce[idx].eventsToTrace_paraCUDA[bin->curr3DEventCount_para] = tmpEvent_para;
                                    atomicAdd(&(bin->curr3DEventCount_para), 1);
                                }
                            }
                        }

                    }


                    if(setup->see_anti){


                        sin_tetatab_del_dir = sin(setup->tetaref + setup->delrot - tetadir);

                        y_pro_C1 = y * cos_tetadir / sin_tetatab_del_dir;
                        z_pro_C1 = z * cos_fidir / cos_fidirtilt2_anti;

                        
                        if(y_pro_C1 < setup->y_max_C2 && y_pro_C1 > setup->y_min_C2 && z_pro_C1 < setup->z_max_C2 && z_pro_C1 > setup->z_min_C2){


                            if(setup->make_imageC2_After_refle){
                                if(setup->make_image_plates){
                                    Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                reduce[idx].max_plot_x,
                                                reduce[idx].max_plot_y,
                                                reduce[idx].hist_image_plate_source,
                                                reduce[idx].hist_image_plate_crystal1,
                                                reduce[idx].hist_image_plate_crystal2_para,
                                                reduce[idx].hist_image_plate_crystal2_anti,
                                                reduce[idx].hist_image_plate_detc_para,
                                                reduce[idx].hist_image_plate_detc_anti,
                                                reduce[idx].max_hist,
                                                reduce[idx].counts_sour,
                                                reduce[idx].counts_C1,
                                                reduce[idx].counts_C2_para,
                                                reduce[idx].counts_detc_para,
                                                reduce[idx].counts_C2_anti,
                                                reduce[idx].counts_detc_anti
                                    );
                                    
                                    Util_CUDA::Make(5, y_pro_C1, z_pro_C1, pars);

                                    //Event point at second crystal in antiparallel
                                    tmpEvent_anti[6] = 0; //X
                                    tmpEvent_anti[7] = y_pro_C1; //Y
                                    tmpEvent_anti[8] = z_pro_C1; //Z
                                    tmpSize_anti = 9;
                                }
                            }

                            
                            tetabra2 = asin(lamda / setup->tw_d2_anti);

                            if(setup->Curve_crystall){
                                double2 corrRes = Util_CUDA::horCorr(
                                    -y_pro_C1,
                                    setup->y_max_C2,
                                    z_pro_C1,
                                    setup->z_max_C2,
                                    false,
                                    setup->R_cur_crys_1,
                                    setup->R_cur_crys_2);

                                Costeta_CHC = corrRes.x;
                                Sinteta_CHC = corrRes.y;

                                n2x_anti = setup->n2x_anti_temp * Costeta_CHC + setup->n2y_anti_temp * Sinteta_CHC;
                                n2y_anti = setup->n2y_anti_temp * Costeta_CHC - setup->n2x_anti_temp * Sinteta_CHC;
                            }


                            double4 angleRes = Util_CUDA::getFullAngle2(r2x, r2y, r2z, n2x_anti, n2y_anti, setup->n2z_anti);
                            angle = angleRes.x;
                            r3x = angleRes.y;
                            r3y = angleRes.z;
                            r3z = angleRes.w;


                            Util_CUDA::ReflectionParameters parsR = {
                                setup->available_energies,
                                setup->min_angle_resp,
                                setup->max_angle_resp,
                                setup->mka_poli,
                                setup->Crystal_Responces
                            };

                            sec_crystal_Antiparallel_reach = Util_CUDA::getReflection(randState + idx + stride,
                                    angle,
                                    tetabra2,
                                    lamda,
                                    false,
                                    parsR,
                                    poliP);

                            if(sec_crystal_Antiparallel_reach){

                                if(setup->make_image_plates){
                                    if(setup->make_imageC2_After_refle){
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                reduce[idx].max_plot_x,
                                                reduce[idx].max_plot_y,
                                                reduce[idx].hist_image_plate_source,
                                                reduce[idx].hist_image_plate_crystal1,
                                                reduce[idx].hist_image_plate_crystal2_para,
                                                reduce[idx].hist_image_plate_crystal2_anti,
                                                reduce[idx].hist_image_plate_detc_para,
                                                reduce[idx].hist_image_plate_detc_anti,
                                                reduce[idx].max_hist,
                                                reduce[idx].counts_sour,
                                                reduce[idx].counts_C1,
                                                reduce[idx].counts_C2_para,
                                                reduce[idx].counts_detc_para,
                                                reduce[idx].counts_C2_anti,
                                                reduce[idx].counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(5, y_pro_C1, z_pro_C1, pars);
                                    }
                                }


                                rx_rot_sec = setup->cos_tetartabdete_anti * r3x + setup->sin_tetartabdete_anti * r3y;
                                ry_rot_sec = - setup->sin_tetartabdete_anti * r3x + setup->cos_tetartabdete_anti * r3y;


                                var_temp = - ry_rot_sec / rx_rot_sec;
                                tetadir_det = atan(var_temp);
                                tan_tetadir_det = var_temp;
                                cos_tetadir_det = cos(tetadir_det);


                                var_temp = rz / sqrt(pow(rx_rot, 2) + pow(ry_rot, 2));
                                fidir_det = atan(var_temp);
                                tan_fidir_det = var_temp;
                                cos_fidir_det = cos(fidir_det);
                                cos_fidirtilt2_anti = cos(fidir + setup->tilt_C2_anti_temp);

                                corr_dis_d_pa = y_pro_C1 * setup->cos_difteC2_det_para;
                                y_pro_C1_d_pa = y_pro_C1 * setup->sin_difteC2_det_para;


                                if(y_pro_C1_d_pa == 0){
                                    cos_tetap_det = 0;
                                    if(z_pro_C1 > 0)
                                        sin_tetap_det = 1;
                                    else
                                        sin_tetap_det = -1;
                                }else{
                                    var_temp = z_pro_C1 / y_pro_C1_d_pa;
                                    r_det = sqrt(pow(z_pro_C1, 2) + pow(y_pro_C1_d_pa, 2));

                                    if(y_pro_C1_d_pa > 0){
                                        sin_tetap_det = var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = 1 / sqrt(1 + pow(var_temp, 2));
                                    }else{
                                        sin_tetap_det = - var_temp / sqrt(1 + pow(var_temp, 2));
                                        cos_tetap_det = - 1 / sqrt(1 + pow(var_temp, 2));
                                    }
                                }


                                double2 yz = Util_CUDA::getYZ(r_det, sin_tetap_det, cos_tetap_det, tan_tetadir_det, tan_fidir_det, setup->dist_Cr2_det_Db - corr_dis_d_pa);

                                y_det = yz.x;
                                z_det = yz.y;

                                if(y_det < setup->ydetc_2_max && y_det > setup->ydetc_2_min && z_det < setup->zdetc_2_max && z_det > setup->zdetc_2_min){

                                    if (setup->make_image_plates) {
                                        Util_CUDA::MakeParameters *pars = new Util_CUDA::MakeParameters(
                                                reduce[idx].max_plot_x,
                                                reduce[idx].max_plot_y,
                                                reduce[idx].hist_image_plate_source,
                                                reduce[idx].hist_image_plate_crystal1,
                                                reduce[idx].hist_image_plate_crystal2_para,
                                                reduce[idx].hist_image_plate_crystal2_anti,
                                                reduce[idx].hist_image_plate_detc_para,
                                                reduce[idx].hist_image_plate_detc_anti,
                                                reduce[idx].max_hist,
                                                reduce[idx].counts_sour,
                                                reduce[idx].counts_C1,
                                                reduce[idx].counts_C2_para,
                                                reduce[idx].counts_detc_para,
                                                reduce[idx].counts_C2_anti,
                                                reduce[idx].counts_detc_anti
                                        );
                                        
                                        Util_CUDA::Make(6, y_det, z_det, pars);

                                        //Event point at detector in antiparallel
                                        tmpEvent_anti[9] = 0; //X
                                        tmpEvent_anti[10] = y_det; //Y
                                        tmpEvent_anti[11] = z_det; //Z
                                        tmpSize_anti = 12;

                                        if (bin->curr3DEventCount_anti <= setup->number_events) {
                                            reduce[idx].eventsToTrace_antiCUDA[bin->curr3DEventCount_anti] = tmpEvent_anti;
                                            atomicAdd(&(bin->curr3DEventCount_anti), 1);
                                        }
                                        else {
                                            for (int i = 0; i < bin->curr3DEventCount_anti; ++i)
                                                reduce[idx].eventsToTrace_antiCUDA[i] = reduce[idx].eventsToTrace_antiCUDA[i + 1];
                                            
                                            reduce[idx].eventsToTrace_antiCUDA[bin->curr3DEventCount_anti] = tmpEvent_anti;
                                        }
                                    }

                                    reduce[idx].toint_anti++;

                                    reduce[idx].energy_sum_anti_thread += Convert_Ag_minusone_eV / lamda;
                                }
                                else {
                                    if (setup->make_image_plates) {
                                        //If the event does not reach the detector then only add when we have less than maxEventNum
                                        if (bin->curr3DEventCount_anti <= setup->number_events && tmpSize_anti >= 6) {
                                            reduce[idx].eventsToTrace_antiCUDA[bin->curr3DEventCount_anti] = tmpEvent_anti;
                                            atomicAdd(&(bin->curr3DEventCount_anti), 1);
                                        }
                                    }
                                }

                            }
                            else {
                                if (setup->make_image_plates) {
                                    //If the event does not reach the detector then only add when we have less than maxEventNum
                                    if (bin->curr3DEventCount_anti <= setup->number_events && tmpSize_anti >= 6) {
                                        reduce[idx].eventsToTrace_antiCUDA[bin->curr3DEventCount_anti] = tmpEvent_anti;
                                        atomicAdd(&(bin->curr3DEventCount_anti), 1);
                                    }
                                }
                            }

                        }
                        else {
                            if (setup->make_image_plates) {
                                //If the event does not reach the detector then only add when we have less than maxEventNum
                                if (bin->curr3DEventCount_anti < setup->number_events && tmpSize_anti >= 6) {
                                    reduce[idx].eventsToTrace_antiCUDA[bin->curr3DEventCount_anti] = tmpEvent_anti;
                                    atomicAdd(&(bin->curr3DEventCount_anti), 1);
                                }
                            }
                        }
                    }
                }
            }
        }
        total_current_bins++;
    }
}
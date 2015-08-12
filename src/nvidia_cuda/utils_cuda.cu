
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils_cuda.cuh"
#include <thrust/device_ptr.h>

#include <thrust/fill.h>


int
calc_quadrant_cuda(
    float theta_p)
{
    int quadrant;
    if ((theta_p >= 0 && theta_p < M_PI/2) ||
            (theta_p >= M_PI && theta_p < 3*M_PI/2))
    {
        quadrant = 1;
    }
    else
    {
        quadrant = 0;
    }
    return quadrant;
}


__host__ __device__
void pml_preprocess_cuda( const int ngridx, const int ngridy, int dz, int d, float mov,
		int quadrant, float sin_p, float cos_p, float *gridx, float *gridy,
		int *indi, float *dist, int *pcsize)
{

    float yi = (1-dz)/2.0+d+mov;

    float coordx [max_ngridy+1];
    float coordy [max_ngridx+1];
    float ax [max_ngridx+max_ngridy];
    float ay [max_ngridx+max_ngridy];
    float bx [max_ngridx+max_ngridy];
    float by [max_ngridx+max_ngridy];
    float coorx [max_ngridx+max_ngridy];
    float coory [max_ngridx+max_ngridy];

    int asize, bsize;
    int csize;

    calc_coords_cuda(
        ngridx, ngridy, /*xi*/0, yi, sin_p, cos_p, gridx, gridy,
        coordx, coordy);

    // Merge the (coordx, gridy) and (gridx, coordy)
    trim_coords_cuda(
        ngridx, ngridy, coordx, coordy, gridx, gridy,
        &asize, ax, ay, &bsize, bx, by);

    // Sort the array of intersection points (ax, ay) and
    // (bx, by). The new sorted intersection points are
    // stored in (coorx, coory). Total number of points
    // are csize.
    sort_intersections_cuda(
        quadrant, asize, ax, ay, bsize, bx, by,
        &csize, coorx, coory);

    // Calculate the distances (dist) between the
    // intersection points (coorx, coory). Find the
    // indices of the pixels on the reconstruction grid.
    calc_dist_cuda(
        ngridx, ngridy, csize, coorx, coory,
        indi, dist);

    *pcsize=csize;
}

void preprocessing_cuda(
    const int ry, const int rz,
    int num_pixels, float center,
    float *mov, float *dev_gridx, float *dev_gridy)
{
    int i;
	float *gridx =(float *)malloc((ry+1)*sizeof(float));
	float *gridy =(float *)malloc((rz+1)*sizeof(float));

    for(i=0; i<=ry; i++)
    {
        gridx[i] = -ry/2.+i;
    }

    for(i=0; i<=rz; i++)
    {
        gridy[i] = -rz/2.+i;
    }

	cudaError_t cudaStatus = cudaMemcpy(dev_gridx, gridx, (ry+1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	cudaStatus = cudaMemcpy(dev_gridy, gridy, (rz+1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	free(gridx);
	free(gridy);
/*
    *mov = (float)num_pixels/2.0-center;
    if(*mov-ceil(*mov) < 1e-2) {
        *mov += 1e-2;
    }
    */
    float fmov = (float)num_pixels/2.0-center;
    // forever true... let us kill the compare
//    if(fmov-ceil(fmov) < 1e-2) {
        fmov += 1e-2;
//    }
    *mov=fmov;
}



__host__ __device__
void calc_coords_cuda(
    int ry, int rz,
    float xi, float yi,
    float sin_p, float cos_p,
    float *gridx, float *gridy,
    float *coordx, float *coordy)
{
    float srcx, srcy;//, detx, dety;
    float slope;//, islope;
    int n;

    srcx = /*xi*cos_p*/-yi*sin_p;
    srcy = /*xi*sin_p*/+yi*cos_p;
//    detx = -xi*cos_p-yi*sin_p;
//    dety = -xi*sin_p+yi*cos_p;

//    slope = (srcy-dety)/(srcx-detx);
    slope = sin_p/cos_p;
//    islope = 1/slope;
    for (n=0; n<=ry; n++)
    {
        coordy[n] = slope*(gridx[n]-srcx)+srcy;
    }
    for (n=0; n<=rz; n++)
    {
        coordx[n] = (gridy[n]-srcy)/slope+srcx;
    }
}


__host__ __device__
void trim_coords_cuda(
    int ry, int rz,
    float *coordx, float *coordy,
    float *gridx, float* gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by)
{
    int n;

    *asize = 0;
    *bsize = 0;
    for (n=0; n<=rz; n++)
    {
        if (coordx[n] > gridx[0])
        {
            if (coordx[n] < gridx[ry])
            {
                ax[*asize] = coordx[n];
                ay[*asize] = gridy[n];
                (*asize)++;
            }
        }
    }
    for (n=0; n<=ry; n++)
    {
        if (coordy[n] > gridy[0])
        {
            if (coordy[n] < gridy[rz])
            {
                bx[*bsize] = gridx[n];
                by[*bsize] = coordy[n];
                (*bsize)++;
            }
        }
    }
}


__host__ __device__
void sort_intersections_cuda(
    int ind_condition,
    int asize, float *ax, float *ay,
    int bsize, float *bx, float *by,
    int *csize, float *coorx, float *coory)
{
    int i=0, j=0, k=0;
    int a_ind;
    while (i<asize && j<bsize)
    {
        a_ind = (ind_condition) ? i : (asize-1-i);
        if (ax[a_ind] < bx[j])
        {
            coorx[k] = ax[a_ind];
            coory[k] = ay[a_ind];
            i++;
            k++;
        }
        else
        {
            coorx[k] = bx[j];
            coory[k] = by[j];
            j++;
            k++;
        }
    }
    while (i < asize)
    {
        a_ind = (ind_condition) ? i : (asize-1-i);
        coorx[k] = ax[a_ind];
        coory[k] = ay[a_ind];
        i++;
        k++;
    }
    while (j < bsize)
    {
        coorx[k] = bx[j];
        coory[k] = by[j];
        j++;
        k++;
    }
    *csize = asize+bsize;
}


__host__ __device__
void calc_dist_cuda(
    int ry, int rz,
    int csize, float *coorx, float *coory,
    int *indi, float *dist)
{
//    int n, x1, x2, i1, i2;
    float diffx, diffy, midx, midy;
    float x1,x2;
    int indx, indy;

//    ANNOTATE_SITE_BEGIN( MySite1 );  // Place before the loop control statement to begin a parallel code region (parallel site).
    for (int n=0; n<csize-1; n++)
    {
//        ANNOTATE_ITERATION_TASK( MyTask1 );  // Place at the start of loop body. This annotation identifies an entire body as a task.
//        {
        diffx = coorx[n+1]-coorx[n];
        diffy = coory[n+1]-coory[n];
        dist[n] = sqrt(diffx*diffx+diffy*diffy);
        midx = (coorx[n+1]+coorx[n])/2;
        midy = (coory[n+1]+coory[n])/2;
        x1 = midx+ry/2.;
        x2 = midy+rz/2.;
        if (x1<0) x1--;
        if (x2<0) x2--;
//        i1 = (int)(midx+ry/2.);
//        i2 = (int)(midy+rz/2.);
        indx = x1;//i1-(i1>x1);
        indy = x2;//i2-(i2>x2);
        indi[n] = indy+(indx*rz);
//        }
    }
//    ANNOTATE_SITE_END();  // End the parallel code region, after task execution completes
}


__host__ __device__
void calc_simdata_cuda(
    int p, int s, int c,
    int ry, int rz,
    int num_slices, int num_pixels,
    int csize, int *indi, float *dist,
    float *model, float *simdata)
{
    int n;

    int index_model = s*ry*rz;
    int index_data = c+s*num_pixels+p*num_slices*num_pixels;
    for (n=0; n<csize-1; n++)
    {
        simdata[index_data] += model[indi[n]+index_model]*dist[n];
    }
}


// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
typedef struct
{
	int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
	int Cores;
} sSMtoCores;

//! Beginning of GPU Architecture definitions
int _ConvertSMVer2Cores(int major, int minor)
{
	
    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };
	
    int index = 0;
	
    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }
		
        index++;
    }
	
    // If we don't find the values, we default use the previous one to run properly
	//    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

int findCudaDevice(){
	int deviceId=-1;
	int deviceCount;
	int max_multiprocessors = 0;
	int i;
	struct cudaDeviceProp deviceProp;
	if(cudaGetDeviceCount(&deviceCount)!=cudaSuccess)
	{
		fprintf(stderr,"cudaGetDeviceCount() failed\n");
		return -1;
	}
	for (i=0;i<deviceCount;i++)
	{
		if (cudaGetDeviceProperties(&deviceProp,i)!=cudaSuccess)
		{
			fprintf(stderr,"cudaGetDeviceProperties(%d) failed\n",i);
			return -1;
		}
		
//		printf("INFO: Found Cuda Device %d, name is %s\n",i,deviceProp.name);

		//sm>=2.0 is required!
		if((deviceProp.major<2)||(deviceProp.computeMode==cudaComputeModeProhibited))
			continue;
		if (max_multiprocessors < _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount) {
			max_multiprocessors = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
			deviceId = i;
		}
		
	}
	return deviceId;
}


cudaError_t CCudaData::initCuda(){
	
	cudaError_t cudaStatus=cudaErrorInitializationError;
	float *devptr;	
	int deviceid=-1;

	
	deviceid=findCudaDevice();
    // Choose which GPU to run on, based on the fastest qulified GPU
    if (deviceid <0) {
        fprintf(stderr, "findCudaDevice failed!\nDo you have a CUDA-capable GPU with at least sm2.0 support and available?\n");
        goto Error;
    }
	
    cudaStatus = cudaSetDevice(deviceid);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        goto Error;
    }
	
//	printf("INFO: Cuda Device %d is actived!\n",deviceid);

	/*
	size_t heapSize;
	cudaDeviceGetLimit ( &heapSize,cudaLimitMallocHeapSize);
	printf("original heap Size :%ld\n",heapSize);
	cudaDeviceSetLimit ( cudaLimitMallocHeapSize,heapSize*20);
	cudaDeviceGetLimit ( &heapSize,cudaLimitMallocHeapSize);
	printf("required new heap Size :%ld\n",heapSize);
	 */
	//	cudaDeviceReset();
	//	exit(0);
	
	
    // Allocate GPU buffers 
//    sum_dist = (float *)calloc((ngridx*ngridy), sizeof(float));
    if (cudaMalloc((void**)&devptr, (host_ngridx*host_ngridy) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_sum_dist=devptr;
	
//    E = (float *)calloc((ngridx*ngridy), sizeof(float));
    if (cudaMalloc((void**)&devptr, (host_ngridx*host_ngridy) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_E=devptr;
	
//    F = (float *)calloc((ngridx*ngridy), sizeof(float));
    if (cudaMalloc((void**)&devptr, (host_ngridx*host_ngridy) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_F=devptr;
	
//    G = (float *)calloc((ngridx*ngridy), sizeof(float));
    if (cudaMalloc((void**)&devptr, (host_ngridx*host_ngridy) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_G=devptr;

	
    //gridx = (float *)malloc((host_ngridx+1)*sizeof(float));
    if (cudaMalloc((void**)&devptr, (host_ngridx+1) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_gridx=devptr;
	
	//float* gridy = (float *)malloc((host_ngridy+1) * sizeof(float));
    if (cudaMalloc((void**)&devptr, (host_ngridy+1) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_gridy=devptr;
	

	//float* data
    if (cudaMalloc((void**)&devptr, (host_dx*host_dy*host_dz) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_data=devptr;
	
	//float* simdata
    if (cudaMalloc((void**)&devptr, (host_dx*host_dy*host_dz) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_simdata=devptr;

	//float* theta
    if (cudaMalloc((void**)&devptr, (host_dx) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_theta=devptr;
	
	//float* recon
    if (cudaMalloc((void**)&devptr, (host_dy*host_ngridx*host_ngridy) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_recon=devptr;
	
	//float* host_reg_pars
    if (cudaMalloc((void**)&devptr, (2) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_reg_pars=devptr;

//    float *dev_sinp;
    if (cudaMalloc((void**)&devptr, (host_dx) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_sinp=devptr;

//	float *dev_cosp;
     if (cudaMalloc((void**)&devptr, (host_dx) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_cosp=devptr;

//	int *dev_quadrant;
    if (cudaMalloc((void**)&devptr, (host_dx) * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	dev_quadrant=(int *)devptr;

	if (host_num_block>0)
	{
	//	float *dev_ind_block;
		if (cudaMalloc((void**)&devptr, (host_dx) * sizeof(float)) != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		dev_ind_block=devptr;
		cudaStatus = cudaMemcpy(dev_ind_block, host_ind_block, (host_dx) * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	cudaStatus = cudaMemcpy(dev_data, host_data, (host_dx*host_dy*host_dz) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(dev_theta, host_theta, (host_dx) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(dev_recon, host_recon, (host_dy*host_ngridx*host_ngridy) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


	cudaStatus = cudaMemcpy(dev_reg_pars, host_reg_pars, (2) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_sinp, host_sinp, (host_dx) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_cosp, host_cosp, (host_dx) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

		cudaStatus = cudaMemcpy(dev_quadrant, host_quadrant, (host_dx) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

/*	
    cudaStatus = cudaMemcpy(*pdev_gridx, gridx, (num_grid+1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_gridy, gridy, (num_grid+1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
*/	
//	printf("INFO: Memory on Device %d allocated and copied!\n",deviceid);
	
Error:
	return cudaStatus;
}


void CCudaData::cleanCuda(){
	cudaError_t cudaStatus;
	
    cudaFree(dev_sum_dist);
	cudaFree(dev_simdata);
    cudaFree(dev_E);
    cudaFree(dev_F);
    cudaFree(dev_G);
    cudaFree(dev_gridx);
    cudaFree(dev_gridy);
    cudaFree(dev_data);
    cudaFree(dev_theta);
    cudaFree(dev_recon);
	cudaFree(dev_reg_pars);
    cudaFree(dev_sinp);
    cudaFree(dev_cosp);
    cudaFree(dev_quadrant);
	if (host_num_block>0)
	{
	    cudaFree(dev_ind_block);
	}
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
    }
//	else
//		printf("INFO: Cuda device reset!\n");
	
};

CCudaData::CCudaData(float *data, const int dx, const int dy, const int dz, float *center, float *theta,
		float *recon, const int ngridx, const int ngridy, const int num_iter, float *reg_pars, int num_block, float* ind_block)
{
	host_data=data;
	host_dx=dx;
	host_dy=dy;
	host_dz=dz;
	host_center=center;
	host_theta=theta;
	host_recon=recon;
	host_ngridx=ngridx;
	host_ngridy=ngridy;
	host_num_iter=num_iter;
	host_reg_pars=reg_pars;
	host_num_block=num_block;
	host_ind_block=ind_block;

	// Calculate the sin and cos values
    // of the projection angle and find
    // at which quadrant on the cartesian grid.
	host_sinp=new float[dx];
	host_cosp=new float[dx];
	host_quadrant=new int[dx];

    for (int i=0;i<dx;i++){
    	host_sinp[i]=sinf(theta[i]);
    	host_cosp[i]=cosf(theta[i]);
    	host_quadrant[i]=calc_quadrant_cuda(theta[i]);
    }
	
	if ((ngridx > max_ngridx)||(ngridy > max_ngridy))
	{
		printf("ERROR: ngridx or ngridy too large!\n");
		throw cudaErrorMemoryAllocation;
	}
	cudaError_t cudaStatus=initCuda();
	if (cudaStatus!=cudaSuccess)
		throw cudaStatus;


}

CCudaData::~CCudaData()
{
	cleanCuda();
	delete [] host_sinp;
	delete [] host_cosp;
	delete [] host_quadrant;
}

void CCudaData::cleanEFGS()
{
	cudaMemset(dev_E,0,(host_ngridx*host_ngridy) * sizeof(float));
	cudaMemset(dev_F,0,(host_ngridx*host_ngridy) * sizeof(float));
	cudaMemset(dev_G,0,(host_ngridx*host_ngridy) * sizeof(float));
	cudaMemset(dev_sum_dist,0,(host_ngridx*host_ngridy) * sizeof(float));
	/*
	thrust::device_ptr<float> dev_ptr(dev_E);
	thrust::fill(dev_ptr, dev_ptr + host_ngridx*host_ngridy, 0.0f);
	dev_ptr = thrust::device_ptr<float>(dev_F);
	thrust::fill(dev_ptr, dev_ptr + host_ngridx*host_ngridy, 0.0f);
	dev_ptr = thrust::device_ptr<float>(dev_G);
	thrust::fill(dev_ptr, dev_ptr + host_ngridx*host_ngridy, 0.0f);
	dev_ptr = thrust::device_ptr<float>(dev_sum_dist);
	thrust::fill(dev_ptr, dev_ptr + host_ngridx*host_ngridy, 0.0f);
	*/
}

void CCudaData::cleanSimdata()
{
	cudaMemset(dev_simdata,0,(host_dx*host_dy*host_dz) * sizeof(float));
	/*
	thrust::device_ptr<float> dev_ptr(dev_simdata);
	thrust::fill(dev_ptr, dev_ptr + host_dx*host_dy*host_dz, 0.0f);
	*/
}

cudaError_t CCudaData::retrieveRecon()
{
	cudaError_t cudaStatus = cudaMemcpy(host_recon, dev_recon, (host_dy*host_ngridx*host_ngridy) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed on retrieve recon");
    }
	return cudaStatus;
}


//////////////////////////////Kernels//////////////////////////////////////////


__global__ void _updateRecon_quad_cuda(int s, int ngridx, int ngridy, float *dev_E, float *dev_F, float *dev_G, float *dev_sum_dist, float *dev_recon)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = blockIdx.y*blockDim.y + threadIdx.y;
	uint k = s;

	if (/*(k>=s)||*/(n>=ngridx)||(m>=ngridy))
		return;

	int q = n + m*ngridy;
	dev_G[q] += dev_sum_dist[q];
    if (dev_F[q] != 0.0) {
        int ind0 = q + s*ngridx*ngridy;
        float sss = dev_G[q]*dev_G[q]-8*dev_E[q]*dev_F[q];
        if (sss>0)
        	dev_recon[ind0] = (-dev_G[q]+sqrt(sss))/(4*dev_F[q]);
	}

	/*
	int q=0;
	int ind0;
    for (int n = 0; n < ngridx; n++) {
        for (int m = 0; m < ngridy; m++) {
            q = m + n*ngridy;
			dev_G[q] += dev_sum_dist[q];
            if (dev_F[q] != 0.0) {
                ind0 = q + s*ngridx*ngridy;
                dev_recon[ind0] = (-dev_G[q]+sqrt(dev_G[q]*dev_G[q]-8*dev_E[q]*dev_F[q]))/(4*dev_F[q]);
            }
        }
    }
	return cudaSuccess;
	*/
}

cudaError_t  updateReconCUDA(CCudaData* pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_ngridx,blockx),iDivUp(pData->host_ngridy,blocky));//,s);
	dim3 size(blockx,blocky);
	
	_updateRecon_quad_cuda<<<grid, size>>>(s, 
		pData->host_ngridx, pData->host_ngridy, pData->dev_E, pData->dev_F, pData->dev_sum_dist, pData->dev_G, pData->dev_recon);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_updateRecon_quad_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _updateRecon_quad_cuda!\n");
	}
	return cudaStatus;

}


__global__ void _pml_forEachProjectionPixelCUDA(const int dx, const int dy, const int dz,
		const int ngridx, const int ngridy, const int s, float* dev_sinp, float* dev_cosp, int* dev_quadrant, float mov,
		float *dev_gridx, float *dev_gridy,float * dev_data,float * dev_recon, float * dev_simdata, float *dev_E, float* dev_sum_dist)
{
    uint p = blockIdx.x*blockDim.x + threadIdx.x;
    uint d = blockIdx.y*blockDim.y + threadIdx.y;
	uint k = s;

	if (/*(k>=s)||*/(p>=dx)||(d>=dz))
		return;
		int csize;
	float upd;
	int ind_data, ind_recon;

    float dist [max_ngridx+max_ngridy];
    int indi [max_ngridx+max_ngridy];

	float sin_p=dev_sinp[p];
	float cos_p=dev_cosp[p];
	int quadrant=dev_quadrant[p];

    pml_preprocess_cuda( ngridx, ngridy, dz, d, mov,
    		quadrant, sin_p, cos_p, dev_gridx, dev_gridy,
    		indi, dist, &csize);


    // Calculate simdata
    calc_simdata_cuda(p, s, d, ngridx, ngridy, dy, dz,
        csize, indi, dist, dev_recon,
        dev_simdata); // Output: simdata


    // Calculate dist*dist
    float sum_dist2 = 0.0;
    for (int n=0; n<csize-1; n++)
    {
        sum_dist2 += dist[n]*dist[n];

//		dev_sum_dist[indi[n]] += dist[n];
		atomicAdd(&(dev_sum_dist[indi[n]]),dist[n]);
    }

    // Update
    if (sum_dist2 != 0.0)
    {
        ind_data = d+s*dz+p*dy*dz;
        ind_recon = s*ngridx*ngridy;
        upd = dev_data[ind_data]/dev_simdata[ind_data];
        for (int n=0; n<csize-1; n++)
        {
//            dev_E[indi[n]] -= dev_recon[indi[n]+ind_recon]*upd*dist[n];
			atomicAdd(&(dev_E[indi[n]]),-dev_recon[indi[n]+ind_recon]*upd*dist[n]);
        }
    }
}

cudaError_t pml_forEachProjectionPixelCUDA(CCudaData* pData, int s, float mov)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_dx,blockx),iDivUp(pData->host_dz,blocky));//,s);
	dim3 size(blockx,blocky);
	
	_pml_forEachProjectionPixelCUDA<<<grid, size>>>(pData->host_dx, pData->host_dy, pData->host_dz, pData->host_ngridx, pData->host_ngridy,s,
		pData->dev_sinp,pData->dev_cosp,pData->dev_quadrant,mov,pData->dev_gridx,pData->dev_gridy,	pData->dev_data, 
		pData->dev_recon, pData->dev_simdata, pData->dev_E, pData->dev_sum_dist);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_pml_forEachProjectionPixelCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _pml_forEachProjectionPixelCUDA!\n");
	}
	return cudaStatus;

}


__global__ void _ospml_forEachProjectionPixelCUDA(const int dx, const int dy, const int dz,
		const int ngridx, const int ngridy, const int s, float* dev_sinp, float* dev_cosp, int* dev_quadrant, float mov,
		float *dev_gridx, float *dev_gridy,float * dev_data,float * dev_recon, float * dev_simdata, float *dev_E, float* dev_sum_dist,
		int subset_ind1, int subset_ind2, int os, float *dev_ind_block)
{
    uint q = blockIdx.x*blockDim.x + threadIdx.x;
    uint d = blockIdx.y*blockDim.y + threadIdx.y;
	uint k = s;

	if (/*(k>=s)||*/(q>=subset_ind2)||(d>=dz))
		return;
	uint p = dev_ind_block[q+os*subset_ind1];

	int csize;
	float upd;
	int ind_data, ind_recon;

    float dist [max_ngridx+max_ngridy];
    int indi [max_ngridx+max_ngridy];

	float sin_p=dev_sinp[p];
	float cos_p=dev_cosp[p];
	int quadrant=dev_quadrant[p];

    pml_preprocess_cuda( ngridx, ngridy, dz, d, mov,
    		quadrant, sin_p, cos_p, dev_gridx, dev_gridy,
    		indi, dist, &csize);


    // Calculate simdata
    calc_simdata_cuda(p, s, d, ngridx, ngridy, dy, dz,
        csize, indi, dist, dev_recon,
        dev_simdata); // Output: simdata


    // Calculate dist*dist
    float sum_dist2 = 0.0;
    for (int n=0; n<csize-1; n++)
    {
        sum_dist2 += dist[n]*dist[n];

//		dev_sum_dist[indi[n]] += dist[n];
		atomicAdd(&(dev_sum_dist[indi[n]]),dist[n]);
    }

    // Update
    if (sum_dist2 != 0.0)
    {
        ind_data = d+s*dz+p*dy*dz;
        ind_recon = s*ngridx*ngridy;
        upd = dev_data[ind_data]/dev_simdata[ind_data];
        for (int n=0; n<csize-1; n++)
        {
//            dev_E[indi[n]] -= dev_recon[indi[n]+ind_recon]*upd*dist[n];
			atomicAdd(&(dev_E[indi[n]]),-dev_recon[indi[n]+ind_recon]*upd*dist[n]);
        }
    }
}

cudaError_t ospml_forEachProjectionPixelCUDA(CCudaData *pData, int s, float mov, int subset_ind1, int subset_ind2, int os)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(subset_ind2,blockx),iDivUp(pData->host_dz,blocky));//,s);
	dim3 size(blockx,blocky);
	
	_ospml_forEachProjectionPixelCUDA<<<grid, size>>>(pData->host_dx, pData->host_dy, pData->host_dz, pData->host_ngridx, pData->host_ngridy,s,
		pData->dev_sinp,pData->dev_cosp,pData->dev_quadrant,mov,pData->dev_gridx,pData->dev_gridy,	pData->dev_data, 
		pData->dev_recon, pData->dev_simdata, pData->dev_E, pData->dev_sum_dist,subset_ind1,subset_ind2,os,pData->dev_ind_block);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_ospml_forEachProjectionPixelCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _ospml_forEachProjectionPixelCUDA!\n");
	}
	return cudaStatus;

}

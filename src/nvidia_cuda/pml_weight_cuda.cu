
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


__global__ void _weightInnerkernel_quad_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];
    const float totalwg = 4+4/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/totalwg, 1/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg};
	float mg[8];

	if (/*(k>=s)||*/(n<1)||(n>=(ngridx-1))||(m<1)||(m>=(ngridy-1)))
		return;

//    ind0 = m + n*ngridy + k*ngridx*ngridy;
	ind0 = m + n*ngridy;
    ind1 = ind0 + k*ngridx*ngridy;
                    
    indg[0] = ind1+1;
    indg[1] = ind1-1;
    indg[2] = ind1+ngridy;
    indg[3] = ind1-ngridy;
    indg[4] = ind1+ngridy+1;
    indg[5] = ind1+ngridy-1;
    indg[6] = ind1-ngridy+1;
    indg[7] = ind1-ngridy-1;

	for (int q = 0; q < 8; q++) {
		mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
		dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
		dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
	}
	/*
    for (q = 0; q < 8; q++) {
        dev_F[ind0] += 2*beta*dev_wg8[q];
        dev_G[ind0] -= 2*beta*dev_wg8[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
	*/
}


cudaError_t  weightInnerQuadCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_ngridx-2,blockx),iDivUp(pData->host_ngridy-2,blocky));//,s);
	dim3 size(blockx,blocky);
	
	_weightInnerkernel_quad_cuda<<<grid, size>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightInnerkernel_quad_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightInnerkernel_quad_cuda!\n");
	}
	return cudaStatus;
}


__global__ void _weightEdgeLRkernel_quad_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
//    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];

    const float totalwg = 3+2/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg};
	float mg[8];

	if (/*(k>=s)||*/(m<1)||(m>=(ngridx-1)))
		return;

	ind0 = m*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1-ngridy;
    indg[3] = ind1+ngridy+1;
    indg[4] = ind1-ngridy+1;

    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }

	ind0 = (ngridy-1) + m*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1-ngridy;
    indg[3] = ind1+ngridy-1;
    indg[4] = ind1-ngridy-1;

    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }
}


cudaError_t  weightEdgeLRQuadCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_ngridx-2,blockx*blocky));//,s);
	dim3 size(blockx*blocky);
	
	_weightEdgeLRkernel_quad_cuda<<<grid, size>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightEdgeLRkernel_quad_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightEdgeLRkernel_quad_cuda!\n");
	}
	return cudaStatus;
}


__global__ void _weightEdgeTBkernel_quad_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
//    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];

    const float totalwg = 3+2/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg};
	float mg[8];

	if (/*(k>=s)||*/(m<1)||(m>=(ngridy-1)))
		return;

    ind0 = m;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1-1;
    indg[2] = ind1+ngridy;
    indg[3] = ind1+ngridy+1;
    indg[4] = ind1+ngridy-1;

    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }

    ind0 = m + (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1-1;
    indg[2] = ind1-ngridy;
    indg[3] = ind1-ngridy+1;
    indg[4] = ind1-ngridy-1;


    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }
}


cudaError_t  weightEdgeTBQuadCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_ngridy-2,blockx*blocky));//,s);
	dim3 size(blockx*blocky);
	
	_weightEdgeTBkernel_quad_cuda<<<grid, size>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightEdgeTBkernel_quad_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightEdgeTBkernel_quad_cuda!\n");
	}
	return cudaStatus;
}


__global__ void _weightCornerkernel_quad_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
//    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];

    const float totalwg = 2+1/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/sqrt_2/totalwg};
	float mg[8];

    // (top-left)
    ind0 = 0;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1+ngridy+1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }

    // (top-right)
    ind0 = (ngridy-1);
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1+ngridy-1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }

    // (bottom-left)
    ind0 = (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1-ngridy;
    indg[2] = ind1-ngridy+1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }

    // (bottom-right)
    ind0 = (ngridy-1) + (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1-ngridy;
    indg[2] = ind1-ngridy-1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*mg[q];
    }
}

cudaError_t  weightCornerQuadCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	
	_weightCornerkernel_quad_cuda<<<1, 1>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightCornerkernel_quad_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightCornerkernel_quad_cuda!\n");
	}
	return cudaStatus;
}


cudaError_t quad_weightProjection(CCudaData *pData, int s)
{
	cudaError_t cudaStatus;
    // Weights for inner neighborhoods.
	cudaStatus = weightInnerQuadCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

	cudaStatus = weightEdgeLRQuadCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

	cudaStatus = weightCornerQuadCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

	cudaStatus = weightEdgeTBQuadCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

Error:
	return cudaStatus;
}

__global__ void _weightInnerkernel_hybrid_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];
    const float totalwg = 4+4/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/totalwg, 1/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg};
	float mg[8], rg[8], gammag[8];

	if (/*(k>=s)||*/(n<1)||(n>=(ngridx-1))||(m<1)||(m>=(ngridy-1)))
		return;

//    ind0 = m + n*ngridy + k*ngridx*ngridy;
	ind0 = m + n*ngridy;
    ind1 = ind0 + k*ngridx*ngridy;
                    
    indg[0] = ind1+1;
    indg[1] = ind1-1;
    indg[2] = ind1+ngridy;
    indg[3] = ind1-ngridy;
    indg[4] = ind1+ngridy+1;
    indg[5] = ind1+ngridy-1;
    indg[6] = ind1-ngridy+1;
    indg[7] = ind1-ngridy-1;

	for (int q = 0; q < 8; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
	}
	/*
    for (q = 0; q < 8; q++) {
        mg[q] = recon[ind1]+recon[indg[q]];
        rg[q] = recon[ind1]-recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
        F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
        G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }
	*/
}


cudaError_t  weightInnerHybridCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_ngridx-2,blockx),iDivUp(pData->host_ngridy-2,blocky));//,s);
	dim3 size(blockx,blocky);
	
	_weightInnerkernel_hybrid_cuda<<<grid, size>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightInnerkernel_hybrid_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightInnerkernel_hybrid_cuda!\n");
	}
	return cudaStatus;
}


__global__ void _weightEdgeLRkernel_hybrid_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
//    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];

    const float totalwg = 3+2/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg};
	float mg[8], rg[8], gammag[8];

	if (/*(k>=s)||*/(m<1)||(m>=(ngridx-1)))
		return;

	ind0 = m*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1-ngridy;
    indg[3] = ind1+ngridy+1;
    indg[4] = ind1-ngridy+1;

    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

	ind0 = (ngridy-1) + m*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1-ngridy;
    indg[3] = ind1+ngridy-1;
    indg[4] = ind1-ngridy-1;

    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }
}


cudaError_t  weightEdgeLRHybridCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_ngridx-2,blockx*blocky));//,s);
	dim3 size(blockx*blocky);
	
	_weightEdgeLRkernel_hybrid_cuda<<<grid, size>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightEdgeLRkernel_hybrid_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightEdgeLRkernel_hybrid_cuda!\n");
	}
	return cudaStatus;
}


__global__ void _weightEdgeTBkernel_hybrid_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
//    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];

    const float totalwg = 3+2/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/totalwg, 1/sqrt_2/totalwg, 1/sqrt_2/totalwg};
	float mg[8], rg[8], gammag[8];

	if (/*(k>=s)||*/(m<1)||(m>=(ngridy-1)))
		return;

    ind0 = m;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1-1;
    indg[2] = ind1+ngridy;
    indg[3] = ind1+ngridy+1;
    indg[4] = ind1+ngridy-1;

    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    ind0 = m + (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1-1;
    indg[2] = ind1-ngridy;
    indg[3] = ind1-ngridy+1;
    indg[4] = ind1-ngridy-1;


    for (int q = 0; q < 5; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }
}


cudaError_t  weightEdgeTBHybridCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	dim3 grid(iDivUp(pData->host_ngridy-2,blockx*blocky));//,s);
	dim3 size(blockx*blocky);
	
	_weightEdgeTBkernel_hybrid_cuda<<<grid, size>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightEdgeTBkernel_hybrid_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightEdgeTBkernel_hybrid_cuda!\n");
	}
	return cudaStatus;
}


__global__ void _weightCornerkernel_hybrid_cuda(int s, int ngridx, int ngridy, float *dev_F, float *dev_G, float *dev_recon, float *dev_reg_pars)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
//    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
//	uint k = blockIdx.z;
	uint k = s;
    int ind0, ind1, indg[8];

    const float totalwg = 2+1/sqrt_2;
	const float wg[] = {1/totalwg, 1/totalwg, 1/sqrt_2/totalwg};
	float mg[8], rg[8], gammag[8];

    // (top-left)
    ind0 = 0;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1+ngridy+1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    // (top-right)
    ind0 = (ngridy-1);
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1+ngridy-1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    // (bottom-left)
    ind0 = (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1-ngridy;
    indg[2] = ind1-ngridy+1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    // (bottom-right)
    ind0 = (ngridy-1) + (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1-ngridy;
    indg[2] = ind1-ngridy-1;

    for (int q = 0; q < 3; q++) {
        mg[q] = dev_recon[ind1]+dev_recon[indg[q]];
        rg[q] = dev_recon[ind1]-dev_recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/dev_reg_pars[1]));
        dev_F[ind0] += 2*dev_reg_pars[0]*wg[q]*gammag[q];
        dev_G[ind0] -= 2*dev_reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }
}

cudaError_t  weightCornerHybridCUDA(CCudaData *pData, int s)
{
	cudaError_t cudaStatus=cudaSuccess;
	
	_weightCornerkernel_hybrid_cuda<<<1, 1>>>(s, pData->host_ngridx, pData->host_ngridy, pData->dev_F, pData->dev_G, pData->dev_recon, pData->dev_reg_pars);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
	    fprintf(stderr, "_weightCornerkernel_hybrid_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
		
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus=cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightCornerkernel_hybrid_cuda!\n");
	}
	return cudaStatus;
}

cudaError_t hybrid_weightProjection(CCudaData *pData, int s)
{
	cudaError_t cudaStatus;
    // Weights for inner neighborhoods.
	cudaStatus = weightInnerHybridCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

	cudaStatus = weightEdgeLRHybridCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

	cudaStatus = weightCornerHybridCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

	cudaStatus = weightEdgeTBHybridCUDA(pData, s);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "call Kernel failed!");
        goto Error;
    }

Error:
	return cudaStatus;
}

/*
    Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com
    This file is part of TomograPeri.
    TomograPeri is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    TomograPeri is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.
*/



#ifndef UTILS_CUDA_CUH_
#define UTILS_CUDA_CUH_


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif


#define max_ngridx 256
#define max_ngridy 256

#define sqrt_2 1.41421356237

#define blockx 16
#define blocky 16


inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

typedef unsigned int  uint;

class CCudaData
{
public:
	float* dev_sum_dist;
	float* dev_simdata;
	float* dev_E;
	float* dev_F;
	float* dev_G;
	float* dev_gridx;
	float* dev_gridy;
	float* dev_data;
	float* dev_theta;
	float* dev_recon;
	float* dev_reg_pars;
    float *dev_sinp;
	float *dev_cosp;
    int *dev_quadrant;
	float *dev_ind_block;

	float* host_data;
	int host_dx;
	int host_dy;
	int host_dz;
	float *host_center;
	float *host_theta;
	float *host_recon;
	int host_ngridx;
	int host_ngridy;
	int host_num_iter;
	float *host_reg_pars;
	int host_num_block;
	float *host_ind_block;

    float *host_sinp;
	float *host_cosp;
    int *host_quadrant;

public:
//	CCudaData(float *data, const int dx, const int dy, const int dz, float *center, float *theta,
//		float *recon, const int ngridx, const int ngridy, const int num_iter, float *reg_pars);
	CCudaData(float *data, const int dx, const int dy, const int dz, float *center, float *theta,
		float *recon, const int ngridx, const int ngridy, const int num_iter, float *reg_pars, int num_block=-1, float* ind_block=0);
	~CCudaData();

	cudaError_t retrieveRecon();
	void cleanSimdata();
	void cleanEFGS();
private:
	cudaError_t initCuda();
	void cleanCuda();
};

int
calc_quadrant_cuda(
    float theta_p);

__host__ __device__
void pml_preprocess_cuda( const int ngridx, const int ngridy, int dz, int d, float mov,
		int quadrant, float sin_p, float cos_p, float *gridx, float *gridy,
		int *indi, float *dist, int *pcsize);

void preprocessing_cuda(
    int ngridx, int ngridy,
    int dz,
    float center, float *mov,
    float *gridx, float *gridy);

__host__ __device__
void calc_coords_cuda(
    int ngridx, int ngridy,
    float xi, float yi,
    float sin_p, float cos_p,
    float *gridx, float *gridy,
    float *coordx, float *coordy);

__host__ __device__
void trim_coords_cuda(
    int ngridx, int ngridy,
    float *coordx, float *coordy,
    float *gridx, float *gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by);

__host__ __device__
void sort_intersections_cuda(
    int ind_condition,
    int asize, float *ax, float *ay,
    int bsize, float *bx, float *by,
    int *csize,
    float *coorx, float *coory);

__host__ __device__
void calc_dist_cuda(
    int ngridx, int ngridy,
    int csize,
    float *coorx, float *coory,
    int *indi,
    float *dist);

__host__ __device__
void calc_simdata_cuda(
    int p, int s, int d,
    int ngridx, int ngridy,
    int dy, int dz,
    int csize,
    int *indi,
    float *dist,
    float *model,
    float *simdata);



cudaError_t pml_forEachProjectionPixelCUDA(CCudaData *pData, int s, float mov);
cudaError_t ospml_forEachProjectionPixelCUDA(CCudaData *pData, int s, float mov, int subset_ind1, int subset_ind2, int os);

cudaError_t quad_weightProjection(CCudaData *pData, int s);
cudaError_t hybrid_weightProjection(CCudaData *pData, int s);


cudaError_t updateReconCUDA(CCudaData *pData, int s);

#endif /* UTILS_CUDA_CUH_ */

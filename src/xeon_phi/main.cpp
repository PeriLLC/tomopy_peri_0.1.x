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


//#include "utils.h"
//#include "pml.h"
#include <math.h>
#include "recon.h"

#include "../cnpy/cnpy.h"

using namespace std;
using namespace cnpy;

#define LARGESET false


//  Windows
#ifdef _WIN32
#include <Windows.h>
double get_wall_time(){
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}
double get_cpu_time(){
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime |
            ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}

//  Posix/Linux
#else
#include <time.h>
#include <sys/time.h>
double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}
#endif


int assert_allclose(float *x, float *y, int len, float rtol=1e-5)
{
	int ret=0;
	for (int i=0;i<len;i++)
	{
		if (fabs(x[i]-y[i])>rtol)
		{
			ret ++;
//			cout << i << "->" <<x[i]<<":"<<y[i]<<endl;
		}
	}
	return ret;
}


int loadTestData(bool largeSet, int *pdx, int *pdy, int *pdz, int *pgridx, int *pgridy, float **pdata, float **ptheta,
		float **pospml_hybrid_recon,float **pospml_quad_recon,
		float **  ppml_hybrid_recon,float **  ppml_quad_recon)
{
	NpyArray data_npy =
			npy_load(largeSet?"../../test/data/pml/largeTestData.npy"
					: "../../test/data/pml/proj.npy");
	NpyArray theta_npy =
			npy_load(largeSet?"../../test/data/pml/largeTestTheta.npy"
					: "../../test/data/pml/angle.npy");
	NpyArray ospml_hybrid_npy =
			npy_load(largeSet?"../../test/data/pml/largeTestRecon_OH.npy"
					: "../../test/data/pml/ospml_hybrid.npy");
	NpyArray ospml_quad_npy   =
			npy_load(largeSet?"../../test/data/pml/largeTestRecon_OQ.npy"
					: "../../test/data/pml/ospml_quad.npy");
	NpyArray   pml_hybrid_npy =
			npy_load(largeSet?"../../test/data/pml/largeTestRecon_PH.npy"
					: "../../test/data/pml/pml_hybrid.npy");
	NpyArray   pml_quad_npy   =
			npy_load(largeSet?"../../test/data/pml/largeTestRecon_PQ.npy"
					: "../../test/data/pml/pml_quad.npy");

	if (data_npy.shape.size()==3)
	{
		*pdx =data_npy.shape[0];
		*pdy =data_npy.shape[1];
		*pdz =data_npy.shape[2];
		cout << "load data dx: " << *pdx << ", dy: " << *pdy << ", dz: " << *pdz << endl;
	}
	else
	{
		cout << "data shape error!" <<endl;
		return -1;
	}

	if ((theta_npy.shape.size()==1)&&(theta_npy.shape[0]==data_npy.shape[0]))
	{
		cout << "load theta counts: " << theta_npy.shape[0] << endl;
	}
	else
	{
		cout << "theta shape error!" <<endl;
		return -1;
	}

	float* loaded_data = reinterpret_cast<float*>(data_npy.data);
	double* loaded_theta = reinterpret_cast<double*>(theta_npy.data);

	int data_len=(*pdx)*(*pdy)*(*pdz);

	*pdata=new float[data_len];
	std::copy (loaded_data, loaded_data+data_len, *pdata);

	*ptheta=new float[*pdx];
	std::copy (loaded_theta, loaded_theta+*pdx, *ptheta);

	// will not check the validity of expected recon output. they all should be there and same size

	float* loaded_ospml_hybrid = reinterpret_cast<float*>(ospml_hybrid_npy.data);
	float* loaded_ospml_quad   = reinterpret_cast<float*>(ospml_quad_npy.data);
	float*   loaded_pml_hybrid = reinterpret_cast<float*>(  pml_hybrid_npy.data);
	float*   loaded_pml_quad   = reinterpret_cast<float*>(  pml_quad_npy.data);

	*pgridx=ospml_hybrid_npy.shape[1];
	*pgridy=ospml_hybrid_npy.shape[2];

	int recon_len=(*pdy)*(*pgridx)*(*pgridy);
	*pospml_hybrid_recon = new float [recon_len];
	*pospml_quad_recon = new float [recon_len];
	*  ppml_hybrid_recon = new float [recon_len];
	*  ppml_quad_recon = new float [recon_len];

	std::copy (loaded_ospml_hybrid, loaded_ospml_hybrid+recon_len, *pospml_hybrid_recon);
	std::copy (loaded_ospml_quad,   loaded_ospml_quad  +recon_len, *pospml_quad_recon);
	std::copy (  loaded_pml_hybrid,   loaded_pml_hybrid+recon_len, *  ppml_hybrid_recon);
	std::copy (  loaded_pml_quad,     loaded_pml_quad  +recon_len, *  ppml_quad_recon);

	delete[] data_npy.data;
	delete[] theta_npy.data;
	delete[] ospml_hybrid_npy.data;
	delete[] ospml_quad_npy.data;
	delete[] pml_hybrid_npy.data;
	delete[] pml_quad_npy.data;

	return 0;
}

int testData( bool largeSet){
	int dx,dy,dz,ngridx,ngridy;
	float *data,*theta;
	float *ospml_hybrid_recon, *ospml_quad_recon;
	float *  pml_hybrid_recon, *  pml_quad_recon;


    double wall0,cpu0,wall1,cpu1;

	cout << "Test the pml algorithms against provided "<< (largeSet ? "large": "small") <<" data sets." << endl;

	if (loadTestData(largeSet, &dx,&dy,&dz,&ngridx,&ngridy,&data,&theta,
			&ospml_hybrid_recon,&ospml_quad_recon,
			&  pml_hybrid_recon,&  pml_quad_recon)!=0)
		return -1;

	int recon_len=dy*ngridx*ngridy;
	float * recon = new float[recon_len];
	float * center = new float[dx];
	float * reg_pars = new float[10];
	float * ind_block = new float[dx];

	int num_iter = (largeSet)?2:4;

	for (int i=0;i<dx;i++)
		center[i]=(float)(dz)/2.0;

	for (int i=0;i<10;i++)
		reg_pars[i]=1.0;

	for (int i=0;i<dx;i++)
		ind_block[i]=i;


	cout << "-----------------MIC---------------------" << endl;

	for (int i=0;i<recon_len;i++)
		recon[i]=1e-6;
	cpu0  = get_cpu_time();
	wall0 = get_wall_time();
	ospml_quad(data, dx, dy, dz, center, theta,
		recon, ngridx, ngridy, num_iter, reg_pars, 1, ind_block);
	cpu1  = get_cpu_time();
	wall1 = get_wall_time();

	cout << "Testing ospml_quad, error counts: " <<
			assert_allclose(recon,ospml_quad_recon,dy*ngridx*ngridy) <<"/" << recon_len << endl;
    cout << "Wall Time = " << wall1 - wall0 << endl;
    cout << "CPU Time  = " << cpu1  - cpu0  << endl;

	for (int i=0;i<recon_len;i++)
		recon[i]=1e-6;
	cpu0  = get_cpu_time();
	wall0 = get_wall_time();
	ospml_hybrid(data, dx, dy, dz, center, theta,
		recon, ngridx, ngridy, num_iter, reg_pars, 1, ind_block);
	cpu1  = get_cpu_time();
	wall1 = get_wall_time();

	cout << "Testing ospml_hybrid, error counts: " <<
			assert_allclose(recon,ospml_hybrid_recon,dy*ngridx*ngridy) <<"/" << recon_len << endl;
    cout << "Wall Time = " << wall1 - wall0 << endl;
    cout << "CPU Time  = " << cpu1  - cpu0  << endl;

	for (int i=0;i<recon_len;i++)
		recon[i]=1e-6;
	cpu0  = get_cpu_time();
	wall0 = get_wall_time();
	pml_quad(data, dx, dy, dz, center, theta,
	    recon, ngridx, ngridy, num_iter, reg_pars);
	cpu1  = get_cpu_time();
	wall1 = get_wall_time();

	cout << "Testing pml_quad, error counts: " <<
			assert_allclose(recon,pml_quad_recon,dy*ngridx*ngridy) <<"/" << recon_len << endl;
    cout << "Wall Time = " << wall1 - wall0 << endl;
    cout << "CPU Time  = " << cpu1  - cpu0  << endl;

	for (int i=0;i<recon_len;i++)
		recon[i]=1e-6;
	cpu0  = get_cpu_time();
	wall0 = get_wall_time();
	pml_hybrid(data, dx, dy, dz, center, theta,
	    recon, ngridx, ngridy, num_iter, reg_pars);
	cpu1  = get_cpu_time();
	wall1 = get_wall_time();

	cout << "Testing pml_hybrid, error counts: " <<
			assert_allclose(recon,pml_hybrid_recon,dy*ngridx*ngridy) <<"/" << recon_len << endl;
    cout << "Wall Time = " << wall1 - wall0 << endl;
    cout << "CPU Time  = " << cpu1  - cpu0  << endl;

	delete []data;
	delete []theta;
	delete []recon;
	delete []center;
	delete []reg_pars;
	delete []ind_block;
	delete []ospml_hybrid_recon;
	delete []ospml_quad_recon;
	delete []pml_hybrid_recon;
	delete []pml_quad_recon;
	return 0;
}

int main()
{

	return testData(LARGESET);
}

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

/*
 * V.1.1.0 7_18_2015
 */

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <malloc.h>
#include <sys/time.h>
#include <assert.h>
#include "string.h"

#include "utils_cilk.h"
#include "pml_cilk.h"

//#include "advisor-annotate.h"  // Add to each module that contains Intel Advisor annotations
#include <cilk/cilk.h>

#pragma offload_attribute(push, target(mic))


void pml_hybrid_forEachProjection(const int dx, const int dy, const int dz,
		const int ngridx, const int ngridy, const int s, const int p, float sin_p, float cos_p, int quadrant, float mov,
		float *gridx, float *gridy,float * data,float * recon, float * simdata, float *E, float* sum_dist)
{
	int csize;
	float upd;
	int ind_data, ind_recon;

    // For each detector pixel
    for (int d=0; d<dz; d++)
    {
    	float dist [ngridx+ngridy];
    	int indi [ngridx+ngridy];


    	pml_preprocess_cilk( ngridx, ngridy, dz, d, mov,
    			quadrant, sin_p, cos_p, gridx, gridy,
    			indi, dist, &csize);


        // Calculate simdata
        calc_simdata_cilk(p, s, d, ngridx, ngridy, dy, dz,
            csize, indi, dist, recon,
            simdata); // Output: simdata


        // Calculate dist*dist
        float sum_dist2 = 0.0;
        for (int n=0; n<csize-1; n++)
        {
            sum_dist2 += dist[n]*dist[n];
//            ANNOTATE_LOCK_ACQUIRE(&sum_dist[indi[n]]);
            sum_dist[indi[n]] += dist[n];
//            ANNOTATE_LOCK_RELEASE(&sum_dist[indi[n]]);
        }

        // Update
        if (sum_dist2 != 0.0)
        {
            ind_data = d+s*dz+p*dy*dz;
            ind_recon = s*ngridx*ngridy;
            upd = data[ind_data]/simdata[ind_data];
            for (int n=0; n<csize-1; n++)
            {
//                ANNOTATE_LOCK_ACQUIRE(&E[indi[n]]);
                E[indi[n]] -= recon[indi[n]+ind_recon]*upd*dist[n];
//                ANNOTATE_LOCK_RELEASE(&E[indi[n]]);
            }
        }
    }
}

void pml_hybrid_forEachSlice(const int dx, const int dy, const int dz,
		const int ngridx, const int ngridy, int s, float* center,
		float * sinp, float* cosp, int* quadrant, float * data, float *reg_pars,
		float * recon, float * simdata)
{
	float mov;
	float *E, *F, *G;
	int ind0, ind1, indg[8];
	float totalwg, wg[8], mg[8], rg[8], gammag[8];
	float gridx[ngridx+1];
	float gridy[ngridy+1];

	preprocessing_cilk(ngridx, ngridy, dz, center[s],
        &mov, gridx, gridy); // Outputs: mov, gridx, gridy

	float *sum_dist;
//	float sum_dist2;
    sum_dist = (float *)calloc((ngridx*ngridy), sizeof(float));
    E = (float *)calloc((ngridx*ngridy), sizeof(float));
    F = (float *)calloc((ngridx*ngridy), sizeof(float));
    G = (float *)calloc((ngridx*ngridy), sizeof(float));

    assert( (sum_dist!=NULL)&&(E!=NULL)&&(F!=NULL)&&(G!=NULL));

    // For each projection angle
//    ANNOTATE_SITE_BEGIN( EachProjection );  // Place before the loop control statement to begin a parallel code region (parallel site).
    for (int p=0; p<dx; p++)
    {
//       	ANNOTATE_ITERATION_TASK( MyTask1 );  // Place at the start of loop body. This annotation identifies an entire body as a task.
    	pml_hybrid_forEachProjection(dx, dy, dz, ngridx, ngridy,s,p,sinp[p],cosp[p],quadrant[p],mov,gridx,gridy,
    			data, recon, simdata, E, sum_dist);
    }
//    ANNOTATE_SITE_END();


    // Weights for inner neighborhoods.
    totalwg = 4+4/sqrt(2);
    wg[0] = 1/totalwg;
    wg[1] = 1/totalwg;
    wg[2] = 1/totalwg;
    wg[3] = 1/totalwg;
    wg[4] = 1/sqrt(2)/totalwg;
    wg[5] = 1/sqrt(2)/totalwg;
    wg[6] = 1/sqrt(2)/totalwg;
    wg[7] = 1/sqrt(2)/totalwg;

    // (inner region)
    for (int n = 1; n < ngridx-1; n++) {
        for (int m = 1; m < ngridy-1; m++) {
            ind0 = m + n*ngridy;
            ind1 = ind0 + s*ngridx*ngridy;

            indg[0] = ind1+1;
            indg[1] = ind1-1;
            indg[2] = ind1+ngridy;
            indg[3] = ind1-ngridy;
            indg[4] = ind1+ngridy+1;
            indg[5] = ind1+ngridy-1;
            indg[6] = ind1-ngridy+1;
            indg[7] = ind1-ngridy-1;


            for (int q = 0; q < 8; q++) {
                mg[q] = recon[ind1]+recon[indg[q]];
                rg[q] = recon[ind1]-recon[indg[q]];
                gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
                F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
                G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
            }
        }
    }

    // Weights for edges.
    totalwg = 3+2/sqrt(2);
    wg[0] = 1/totalwg;
    wg[1] = 1/totalwg;
    wg[2] = 1/totalwg;
    wg[3] = 1/sqrt(2)/totalwg;
    wg[4] = 1/sqrt(2)/totalwg;

    // (top)
    for (int m = 1; m < ngridy-1; m++) {
        ind0 = m;
        ind1 = ind0 + s*ngridx*ngridy;

        indg[0] = ind1+1;
        indg[1] = ind1-1;
        indg[2] = ind1+ngridy;
        indg[3] = ind1+ngridy+1;
        indg[4] = ind1+ngridy-1;

        for (int q = 0; q < 5; q++) {
            mg[q] = recon[ind1]+recon[indg[q]];
            rg[q] = recon[ind1]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
            F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
            G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
        }
    }

    // (bottom)
    for (int m = 1; m < ngridy-1; m++) {
        ind0 = m + (ngridx-1)*ngridy;
        ind1 = ind0 + s*ngridx*ngridy;

        indg[0] = ind1+1;
        indg[1] = ind1-1;
        indg[2] = ind1-ngridy;
        indg[3] = ind1-ngridy+1;
        indg[4] = ind1-ngridy-1;

        for (int q = 0; q < 5; q++) {
            mg[q] = recon[ind1]+recon[indg[q]];
            rg[q] = recon[ind1]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
            F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
            G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
        }
    }

    // (left)
    for (int n = 1; n < ngridx-1; n++) {
        ind0 = n*ngridy;
        ind1 = ind0 + s*ngridx*ngridy;

        indg[0] = ind1+1;
        indg[1] = ind1+ngridy;
        indg[2] = ind1-ngridy;
        indg[3] = ind1+ngridy+1;
        indg[4] = ind1-ngridy+1;

        for (int q = 0; q < 5; q++) {
            mg[q] = recon[ind1]+recon[indg[q]];
            rg[q] = recon[ind1]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
            F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
            G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
        }
    }

    // (right)
    for (int n = 1; n < ngridx-1; n++) {
        ind0 = (ngridy-1) + n*ngridy;
        ind1 = ind0 + s*ngridx*ngridy;

        indg[0] = ind1-1;
        indg[1] = ind1+ngridy;
        indg[2] = ind1-ngridy;
        indg[3] = ind1+ngridy-1;
        indg[4] = ind1-ngridy-1;

        for (int q = 0; q < 5; q++) {
            mg[q] = recon[ind1]+recon[indg[q]];
            rg[q] = recon[ind1]-recon[indg[q]];
            gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
            F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
            G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
        }
    }

    // Weights for corners.
    totalwg = 2+1/sqrt(2);
    wg[0] = 1/totalwg;
    wg[1] = 1/totalwg;
    wg[2] = 1/sqrt(2)/totalwg;

    // (top-left)
    ind0 = 0;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1+ngridy+1;

    for (int q = 0; q < 3; q++) {
        mg[q] = recon[ind1]+recon[indg[q]];
        rg[q] = recon[ind1]-recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
        F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
        G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    // (top-right)
    ind0 = (ngridy-1);
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1+ngridy;
    indg[2] = ind1+ngridy-1;

    for (int q = 0; q < 3; q++) {
        mg[q] = recon[ind1]+recon[indg[q]];
        rg[q] = recon[ind1]-recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
        F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
        G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    // (bottom-left)
    ind0 = (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1+1;
    indg[1] = ind1-ngridy;
    indg[2] = ind1-ngridy+1;

    for (int q = 0; q < 3; q++) {
        mg[q] = recon[ind1]+recon[indg[q]];
        rg[q] = recon[ind1]-recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
        F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
        G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    // (bottom-right)
    ind0 = (ngridy-1) + (ngridx-1)*ngridy;
    ind1 = ind0 + s*ngridx*ngridy;

    indg[0] = ind1-1;
    indg[1] = ind1-ngridy;
    indg[2] = ind1-ngridy-1;

    for (int q = 0; q < 3; q++) {
        mg[q] = recon[ind1]+recon[indg[q]];
        rg[q] = recon[ind1]-recon[indg[q]];
        gammag[q] = 1/(1+fabs(rg[q]/reg_pars[1]));
        F[ind0] += 2*reg_pars[0]*wg[q]*gammag[q];
        G[ind0] -= 2*reg_pars[0]*wg[q]*gammag[q]*mg[q];
    }

    int q = 0;
    for (int n = 0; n < ngridx*ngridy; n++) {
        G[q] += sum_dist[n];
        q++;
    }

    for (int n = 0; n < ngridx; n++) {
        for (int m = 0; m < ngridy; m++) {
        	int q = m + n*ngridy;
            if (F[q] != 0.0) {
                ind0 = q + s*ngridx*ngridy;
                recon[ind0] = (-G[q]+sqrt(G[q]*G[q]-8*E[q]*F[q]))/(4*F[q]);
            }
        }
    }

    free(sum_dist);
    free(E);
    free(F);
    free(G);

}

void pml_hybrid_forEachIter(const int dx, const int dy, const int dz, const int ngridx, const int ngridy, float* center,
		float* sinp, float* cosp, int* quadrant, float * data, float *reg_pars, float* recon
		, float *simdata)
{

	simdata[0:(dx*dy*dz)]=0.0;

//        ANNOTATE_SITE_BEGIN( EachSlice );  // Place before the loop control statement to begin a parallel code region (parallel site).
   	cilk_for (int s=0; s<dy; s++)
    {
	//        	ANNOTATE_ITERATION_TASK( MyTask1 );  // Place at the start of loop body. This annotation identifies an entire body as a task.
   		pml_hybrid_forEachSlice(dx, dy, dz, ngridx, ngridy, s, center,
        			sinp,cosp,quadrant, data,reg_pars, recon,  simdata);
    }
	//        ANNOTATE_SITE_END();  // End the parallel code region, after task execution completes

}

#pragma offload_attribute(pop)

void
pml_hybrid_cilk(
    float *data, const int dx, const int dy, const int dz, float *center, float *theta,
    float *recon, const int ngridx, const int ngridy, const int num_iter, float *reg_pars)
{

//    float *simdata=(float*)_mm_malloc((dx*dy*dz)*sizeof(float), 64);
//    simdata = (float *)malloc((dx*dy*dz)* sizeof(float));
//    assert(simdata!=NULL);

    // Calculate the sin and cos values
    // of the projection angle and find
    // at which quadrant on the cartesian grid.
    float sinp[dx],cosp[dx];
    int quadrant[dx];
    for (int i=0;i<dx;i++){
    	sinp[i]=sinf(theta[i]);
    	cosp[i]=cosf(theta[i]);
    	quadrant[i]=calc_quadrant_cilk(theta[i]);
    }

        // For each slice
#pragma offload target(mic) inout (recon : length(dy*ngridx*ngridy)), \
			in (center : length(dy)), in (sinp, cosp, quadrant : length(dx)),\
			in (data : length (dx*dy*dz)), in (reg_pars : length(1))
    {
//    	recon[0:dy*ngridx*ngridy]=1e-6;
    	float *simdata =(float*)_mm_malloc((dx*dy*dz)*sizeof(float), 64);
        for (int i=0; i<num_iter; i++)
        	pml_hybrid_forEachIter(dx, dy, dz, ngridx, ngridy, center,
        			sinp,cosp,quadrant, data,reg_pars, recon,  simdata);
        _mm_free(simdata);
    }


//        free(simdata);
//    _mm_free(simdata);
}

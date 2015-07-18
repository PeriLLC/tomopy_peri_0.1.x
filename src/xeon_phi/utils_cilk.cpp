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
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>
#include <assert.h>
#include "string.h"


#include "utils_cilk.h"

//#include "advisor-annotate.h"  // Add to each module that contains Intel Advisor annotations



int
calc_quadrant_cilk(
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

#pragma offload_attribute(push, target(mic))


void pml_preprocess_cilk( const int ngridx, const int ngridy, int dz, int d, float mov,
		int quadrant, float sin_p, float cos_p, float *gridx, float *gridy,
		int *indi, float *dist, int *pcsize)
{

    float yi = (1-dz)/2.0+d+mov;

    float coordx [ngridy+1];
    float coordy [ngridx+1];
    float ax [ngridx+ngridy];
    float ay [ngridx+ngridy];
    float bx [ngridx+ngridy];
    float by [ngridx+ngridy];
    float coorx [ngridx+ngridy];
    float coory [ngridx+ngridy];

    int asize, bsize;
    int csize;

    calc_coords_cilk(
        ngridx, ngridy, /*xi*/0, yi, sin_p, cos_p, gridx, gridy,
        coordx, coordy);

    // Merge the (coordx, gridy) and (gridx, coordy)
    trim_coords_cilk(
        ngridx, ngridy, coordx, coordy, gridx, gridy,
        &asize, ax, ay, &bsize, bx, by);

    // Sort the array of intersection points (ax, ay) and
    // (bx, by). The new sorted intersection points are
    // stored in (coorx, coory). Total number of points
    // are csize.
    sort_intersections_cilk(
        quadrant, asize, ax, ay, bsize, bx, by,
        &csize, coorx, coory);

    // Calculate the distances (dist) between the
    // intersection points (coorx, coory). Find the
    // indices of the pixels on the reconstruction grid.
    calc_dist_cilk(
        ngridx, ngridy, csize, coorx, coory,
        indi, dist);

    *pcsize=csize;
}

void
preprocessing_cilk(
    int ry, int rz,
    int num_pixels, float center,
    float *mov, float *gridx, float *gridy)
{
    int i;

    for(i=0; i<=ry; i++)
    {
        gridx[i] = -ry/2.+i;
    }

    for(i=0; i<=rz; i++)
    {
        gridy[i] = -rz/2.+i;
    }
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




void
calc_coords_cilk(
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


void
trim_coords_cilk(
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


void
sort_intersections_cilk(
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


void
calc_dist_cilk(
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


void
calc_simdata_cilk(
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

#pragma offload_attribute(pop)

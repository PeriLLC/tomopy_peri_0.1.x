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

#ifndef UTILS_CILK_H_
#define UTILS_CILK_H_


#include <cilk/cilk.h>

#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif


int
calc_quadrant_cilk(
    float theta_p);

#pragma offload_attribute(push, target(mic))

void pml_preprocess_cilk( const int ngridx, const int ngridy, int dz, int d, float mov,
		int quadrant, float sin_p, float cos_p, float *gridx, float *gridy,
		int *indi, float *dist, int *pcsize);

void
preprocessing_cilk(
    int ngridx, int ngridy,
    int dz,
    float center, float *mov,
    float *gridx, float *gridy);

void
calc_coords_cilk(
    int ngridx, int ngridy,
    float xi, float yi,
    float sin_p, float cos_p,
    float *gridx, float *gridy,
    float *coordx, float *coordy);

void
trim_coords_cilk(
    int ngridx, int ngridy,
    float *coordx, float *coordy,
    float *gridx, float *gridy,
    int *asize, float *ax, float *ay,
    int *bsize, float *bx, float *by);

void
sort_intersections_cilk(
    int ind_condition,
    int asize, float *ax, float *ay,
    int bsize, float *bx, float *by,
    int *csize,
    float *coorx, float *coory);

void
calc_dist_cilk(
    int ngridx, int ngridy,
    int csize,
    float *coorx, float *coory,
    int *indi,
    float *dist);

void
calc_simdata_cilk(
    int p, int s, int d,
    int ngridx, int ngridy,
    int dy, int dz,
    int csize,
    int *indi,
    float *dist,
    float *model,
    float *simdata);

#pragma offload_attribute(pop)

#endif /* UTILS_CILK_H_ */

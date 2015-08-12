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
 * V.1.1.1 8_3_2015
 */
#ifndef PML_CUDA_H_
#define PML_CUDA_H_

void
ospml_quad_cuda(
    float *data,
    int dx,
    int dy,
    int dz,
    float *center,
    float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    float *reg_pars,
    int num_block,
    float *ind_block);

void
ospml_hybrid_cuda(
    float *data,
    int dx,
    int dy,
    int dz,
    float *center,
    float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    float *reg_pars,
    int num_block,
    float *ind_block);

void
pml_quad_cuda(
    float *data,
    int dx,
    int dy,
    int dz,
    float *center,
    float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    float *reg_pars);

void
pml_hybrid_cuda(
    float *data,
    int dx,
    int dy,
    int dz,
    float *center,
    float *theta,
    float *recon,
    int ngridx,
    int ngridy,
    int num_iter,
    float *reg_pars);


#endif /* PML_CUDA_H_ */
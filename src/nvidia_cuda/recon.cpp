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


#include "../recon.h"
#include "pml_cuda.h"

extern "C" {

void
ospml_quad(
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
    float *ind_block)
{
ospml_quad_cuda(
    data, dx, dy, dz, center, theta, recon,
    ngridx, ngridy, num_iter, reg_pars,
    num_block, ind_block);
}

void
ospml_hybrid(
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
    float *ind_block)
{
ospml_hybrid_cuda(
    data, dx, dy, dz, center, theta, recon,
    ngridx, ngridy, num_iter, reg_pars,
    num_block, ind_block);
}


void
pml_quad(
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
    float *reg_pars)
{
pml_quad_cuda(
    data, dx, dy, dz, center, theta, recon,
    ngridx, ngridy, num_iter, reg_pars);
}


void
pml_hybrid(
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
    float *reg_pars)
{
pml_hybrid_cuda(
    data, dx, dy, dz, center, theta, recon,
    ngridx, ngridy, num_iter, reg_pars);
}


}


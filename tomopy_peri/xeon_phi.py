#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
#    Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com
#
#    This file is part of TomograPeri.
#
#    TomograPeri is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    TomograPeri is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import ctypes

# --------------------------------------------------------------------

# Get the shared library.
if os.name == 'nt':
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/recon_phi.dll'))
    librecon_phi = ctypes.CDLL(libpath)
else:
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'lib/recon_phi.so'))
    librecon_phi = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def pml_quad_phi(data, dx, dy, dz, center, theta, recon, n_gridx, n_gridy, num_iter, reg_pars):
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

    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    librecon_phi.pml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon_phi.pml_quad(data.ctypes.data_as(c_float_p),
                  theta.ctypes.data_as(c_float_p),
                  ctypes.c_float(center),
                  ctypes.c_int(num_projections),
                  ctypes.c_int(num_slices),
                  ctypes.c_int(num_pixels),
                  ctypes.c_int(num_grid),
                  ctypes.c_int(iters),
                  ctypes.c_float(beta),
                  init_matrix.ctypes.data_as(c_float_p))
    return init_matrix

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
import tomopy.util.dtype as dtype

# --------------------------------------------------------------------

# Get the shared library.
if os.name == 'nt':
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_phi.dll'))
    librecon_phi = ctypes.CDLL(libpath)
else:
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'lib/libtomoperi_phi.so'))
    librecon_phi = ctypes.CDLL(libpath)

# --------------------------------------------------------------------

def c_pml_quad(args):
    data=args[0]
    recon=args[6]
    # Call C function.
    c_float_p = ctypes.POINTER(ctypes.c_float)
    librecon_phi.pml_quad.restype = ctypes.POINTER(ctypes.c_void_p)
    librecon_phi.pml_quad(data.ctypes.data_as(c_float_p),
        dtype.as_c_int(args[1]),  # dx
        dtype.as_c_int(args[2]),  # dy
        dtype.as_c_int(args[3]),  # dz
        dtype.as_c_float_p(args[4]),  # center
        dtype.as_c_float_p(args[5]),  # theta
        recon.ctypes.data_as(c_float_p),
        dtype.as_c_int(args[7]['num_gridx']),
        dtype.as_c_int(args[7]['num_gridy']),
        dtype.as_c_int(args[7]['num_iter']),
        dtype.as_c_float_p(args[7]['reg_par']))
    return recon

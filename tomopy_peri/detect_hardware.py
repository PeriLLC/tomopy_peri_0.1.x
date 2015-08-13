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

def detect_hardware() :
    ihardwares= []

    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_gpu.dll'))
    libpath1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_gpu.so'))
    if os.path.isfile(libpath) or os.path.isfile(libpath1) :
        ihardwares.append( 'nVidia_GPU' );

    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_phi.dll'))
    libpath1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_phi.so'))
    if os.path.isfile(libpath) or os.path.isfile(libpath1) :
        ihardwares.append( 'Xeon_Phi' );

    return ihardwares

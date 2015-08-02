#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
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
'''

import numpy

from tomopy.io.phantom import *
from tomopy.sim.project import *
from tomopy.recon.algorithm import *

'''
import os, errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
'''


print('Creating Object...')
obj=shepp3d((128,128,128))
numpy.save('./data/largeTestObj.npy',obj)

print('Creating Angle...')
theta=angles(200)
numpy.save('./data/largeTestTheta.npy',theta)

print('Creating Projection...')
data=project(obj,theta)
numpy.save('./data/largeTestData.npy',data)

print('Creating ospml_hybrid...')
pmlrecon=recon(data, theta, algorithm='ospml_hybrid', num_iter=2)
numpy.save('./data/pml/largeTestRecon_OH.npy',pmlrecon)

print('Creating ospml_quad...')
pmlrecon=recon(data, theta, algorithm='ospml_quad', num_iter=2)
numpy.save('./data/pml/largeTestRecon_OQ.npy',pmlrecon)

print('Creating pml_hybrid...')
pmlrecon=recon(data, theta, algorithm='pml_hybrid', num_iter=2)
numpy.save('./data/pml/largeTestRecon_PH.npy',pmlrecon)

print('Creating pml_quad...')
pmlrecon=recon(data, theta, algorithm='pml_quad', num_iter=2)
numpy.save('./data/pml/largeTestRecon_PQ.npy',pmlrecon)


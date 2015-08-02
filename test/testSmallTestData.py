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
from tomopy.recon.algorithm import *
from tomopy.recon.acceleration import *
from numpy.testing import assert_allclose

print('Loading Angle...')
theta=numpy.load('data/angle.npy')

print('Loading Projection...')
data=numpy.load('data/proj.npy')


print('Testing ospml_quad...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='ospml_quad', num_iter=4),
    numpy.load('data/pml/ospml_quad.npy'), rtol=1e-4)
print('...ok!')

print('Testing pml_quad...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='pml_quad', num_iter=4),
    numpy.load('data/pml/pml_quad.npy'), rtol=1e-4)
print('...ok!')


print('Testing ospml_hybrid...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='ospml_hybrid', num_iter=4),
    numpy.load('data/pml/ospml_hybrid.npy'), rtol=1e-4)
print('...ok!')

print('Testing pml_hybrid...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='pml_hybrid', num_iter=4),
    numpy.load('data/pml/pml_hybrid.npy'), rtol=1e-4)
print('...ok!')



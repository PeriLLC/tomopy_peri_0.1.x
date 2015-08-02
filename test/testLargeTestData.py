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
from tomopy.recon.acceleration import *
from tomopy.recon.algorithm import *
from numpy.testing import assert_allclose

print('Loading Angle...')
theta=numpy.load('data/largeTestTheta.npy')

print('Loading Projection...')
data=numpy.load('data/largeTestData.npy')

print('Testing pml_quad...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='pml_quad', num_iter=2),
    numpy.load('data/pml/largeTestRecon_PQ.npy'), rtol=1e-4, atol=1e-5)
print('...ok!')

print('Testing ospml_quad...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='ospml_quad', num_iter=2),
    numpy.load('data/pml/largeTestRecon_OQ.npy'), rtol=1e-4, atol=1e-5)
print('...ok!')


print('Testing pml_hybrid...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='pml_hybrid', num_iter=2),
    numpy.load('data/pml/largeTestRecon_PH.npy'), rtol=1e-4, atol=1e-5)
print('...ok!')


print('Testing ospml_hybrid...')
assert_allclose(
    recon_accelerated(data, theta, algorithm='ospml_hybrid', num_iter=2),
    numpy.load('data/pml/largeTestRecon_OH.npy'), rtol=1e-4, atol=1e-5)
print('...ok!')



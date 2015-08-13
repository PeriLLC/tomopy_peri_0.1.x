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
from tomopy_peri.detect_hardware import detect_hardware

ihardware = detect_hardware()

print ihardware

print('Loading Angle...')
theta=numpy.load('data/largeTestTheta.npy')

print('Loading Projection...')
data=numpy.load('data/largeTestData.npy')

print('Loading Expected result...')
pq=numpy.load('data/pml/largeTestRecon_PQ.npy')
oq=numpy.load('data/pml/largeTestRecon_OQ.npy')
ph=numpy.load('data/pml/largeTestRecon_PH.npy')
oh=numpy.load('data/pml/largeTestRecon_OH.npy')

for i in ihardware:
    print('Testing pml_quad on %s...' % i)

    assert_allclose(
        recon_accelerated(data, theta, algorithm='pml_quad', hardware=i, num_iter=2),
        pq, rtol=1e-4, atol=1e-5)
    print('...ok!')

    print('Testing ospml_quad on %s...' % i)
    assert_allclose(
        recon_accelerated(data, theta, algorithm='ospml_quad', hardware=i, num_iter=2),
        oq, rtol=1e-4, atol=1e-5)
    print('...ok!')

    print('Testing pml_hybrid on %s...' % i)
    assert_allclose(
        recon_accelerated(data, theta, algorithm='pml_hybrid', hardware=i, num_iter=2),
        ph, rtol=1e-4, atol=1e-5)
    print('...ok!')

    print('Testing ospml_hybrid on %s...' % i)
    assert_allclose(
        recon_accelerated(data, theta, algorithm='ospml_hybrid', hardware=i, num_iter=2),
        oh, rtol=1e-4, atol=1e-5)
    print('...ok!')



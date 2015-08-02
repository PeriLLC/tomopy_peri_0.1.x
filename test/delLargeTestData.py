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

import os, errno

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

print('Removing Object...')
silentremove('./data/largeTestObj.npy')

print('Removing Angle...')
silentremove('./data/largeTestTheta.npy')

print('Removing Projection...')
silentremove('./data/largeTestData.npy')

print('Removing ospml_hybrid...')
silentremove('./data/pml/largeTestRecon_OH.npy')

print('Removing ospml_quad...')
silentremove('./data/pml/largeTestRecon_OQ.npy')

print('Removing pml_hybrid...')
silentremove('./data/pml/largeTestRecon_PH.npy')

print('Removing pml_quad...')
silentremove('./data/pml/largeTestRecon_PQ.npy')


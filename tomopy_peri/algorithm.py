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

from __future__ import absolute_import

import tomopy
import logging
import numpy as np
import tomopy.util.dtype as dtype
from tomopy.sim.project import angles, get_center
import tomopy_peri.xeon_phi as xeon_phi
import os.path

# --------------------------------------------------------------------
def recon_accelerated(
        tomo, theta, center=None, emission=True, algorithm=None, hardware=None,
        acc_option=None, init_recon=None, **kwargs):

    allowed_kwargs = {
        'ospml_hybrid': ['num_gridx', 'num_gridy', 'num_iter',
                         'reg_par', 'num_block', 'ind_block'],
        'ospml_quad': ['num_gridx', 'num_gridy', 'num_iter',
                       'reg_par', 'num_block', 'ind_block'],
        'pml_hybrid': ['num_gridx', 'num_gridy', 'num_iter', 'reg_par'],
        'pml_quad': ['num_gridx', 'num_gridy', 'num_iter', 'reg_par'],
    }

    generic_kwargs = ['num_gridx', 'num_gridy', 'options']

    # Generate kwargs for the algorithm.
    kwargs_defaults = _get_algorithm_kwargs(tomo.shape)

    if isinstance(algorithm, str):
        # Check whether we have an allowed method
        if not algorithm in allowed_kwargs:
            raise ValueError('Keyword "algorithm" must be one of %s.' %
                             (list(allowed_kwargs.keys()),))
        # Make sure have allowed kwargs appropriate for algorithm.
        for key in kwargs:
            if key not in allowed_kwargs[algorithm]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                 (key, allowed_kwargs[algorithm]))
        # Set kwarg defaults.
        for kw in allowed_kwargs[algorithm]:
            kwargs.setdefault(kw, kwargs_defaults[kw])
    else:
        raise ValueError('Keyword "algorithm" must be String.')

    # Initialize tomography data.
    tomo = _init_tomo(tomo, emission)

    # Generate args for the algorithm.
    args = _get_algorithm_args(tomo.shape, theta, center)

    # Initialize reconstruction.
    recon = _init_recon(
        (tomo.shape[1], kwargs['num_gridx'], kwargs['num_gridy']),
        init_recon)
    return _do_recon(
        tomo, recon, _get_func(algorithm,hardware), args, kwargs)


def _init_tomo(tomo, emission):
    tomo = dtype.as_float32(tomo)
    if not emission:
        tomo = -np.log(tomo)
    return tomo


def _init_recon(shape, init_recon, val=1e-6):
    if init_recon is None:
        recon = val * np.ones(shape, dtype='float32')
    else:
        recon = dtype.as_float32(init_recon)
    return recon


def _get_func(algorithm,hardware):
    if hardware == None :
        hardware = _get_hardware()
        print ('Hardware %s is chosen by default. ' %hardware)

    if hardware == 'Xeon_Phi' :
        if algorithm == 'ospml_hybrid':
            func = xeon_phi.c_ospml_hybrid
        elif algorithm == 'ospml_quad':
            func = xeon_phi.c_ospml_quad
        elif algorithm == 'pml_hybrid':
            func = xeon_phi.c_pml_hybrid
        elif algorithm == 'pml_quad':
            func = xeon_phi.c_pml_quad
        else:
            raise ValueError('Algorithm %s not supported yet!' % (algorithm))
    else :
        raise ValueError('Hardware %s not supported yet!' % (hardware))
    return func

def _get_hardware() :
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_phi.dll'))
    libpath1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_phi.so'))
    if os.path.isfile(libpath) or os.path.isfile(libpath1) :
        return 'Xeon_Phi';
    libpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_cuda.dll'))
    libpath1 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib/libtomoperi_cuda.so'))
    if os.path.isfile(libpath) or os.path.isfile(libpath1) :
        return 'Nvidia_GPU';

def _do_recon(tomo, recon, func, args, kwargs):
    # Zip arguments.
    _args = []
    # Generate sorted args.
    _args.append(tomo)
    if args is not None:
        for a in args:
            _args.append(a)
    _args.append(recon)
    if kwargs is not None:
        _args.append(kwargs)
    return func(_args)


def _get_algorithm_args(shape, theta, center):
    dx, dy, dz = shape
    theta = dtype.as_float32(theta)
    center = get_center(shape, center)
    return (dx, dy, dz, center, theta)


def _get_algorithm_kwargs(shape):
    dx, dy, dz = shape
    return {
        'num_gridx': dz,
        'num_gridy': dz,
        'filter_name': np.array('shepp', dtype=(str, 16)),
        'num_iter': dtype.as_int32(1),
        'reg_par': np.ones(10, dtype='float32'),
        'num_block': dtype.as_int32(1),
        'ind_block': np.arange(0, dx, dtype='float32'),
        'options': {},
    }

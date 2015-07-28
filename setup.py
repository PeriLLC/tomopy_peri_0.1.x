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

#from distutils.core import setup, Extension
from setuptools import setup, Extension, find_packages

from distutils.command.build_ext import build_ext
from distutils.command.clean import clean
import os
import platform
import re
import subprocess
import sys
import zlib


VERSION = '0.1.0' 

NVIDIA_INC_DIRS = []
NVCC = 'nvcc'
CUDALIB = '/usr/cuda/lib64'

INTEL_INC_DIRS = []
ICPC = 'icpc'
ICPCLIB = '/opt/intel/composerxe/lib/intel64'

cuda_found = False
icpc_found = False

for path in ('/usr/local/cuda', '/opt/cuda'):
    if os.path.exists(path):
        NVIDIA_INC_DIRS.append(os.path.join(path, 'include'))
        NVCC = os.path.join(path, 'bin', 'nvcc')
        CUDALIB = os.path.join(path, 'lib64')
        cuda_found = True
        break

for path in ('/opt/local/intel/composerxe', '/opt/intel/composerxe'):
    if os.path.exists(path):
        INTEL_INC_DIRS.append(os.path.join(path, 'include'))
        ICPC = os.path.join(path, 'bin', 'icpc')
        ICPCLIB = os.path.join(path, 'lib/intel64')
        icpc_found = True
        break

if not (cuda_found or icpc_found):
    raise ValueError( "The CUDA and ICPC compiler and headers required to build " \
                       "kernel were not found. Cannot build library..." )

CUDA_EXTRA_COMPILE_ARGS = ['-Wall', '-fno-strict-aliasing', \
                      '-DVERSION="%s"' % (VERSION,)]

ICPC_EXTRA_COMPILE_ARGS = ['-ldl', '-shared']

class GPUBuilder(build_ext):

    def _call(self, comm):
        p = subprocess.Popen(comm, stdout=subprocess.PIPE, shell=True)
        stdo, stde = p.communicate()
        if p.returncode == 0:
            return stdo
        else:
            print >>sys.stderr, "%s\nFailed to execute command '%s'" % \
                                (stde, comm)
            return None

    def _makedirs(self, pathname):
        try:
            os.makedirs(pathname)
        except OSError:
            pass

    def run(self):
        nvcc_o = self._call(NVCC + ' -V')
        if nvcc_o is not None:
            nvcc_version = nvcc_o.split('release ')[-1].strip()
        else:
            raise SystemError("Nvidia's CUDA-compiler 'nvcc' can't be " \
                          "found.")
        print "Compiling CUDA module using nvcc %s..." % nvcc_version

        bits, linkage = platform.architecture()
        if bits == '32bit':
            bit_flag = ' -m32'
        elif bits == '64bit':
            bit_flag = ' -m64'
        else:
            print >>sys.stderr, "Can't detect platform, using 32bit"
            bit_flag = ' -m32'

        nvcc_cmd = NVCC + bit_flag + ' -c -arch=sm_20 '\
                                 ' ./src/pml_cuda_kernel.cu' \
                                 ' --compiler-options ''-fPIC'''
        print "Executing '%s'" % nvcc_cmd
        subprocess.check_call(nvcc_cmd, shell=True)

        print "Building modules..."
        build_ext.run(self)


class GPUCleaner(clean):

    def _unlink(self, node):
        try:
            if os.path.isdir(node):
                os.rmdir(node)
            else:
                os.unlink(node)
        except OSError:
            pass

    def run(self):
        print "Removing temporary files and pre-built GPU-kernels..."
        try:
            for f in ('pml_cuda_kernel.o'):
                self._unlink(f)
        except Exception, (errno, sterrno):
            print >>sys.stderr, "Exception while cleaning temporary " \
                                "files ('%s')" % sterrno
        clean.run(self)

class PHIBuilder(build_ext):

    def _call(self, comm):
        p = subprocess.Popen(comm, stdout=subprocess.PIPE, shell=True)
        stdo, stde = p.communicate()
        if p.returncode == 0:
            return stdo
        else:
            print >>sys.stderr, "%s\nFailed to execute command '%s'" % \
                                (stde, comm)
            return None

    def _makedirs(self, pathname):
        try:
            os.makedirs(pathname)
        except OSError:
            pass

    def run(self):
        icpc_o = self._call(ICPC + ' -V')
        if icpc_o is not None:
            icpc_version = icpc_o.split('Version ')[-1].strip()
        else:
            raise SystemError("Intel's phi-compiler 'icpc' can't be " \
                          "found.")
        print "Compiling ICPC module using icpc %s..." % icpc_version

        icpc_cmd = ICPC + ' -c -O3 -fPIC src/xeon_phi/utils_cilk.cpp' \
                                 ' src/xeon_phi/pml_quad_cilk.cpp src/xeon_phi/pml_hybrid_cilk.cpp' \
                                 ' src/xeon_phi/ospml_quad_cilk.cpp src/xeon_phi/ospml_hybrid_cilk.cpp' 

        print "Executing '%s'" % icpc_cmd
        subprocess.check_call(icpc_cmd, shell=True)

        print "Building modules..."
        build_ext.run(self)


class PHICleaner(clean):

    def _unlink(self, node):
        try:
            if os.path.isdir(node):
                os.rmdir(node)
            else:
                os.unlink(node)
        except OSError:
            pass

    def run(self):
        print "Removing temporary files and pre-built GPU-kernels..."
        try:
            for f in (['utils_cilk.o', 'pml_quad_cilk.o', 'pml_hybrid_cilk.o', 'ospml_quad_cilk.o', 'ospml_hybrid_cilk.o']):
                self._unlink(f)
        except Exception, (errno, sterrno):
            print >>sys.stderr, "Exception while cleaning temporary " \
                                "files ('%s')" % sterrno
        clean.run(self)


cuda_extension = Extension('lib.pml_cuda',
                    libraries = ['cuda','cudart', 'z'],
                    extra_objects = ['pml_cuda_kernel.o'],
                    sources = ['./src/pml_cuda.c'],
                    include_dirs = NVIDIA_INC_DIRS,
                    library_dirs =[CUDALIB],
                    extra_compile_args = CUDA_EXTRA_COMPILE_ARGS)

icpc_extension = Extension('lib.recon_phi',
                    libraries = ['iomp5'],
                    extra_objects = ['utils_cilk.o', 'pml_quad_cilk.o', 'pml_hybrid_cilk.o', 'ospml_quad_cilk.o', 'ospml_hybrid_cilk.o'],
                    sources = ['./src/xeon_phi/recon.cpp'],
                    include_dirs = INTEL_INC_DIRS,
                    library_dirs =[ICPCLIB],
                    extra_compile_args = ICPC_EXTRA_COMPILE_ARGS)

setup_cuda_args = dict(
        name = 'tomopy_peri',
        version = VERSION,
        packages=find_packages(),
        include_package_data=True,
        description = 'hardware accelerated tomopy algorithms',
        long_description = \
            "Hardware accelleration for tomopy reconstruction algorithms. " \
            "The first version on pml algorithm.",
        license = 'GNU Less General Public License v3',
        author = 'Dake Feng',
        author_email = 'dakefeng@gmail.com',
        url = 'http://www.perillc.com',
        classifiers = \
              ['Development Status :: 4 - Beta',
               'Environment :: Console',
               'License :: OSI Approved :: GNU Less General Public License (LGPL)',
               'Natural Language :: English',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Topic :: Security'],
        platforms = ['any'],
        ext_modules = [icpc_extension],
        cmdclass = {'build_ext': PHIBuilder, 'clean': PHICleaner},
        options = {'install': {'optimize': 1}, \
                    'bdist_rpm': {'requires': 'tomopy = 0.1.11'}})

if __name__ == "__main__":
    setup(**setup_cuda_args)


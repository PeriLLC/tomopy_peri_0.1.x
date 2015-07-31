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

import os
from sys import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.install import install
from distutils.command.build import build
from subprocess import call

VERSION = '0.1.1' 



BASEPATH = os.path.dirname(os.path.abspath(__file__))
TOMOPERI_PATH = os.path.join(BASEPATH, 'src/xeon_phi')


class TomoPeriBuild(build):
    def run(self):
        # run original build code
        build.run(self)

        # build TomoPeri
        build_path = os.path.abspath(self.build_temp)

        cmd = [
            'make',
            'OUT=' + build_path,
            'V=' + str(self.verbose),
        ]

        options = [
            'DEBUG=n',
            'ENABLE_SDL=n',
        ]
        cmd.extend(options)

        targets = ['test']
        cmd.extend(targets)

        target_files = [os.path.join(build_path, 'libtomoperi_phi.so'),os.path.join(build_path, 'tomoperi_phi_test.exe')]


        print cmd

        def compile():
            call(cmd, cwd=TOMOPERI_PATH)

        self.execute(compile, [], 'Compiling Tomopy_Peri on Phi')

        # copy resulting tool to library build folder
        self.mkpath(self.build_lib)

        if not self.dry_run:
            for target in target_files:
                self.copy_file(target, self.build_lib)


class TomoPeriInstall(install):
    def initialize_options(self):
        install.initialize_options(self)
        self.build_scripts = None

    def finalize_options(self):
        install.finalize_options(self)
        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))

    def run(self):
        # run original install code
        install.run(self)

        # install TomoPeri libraries
        self.copy_tree(self.build_lib, self.install_lib)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup_args = dict(
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
        cmdclass={
            'build': TomoPeriBuild,
            'install': TomoPeriInstall,
        },
        options = {'install': {'optimize': 1}, \
                    'bdist_rpm': {'requires': 'tomopy = 0.1.11'}})


if __name__ == "__main__":
    setup(**setup_args)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:39:27 2018

@author: cs
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('cSPMV.pyx'))
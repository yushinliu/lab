from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

#setup(name='density_func',ext_modules=cythonize('density_func.pyx'),)

ext_modules=[ Extension("density_func",["density_func.pyx"],libraries=["m"],extra_compile_args=["-ffast-math"])]

setup(name = 'density_func',cmdclass={"build_ext":build_ext},ext_modules = ext_modules)
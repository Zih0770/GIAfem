#!/usr/bin/env python3
import os

petsc_install=os.path.join(os.getenv('HOME'),'petsc-install')

configure_options = [
  '--prefix='+petsc_install,
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-single-library=0',
  '--download-mpich=1',
  '--download-fblaslapack=1',
  '--download-hypre=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-sundials2=1',
  '--download-strumpack=1',
  '--download-ginkgo=1',
  '--download-suitesparse=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-parms=1',
  '--download-netcdf=1',
  '--download-hdf5=1',
  '--download-adios=1',
  '--download-gslib-1',
  '--download-zlib=1',
  '--download-mfem=1',
  '--download-glvis=1',
  '--with-opengl=1',
  '--download-szlib=1',
  '--with-strict-petscerrorcode',
  '--with-coverage',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)

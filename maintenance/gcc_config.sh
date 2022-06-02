#!/bin/bash

rm -f CMakeCache.txt
rm -fR CMakeFiles

EXTRA_ARGS=$@

cmake .. \
          -D CMAKE_BUILD_TYPE:STRING=RELEASE \
	  -D CMAKE_C_COMPILER:STRING="mpicc" \
	  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
	  -D Python_EXECUTABLE:STRING="python3" \
          -D Trilinos_ENABLE_STK:BOOL=ON \
          -D Trilinos_ENABLE_NOX:BOOL=ON \
		  -D Trilinos_ENABLE_SEACAS:BOOL=ON \
		  -D TPL_ENABLE_Matio=OFF \
		  -D Trilinos_ENABLE_Pamgen:BOOL=ON \
          -D Trilinos_ENABLE_Panzer:BOOL=ON \
	  -D Tempus_ENABLE_TESTS=OFF \
	  -D Thyra_ENABLE_TESTS=OFF \
	  -D Panzer_ENABLE_TESTS:BOOL=ON \
	  -D Panzer_ENABLE_EXAMPLES:BOOL=ON \
	  -D Trilinos_ENABLE_ROL:BOOL=ON \
	  -D Trilinos_ENABLE_Galeri:BOOL=OFF \
	  -D Trilinos_ENABLE_Stokhos:BOOL=OFF \
          -D Trilinos_DUMP_PACKAGE_DEPENDENCIES:BOOL=ON \
          -DTrilinos_ENABLE_Fortran:BOOL=OFF \
          -D TPL_ENABLE_MPI:BOOL=ON \
	  -DTPL_ENABLE_Boost:BOOL=OFF \
          -DTPL_ENABLE_HDF5=ON \
	  -DTPL_ENABLE_Netcdf=ON \
          -DTrilinos_ENABLE_Kokkos:BOOL=ON \
          -DKokkos_ENABLE_OPENMP=ON  \
          -DTrilinos_ENABLE_OpenMP=ON \
          -D BUILD_SHARED_LIBS:BOOL=OFF \
          -D CMAKE_INSTALL_PREFIX:PATH="$HOME/myprograms/Trilinos/install" \
          -D DART_TESTING_TIMEOUT:STRING=300 \
          -D CMAKE_VERBOSE_MAKEFILE:BOOL=FALSE       \
          -D Trilinos_VERBOSE_CONFIGURE:BOOL=FALSE   \
          ${EXTRA_ARGS} \
          ${TRILINOS_PATH}

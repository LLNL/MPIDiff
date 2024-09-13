##############################################################################
# Copyright (c) 2024, Lawrence Livermore National Security, LLC and MPIDiff
# project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(COMPILER_BASE "/usr/tce/packages/intel/intel-2022.1.0-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/icx" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/icpx" CACHE PATH "")

set(GCC_HOME "/usr/tce/packages/gcc/gcc-12.1.1-magic" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")

set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_BASE "/usr/tce/packages/mvapich2/mvapich2-2.3.7-intel-2022.1.0-magic" CACHE PATH "")
set(MPI_C_COMPILER "${MPI_BASE}/bin/mpicc" CACHE PATH "")
set(MPI_CXX_COMPILER "${MPI_BASE}/bin/mpicxx" CACHE PATH "")

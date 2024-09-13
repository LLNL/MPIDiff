##########################################################################
# Copyright (c) 2024, Lawrence Livermore National Security, LLC and
# MPIDiff project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##########################################################################

# Set up software versions
set(ROCM_VERSION "6.2.0" CACHE PATH "")
set(MPICH_VERSION "8.1.30" CACHE PATH "")
set(GCC_VERSION "12.2.1" CACHE PATH "")

# Set up compilers
set(COMPILER_BASE "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VERSION}-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/amdclang++" CACHE PATH "")

# Set up compiler flags
set(GCC_HOME "/usr/tce/packages/gcc/gcc-${GCC_VERSION}" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")

# Set up HIP
set(ENABLE_HIP ON CACHE BOOL "")
set(ROCM_PATH "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VERSION}-magic" CACHE PATH "")
set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "")
set(AMDGPU_TARGETS "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "")

# Set up MPI
set(ENABLE_MPI ON CACHE BOOL "")
set(MPI_BASE "/usr/tce/packages/cray-mpich-tce/cray-mpich-${MPICH_VERSION}-rocmcc-${ROCM_VERSION}" CACHE PATH "")
set(MPI_C_COMPILER "${MPI_BASE}/bin/mpiamdclang"   CACHE PATH "")
set(MPI_CXX_COMPILER "${MPI_BASE}/bin/mpiamdclang++" CACHE PATH "")

##########################################################################
# Copyright (c) 2024, Lawrence Livermore National Security, LLC and
# MPIDiff project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##########################################################################

include(CMakeDependentOption)

set(ENABLE_MPI ON CACHE BOOL "")

if(NOT ENABLE_MPI)
   message(FATAL_ERROR "MPIDiff requires ENABLE_MPI.")
endif()

cmake_dependent_option(MPIDIFF_ENABLE_TESTS "Build tests" ON "ENABLE_TESTS" OFF)
cmake_dependent_option(MPIDIFF_ENABLE_DOCS "Build documentation" ON "ENABLE_DOCS" OFF)
cmake_dependent_option(MPIDIFF_ENABLE_EXAMPLES "Build examples" ON "ENABLE_EXAMPLES" OFF)

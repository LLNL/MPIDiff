######################################################################################
# Copyright 2024 Lawrence Livermore National Security, LLC and other MPIDiff developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

include(CMakeDependentOption)

cmake_dependent_option(MPIDIFF_ENABLE_TESTS "Build tests" ON "ENABLE_TESTS" OFF)
cmake_dependent_option(MPIDIFF_ENABLE_DOCS "Build documentation" ON "ENABLE_DOCS" OFF)
cmake_dependent_option(MPIDIFF_ENABLE_EXAMPLES "Build examples" ON "ENABLE_EXAMPLES" OFF)

if(NOT ENABLE_MPI)
   message(FATAL_ERROR "MPIDiff requires ENABLE_MPI.")
endif()

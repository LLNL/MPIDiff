##########################################################################
# Copyright (c) 2019-24, Lawrence Livermore National Security, LLC and
# MPIDiff project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##########################################################################

##########################################################################
# Set up CMake
##########################################################################

cmake_minimum_required(VERSION 3.14)

##########################################################################
# Set up project
##########################################################################

project(MPIDiff
        LANGUAGES C CXX
        VERSION 0.2.1)

include(${PROJECT_SOURCE_DIR}/cmake/Setup.cmake)

################################
# Set up library
################################

add_subdirectory(src)

##########################################################################
# Set up tests
##########################################################################

if(MPIDIFF_ENABLE_TESTS)
   add_subdirectory(test)
endif()

##########################################################################
# Set up documentation
##########################################################################

if(MPIDIFF_ENABLE_DOCS)
   add_subdirectory(docs)
endif()

##########################################################################
# Set up examples
##########################################################################

if(MPIDIFF_ENABLE_EXAMPLES)
   add_subdirectory(examples)
endif()

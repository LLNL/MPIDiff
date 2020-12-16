######################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

cmake_minimum_required(VERSION 3.4)

project(mpidiff)

################################
# Options
################################
option(ENABLE_TESTS "Enable MPIDiff tests" ON)
option(ENABLE_DOCS "Enable MPIDiff documentation" ON)
option(ENABLE_EXAMPLES "Enable MPIDiff examples" ON)

################################
# Set up variables and flags
################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

################################
# BLT
################################
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
   message(FATAL_ERROR "BLT_SOURCE_DIR not defined")
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

################################
# Build source
################################
add_subdirectory(src)

################################
# Build tests
################################
if (ENABLE_TESTS)
   add_subdirectory(test)
endif()

################################
# Build documentation
################################
if (ENABLE_DOCS)
   add_subdirectory(docs)
endif()

################################
# Build examples
################################
if (ENABLE_EXAMPLES)
   add_subdirectory(examples)
endif()


##########################################################################
# Copyright (c) 2019-24, Lawrence Livermore National Security, LLC and
# MPIDiff project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##########################################################################

set(mpidiff_depends
    mpidiff
    mpi)

blt_add_executable(NAME MPIDiff_Program1
                   SOURCES MPIDiff_Program1.cpp
                   DEPENDS_ON ${mpidiff_depends})

target_include_directories(MPIDiff_Program1 PRIVATE ${PROJECT_SOURCE_DIR}/src)

blt_add_executable(NAME MPIDiff_Program2
                   SOURCES MPIDiff_Program2.cpp
                   DEPENDS_ON ${mpidiff_depends})

target_include_directories(MPIDiff_Program2 PRIVATE ${PROJECT_SOURCE_DIR}/src)


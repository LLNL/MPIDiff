######################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

set (mpidiff_test_dependencies gtest mpidiff mpi)

blt_add_executable( NAME TestMPIDiff
                    SOURCES TestMPIDiff.cpp
                    DEPENDS_ON ${mpidiff_test_dependencies} )

target_include_directories(TestMPIDiff
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestMPIDiff
                           PRIVATE ${PROJECT_SOURCE_DIR}/tpl)

blt_add_test( NAME TestMPIDiff
              COMMAND TestMPIDiff )

blt_add_executable( NAME TestSynchronous
                    SOURCES TestSynchronous.cpp
                    DEPENDS_ON ${mpidiff_test_dependencies} )

target_include_directories(TestSynchronous
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestSynchronous
                           PRIVATE ${PROJECT_SOURCE_DIR}/tpl)

blt_add_test( NAME TestSynchronous
              COMMAND TestSynchronous )

######################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

set(mpidiff_sources
    MPIDiff.cpp)

set(mpidiff_headers
    MPIDiff.h)

set(mpidiff_depends
    mpi)

blt_add_library(NAME mpidiff
                SOURCES ${mpidiff_sources}
                HEADERS ${mpidiff_headers}
                DEPENDS_ON ${mpidiff_depends})

target_include_directories(mpidiff PRIVATE ${PROJECT_SOURCE_DIR}/src)

install(TARGETS mpidiff
        EXPORT ${PROJECT_NAME}-targets
        DESTINATION lib
        INCLUDES DESTINATION include/mpidiff)

install(FILES ${mpidiff_headers}
        DESTINATION include/mpidiff)

install(EXPORT ${PROJECT_NAME}-targets DESTINATION share/mpidiff/cmake)

configure_file(${PROJECT_SOURCE_DIR}/src/mpidiff/mpidiff-config.cmake.in
               ${PROJECT_BINARY_DIR}/share/mpidiff/cmake/mpidiff-config.cmake)

install(FILES ${PROJECT_BINARY_DIR}/share/mpidiff/cmake/mpidiff-config.cmake
        DESTINATION share/mpidiff/cmake)

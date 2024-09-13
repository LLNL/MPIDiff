##########################################################################
# Copyright (c) 2019-24, Lawrence Livermore National Security, LLC and
# MPIDiff project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##########################################################################

set(mpidiff_sources
    MPIDiff.cpp)

set(mpidiff_headers
    MPIDiff.h)

set(mpidiff_includes
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)

set(mpidiff_depends
    mpi)

blt_add_library(NAME mpidiff
                SOURCES ${mpidiff_sources}
                HEADERS ${mpidiff_headers}
                INCLUDES ${mpidiff_includes}
                DEPENDS_ON ${mpidiff_depends})

install(FILES ${mpidiff_headers}
        DESTINATION include/mpidiff)

install(TARGETS mpidiff
        EXPORT  mpidiff-targets
        DESTINATION lib)

install(EXPORT mpidiff-targets
        NAMESPACE mpidiff::
        DESTINATION lib/cmake/mpidiff)

include(CMakePackageConfigHelpers)

configure_package_config_file(
   ${PROJECT_SOURCE_DIR}/cmake/mpidiff-config.cmake.in
   ${PROJECT_BINARY_DIR}/cmake/mpidiff-config.cmake
   INSTALL_DESTINATION lib/cmake/mpidiff)

write_basic_package_version_file(
   ${PROJECT_BINARY_DIR}/cmake/mpidiff-config-version.cmake
   COMPATIBILITY SameMajorVersion)

install(
   FILES
      ${PROJECT_BINARY_DIR}/cmake/mpidiff-config.cmake
      ${PROJECT_BINARY_DIR}/cmake/mpidiff-config-version.cmake
   DESTINATION
      lib/cmake/mpidiff)

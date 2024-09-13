######################################################################################
# Copyright 2024 Lawrence Livermore National Security, LLC and other MPIDiff developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

if(NOT BLT_LOADED)
   if(DEFINED BLT_SOURCE_DIR)
      if(EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
         include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
      else()
         message(FATAL_ERROR
                "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake!")
      endif()
   else()
      message(FATAL_ERROR
              "BLT is required!"
              "Add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command.")
   endif()
endif()

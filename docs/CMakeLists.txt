##########################################################################
# Copyright (c) 2019-24, Lawrence Livermore National Security, LLC and
# MPIDiff project contributors. See the MPIDiff LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##########################################################################

if (DOXYGEN_FOUND)
   add_subdirectory(doxygen)
else()
   message(WARNING "ENABLE_DOCS=On, but Doxygen is not found. \
                    Documentation will not be built.")
endif()

if (SPHINX_FOUND)
   add_subdirectory(sphinx)
endif()


######################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

if (DOXYGEN_FOUND)
   add_subdirectory(doxygen)
else()
   message(WARNING "ENABLE_DOCS=On, but Doxygen is not found. \
                    Documentation will not be built.")
endif()

if (SPHINX_FOUND)
   add_subdirectory(sphinx)
endif()


.. ######################################################################################
   # Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
   # See the top-level LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ######################################################################################

###############
Install MPIDiff
###############

Build Requirements
==================

* BLT (https://github.com/LLNL/blt)
* CMake (https://cmake.org)
* Make (https://www.gnu.org/software/make/)

Dependencies
============

* MPI

Instructions
============

.. code-block:: bash

   git clone ssh://git@rz-bitbucket.llnl.gov:7999/mpidiff/mpidiff.git
   cd mpidiff
   mkdir build
   cd build
   cmake -DENABLE_MPI=ON -DBLT_SOURCE_DIR=<path/to/blt> ../

Configuration Options
=====================

+-----------------+------------------------------+---------+
| Option          | Description                  | Default |
+=================+==============================+=========+
| ENABLE_TESTS    | Enable MPIDiff tests         | ON      |
+-----------------+------------------------------+---------+
| ENABLE_DOC      | Enable MPIDiff documentation | ON      |
+-----------------+------------------------------+---------+
| ENABLE_EXAMPLES | Enable MPIDiff examples      | ON      |
+-----------------+------------------------------+---------+


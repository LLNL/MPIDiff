.. ######################################################################################
   # Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
   # See the top-level LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ######################################################################################

#######################
MPIDiff Advanced Topics
#######################

MPIDiff is completely asynchronous. The advantage to this is that it provides extreme flexibility for executables that do not always follow the same code path. The disadvantage is that messages must be stored until communication with matching tags is found. To combat this growth in memory usage, MPIDiff::Barrier is provided. The barrier allows both executables to catch up with communication. The method also takes an argument that tells MPIDiff whether or not to flush any messages that do not have a match at the time of a barrier. For codes that do a lot of work inside a main loop, this is ideal.

.. code-block:: c++

   static void MPIDiff::Barrier(bool flush=false);


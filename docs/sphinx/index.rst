.. ######################################################################################
   # Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
   # See the top-level LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ######################################################################################

##################
MPIDiff User Guide
##################

Many libraries, such as RAJA and CHAI, provide powerful abstractions for writing portable code. While the same code is used regardless of the backend, what happens under the hood can be quite different depending on the configuration. It is relatively easy to debug CPU only code, but quite difficult to debug GPU code. MPIDiff is a C++ library for pairing a correct version of an executable with a version that may have bugs or race conditions in it. They communicate through MPI, and the correct version checks that the data communicated from the other program matches its own.


.. toctree::
   :maxdepth: 2

   install
   tutorial
   advanced


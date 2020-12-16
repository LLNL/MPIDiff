<!--
######################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################
-->

# MPIDiff

Many libraries, such as RAJA and CHAI, provide powerful abstractions for writing portable code. While the same code is used regardless of the backend, what happens under the hood can be quite different depending on the configuration. It is relatively easy to debug CPU only code, but quite difficult to debug GPU code. MPIDiff is a C++ library for pairing a correct version of an executable with a version that may have bugs or race conditions in it. They communicate through MPI, and the correct version checks that the data communicated from the other program matches its own.

## Build Requirements

* CMake
* BLT (https://github.com/LLNL/blt)
* Make

## Dependencies

* MPI
* gtest-mpi-listener (for test cases only)

## Installation

```bash
git clone ssh://git@rz-bitbucket.llnl.gov:7999/mpidiff/mpidiff.git
cd mpidiff
mkdir build && cd build
cmake -DENABLE_MPI=ON -DBLT_SOURCE_DIR=<path/to/blt> ../
```

## Usage

Write a file that specifies the executables to run along with arguments.

```bash
cat taskfile.txt
0-1 bin/TestMultiProgramDebug -multiprogid 1
2-3 bin/TestMultiProgramDebug -multiprogid 2
```

Pass the file as the argument to --multi-prog for srun.

```bash
srun --multi-prog taskfile.txt
```

## Release

MPIDiff is release under the BSD-3-Clause License. See the LICENSE and NOTICE files for more details. All new contributions must be adhere to this license.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-817465

.. ######################################################################################
   # Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
   # See the top-level LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ######################################################################################

################
MPIDiff Tutorial
################

MPIDiff provides a convenient API for communicating data between two running programs and checking one program against the other for correctness. It uses MPI under the hood but can easily be applied to serial programs.

Like MPI, MPIDiff must be initialized at the beginning of the program and finalized at the end of the program. It depends on MPI being initialized, so all MPIDiff calls should be after MPI_Init() and before MPI_Finalize() are called.

.. code-block:: c++

    #include <mpidiff/MPIDiff.h>

    int main(int argc, char* argv[]) {
       MPI_Init(argc, argv);
       MPIDiff::Init(MPI_COMM_WORLD, 0);

       // Do stuff

       MPIDiff::Finalize();
       MPI_Finalize();
    }

MPIDiff::Init takes two arguments. The first is a communicator over both programs. For simple programs, this may be MPI_COMM_WORLD. For more complex programs, the flexibility is there to provide a different communicator. The second argument is a program ID. If I have two programs, call them A and B, I would give A the program ID 0 and B the program ID 1. One way to give them varying program IDs is to make the program ID a command line option.

.. code-block:: c++

    #include <mpidiff/MPIDiff.h>

    int main(int argc, char* argv[]) {
       // Parse input argument to get the ID of this program
       int programID = -1;

       for (int i = 1; i < argc; ++i) {
          if (strcmp(argv[i], "-programID") == 0) {
             programID = atoi(argv[i+1]);
          }
       }

       MPI_Init(argc, argv);
       MPIDiff::Init(MPI_COMM_WORLD, programID);

       ...

To run multiple instances of this program with a shared communicator and different command line arguments is simple.

.. code-block:: bash

   cat taskfile.txt
   -np 1 A.out -programID 0
   -np 1 B.out -programID 1

   mpirun --app=taskfile.txt

If you are using a different scheduler, the syntax is not much different. For instance, srun can be used to launch multiple executables with a shared communicator in the following manner.

.. code-block:: bash

   cat taskfile.txt
   0 A.out -programID 0
   1 B.out -programID 1

   srun --multi-prog taskfile.txt

The jsrun scheduler can also be used, but is a little more complicated because it requires a reservation to be made.

.. code-block:: bash

   cat taskfile.txt
   1 : res1 : A.out -programID 0
   1 : res1 : B.out -programID 1

   lalloc 1
   jsrun -n 2 -A res1
   jsrun --appfile=taskfile
 
Once the programs can be run in this manner, proceed to add communication between them.

.. code-block:: c++

   // Set up data
   int length = 100;
   int* myArray = new int[length];

   for (int i = 0; i < length; ++i) {
      myArray[i] = i;
   }

   // Check data
   MPIDiff::Reduce(myArray, length, MPI_INT, 0);

   // Clean up data
   delete[] myArray;

The final result is the following:

.. code-block:: c++

    #include <mpidiff/MPIDiff.h>

    int main(int argc, char* argv[]) {
       // Parse input argument to get the ID of this program
       int programID = -1;

       for (int i = 1; i < argc; ++i) {
          if (strcmp(argv[i], "-programID") == 0) {
             programID = atoi(argv[i+1]);
          }
       }

       MPI_Init(argc, argv);
       MPIDiff::Init(MPI_COMM_WORLD, programID);

       // Set up data
       int length = 100;
       int* myArray = new int[length];

       for (int i = 0; i < length; ++i) {
          myArray[i] = i;
       }

       // Check data
       MPIDiff::Reduce(myArray, length, MPI_INT, 0);

       // Clean up data
       delete[] myArray;

       MPIDiff::Finalize();
       MPI_Finalize();
    }

Look in the example folder for two programs that are slightly different (a "bug" has been injected into one), as well as a taskfile to run them and observe the output. These examples are contrived, but one could imagine a race condition in a parallel program that could be caught by hooking up a serial and a parallel version of the same executable.

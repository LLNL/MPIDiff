//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// Standard library headers
#include <cstring>
#include <stdio.h>
#include <unordered_map>
#include <vector>

// Third party library headers
#include <mpi.h>

// MPIDiff headers
#include "mpidiff/MPIDiff.h"

static std::vector<int> failingTags;
static std::unordered_map<int, std::string> tagToInfo;

void custom_handler(MPI_Datatype datatype, int tag,
                    const void* buffer1, int count,
                    const void* buffer2, int bytes) {
   int typeSize;
   MPI_ERROR_CHECK(MPI_Type_size(datatype, &typeSize));

   if (count * typeSize != bytes) {
      failingTags.push_back(tag);

      auto record = tagToInfo.find(tag);

      if (record != tagToInfo.end()) {
         printf("Incorrect array sizes at: %s\n", record->second.c_str());
      }

      return;
   }
   else if (datatype == MPI_INT) {
      int* int_buffer1 = (int*) buffer1;
      int* int_buffer2 = (int*) buffer2;

      for (int i = 0; i < count; ++i) {
         if (int_buffer1[i] != int_buffer2[i]) {
            failingTags.push_back(tag);

            auto record = tagToInfo.find(tag);

            if (record != tagToInfo.end()) {
               printf("array1[%d] = %d != array2[%d] = %d at: %s\n", i, int_buffer1[i], i, int_buffer2[i], record->second.c_str());
            }

            return;
         }
      }
   }
   else if (datatype == MPI_DOUBLE) {
      double* double_buffer1 = (double*) buffer1;
      double* double_buffer2 = (double*) buffer2;

      for (int i = 0; i < count; ++i) {
         if (double_buffer2[i] - double_buffer1[i] > 1e-12) {
            failingTags.push_back(tag);
            return;
         }
      }
   }

   return;
}

int main(int argc, char ** argv) {
   // Initialize MPI
   MPI_Init(NULL, NULL);

   // Parse input argument to get the ID of this program
   int multiprogid = -1;

   for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "-multiprogid") == 0) {
         multiprogid = atoi(argv[i+1]);
      }
   }

   // Initialize MPIDiff
   MPIDiff::Init(MPI_COMM_WORLD, multiprogid);
   MPIDiff::Set_handler(custom_handler);

   // Set up data
   int length = 100;
   int* myArray = new int[length];

   for (int i = 0; i < length; ++i) {
      myArray[i] = i;
   }

   // Check data
   std::string name = "myArray";
   std::size_t id = std::hash<std::string>{}(name) % MPIDiff::Get_max_debug_tag();
   tagToInfo.insert({id, name});
   MPIDiff::Reduce(myArray, length, MPI_INT, id);

   // Clean up data
   delete[] myArray;

   // Make sure all checks are complete
   MPIDiff::Barrier();
   MPIDiff::Finalize();

   // Finalize MPI
   MPI_Finalize();

   // Return
   return 0;
}


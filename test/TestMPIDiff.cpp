//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// std library headers
#include <cstdlib>
#include <limits.h>
#include <time.h>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thread>

// other library headers
#include "gtest/gtest.h"
#include "gtest-mpi-listener/gtest-mpi-listener.hpp"

// ravi headers
#include "mpidiff/MPIDiff.h"

int get_my_rank(MPI_Comm communicator) {
   int rank;
   MPI_ERROR_CHECK(MPI_Comm_rank(communicator, &rank));
   return rank;
}

/// The tags passed to MPIDiff where the messages were not equivalent
static std::vector<int> failingTags;
static std::unordered_map<int, std::string> tagToInfo;

/// Keeps track of the local messages that did not have a match
static std::vector<int> unreadSentTags;

/// Keeps track of the remote messages that did not have a match
static std::vector<int> unreadReceivedTags;

void custom_handler(MPI_Datatype datatype, int tag,
                    const void* buffer1, int count,
                    const void* buffer2, int bytes) {
   int typeSize;
   MPI_ERROR_CHECK(MPI_Type_size(datatype, &typeSize));

   if (count * typeSize != bytes) {
      failingTags.push_back(tag);
      return;
   }
   else if (datatype == MPI_INT) {
      int* int_buffer1 = (int*) buffer1;
      int* int_buffer2 = (int*) buffer2;

      for (int i = 0; i < count; ++i) {
         if (int_buffer1[i] != int_buffer2[i]) {
            failingTags.push_back(tag);
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

void custom_unread_handler(MPI_Datatype, int tag, bool received,
                           const void*, int) {
   if (received) {
      unreadReceivedTags.push_back(tag);
   }
   else {
      unreadSentTags.push_back(tag);
   }

   return;
}

TEST(MPIDiff, MaxTag) {
   // Initialization
   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   int* value, flag;
   MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &flag);

   ASSERT_GT(flag, 0);

   // Send data to myself using the highest tag
   MPI_Request request;
   MPI_ERROR_CHECK(MPI_Isend(testArray, length, MPI_INT, myGlobalRank, *value, MPI_COMM_WORLD, &request));

   MPI_Status status;
   int* receiveBuffer = (int*) malloc(length * sizeof(int));
   MPI_ERROR_CHECK(MPI_Recv(receiveBuffer, length, MPI_INT, myGlobalRank, *value, MPI_COMM_WORLD, &status));
   free(receiveBuffer);

   // Make sure the send has gone through
   MPI_ERROR_CHECK(MPI_Wait(&request, &status));
   free(testArray);
}

TEST(MPIDiff, Init) {
   int worldRank = get_my_rank(MPI_COMM_WORLD);

   if (worldRank == 0) {
      ASSERT_EQ(MPIDiff::Get_multi_program_rank(), 0);
      ASSERT_EQ(MPIDiff::Get_program_rank(), 0);
      ASSERT_EQ(MPIDiff::Get_debug_rank(), 0);
   }
   else if (worldRank == 1) {
      ASSERT_EQ(MPIDiff::Get_multi_program_rank(), 1);
      ASSERT_EQ(MPIDiff::Get_program_rank(), 0);
      ASSERT_EQ(MPIDiff::Get_debug_rank(), 1);
   }
   else if (worldRank == 2) {
      ASSERT_EQ(MPIDiff::Get_multi_program_rank(), 2);
      ASSERT_EQ(MPIDiff::Get_program_rank(), 1);
      ASSERT_EQ(MPIDiff::Get_debug_rank(), 0);
   }
   else if (worldRank == 3) {
      ASSERT_EQ(MPIDiff::Get_multi_program_rank(), 3);
      ASSERT_EQ(MPIDiff::Get_program_rank(), 1);
      ASSERT_EQ(MPIDiff::Get_debug_rank(), 1);
   }
}

TEST(MPIDiff, EqualIntArrays) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 1);

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      EXPECT_TRUE(failingTags.empty());
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, DifferentLengthIntArrays) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length;

   if (myGlobalRank == 0) {
      length = 100;
   }
   else {
      length = 99;
   }

   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 1);

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 1);
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, UnequalIntFirstElement) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank == 0) {
      testArray[0] = 1;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 2);

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 2);
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, UnequalIntMiddleElement) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank == 1) {
      testArray[50] = -50;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 3);

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 3);
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, UnequalIntLastElement) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank == 0) {
      testArray[99] = 98;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 4);

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 4);
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, FloatingPointInsideTolerance) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   double* testArray = (double*) malloc(length * sizeof(double));

   for (int i = 0; i < length; ++i) {
      testArray[i] = (double) i;
   }

   if (myGlobalRank == 0) {
      testArray[99] -= 1e-12;
   }

   MPIDiff::Reduce(testArray, length, MPI_DOUBLE, 12);

   free(testArray);

   MPIDiff::Barrier();

   EXPECT_TRUE(failingTags.empty());
   EXPECT_TRUE(unreadSentTags.empty());
   EXPECT_TRUE(unreadReceivedTags.empty());
}

TEST(MPIDiff, FloatingPointOutsideTolerance) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   double* testArray = (double*) malloc(length * sizeof(double));

   for (int i = 0; i < length; ++i) {
      testArray[i] = (double) i;
   }

   if (myGlobalRank == 0) {
      testArray[99] -= 1e-11;
   }

   MPIDiff::Reduce(testArray, length, MPI_DOUBLE, 12);

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 12);
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, NoInterceptWorldCommunication) {
   // Initialization
   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   // Send data to myself
   MPI_Request request;
   MPI_Isend(testArray, length, MPI_INT, myGlobalRank, 1, MPI_COMM_WORLD, &request);

   // Now test if the communication will be intercepted
   MPIDiff::Barrier();

   // Now try to receive the original message. If the message was intercepted,
   // will result in deadlock, which means the test failed.
   MPI_Status status;
   int* receiveBuffer = (int*) malloc(length * sizeof(int));
   MPI_Recv(receiveBuffer, length, MPI_INT, myGlobalRank, 1, MPI_COMM_WORLD, &status);
   free(receiveBuffer);

   // Make sure the send has gone through
   MPI_Wait(&request, &status);
   free(testArray);
}

TEST(MPIDiff, NoInterceptMultiProgramCommunication) {
   // Initialization
   int myGlobalRank = get_my_rank(MPIDiff::Get_multi_program_communicator());

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   // Send data to myself
   MPI_Request request;
   MPI_Isend(testArray, length, MPI_INT, myGlobalRank, 1,
             MPIDiff::Get_multi_program_communicator(), &request);

   // Now test if the communication will be intercepted
   MPIDiff::Barrier();

   // Now try to receive the original message. If the message was intercepted,
   // will result in deadlock, which means the test failed.
   MPI_Status status;
   int* receiveBuffer = (int*) malloc(length * sizeof(int));
   MPI_Recv(receiveBuffer, length, MPI_INT, myGlobalRank, 1,
            MPIDiff::Get_multi_program_communicator(), &status);
   free(receiveBuffer);

   // Make sure the send has gone through
   MPI_Wait(&request, &status);
   free(testArray);
}

TEST(MPIDiff, NoInterceptProgramCommunication) {
   // Initialization
   int myGlobalRank = get_my_rank(MPIDiff::Get_program_communicator());

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   // Send data to myself
   MPI_Request request;
   MPI_Isend(testArray, length, MPI_INT, myGlobalRank, 1,
             MPIDiff::Get_program_communicator(), &request);

   // Now test if the communication will be intercepted
   MPIDiff::Barrier();

   // Now try to receive the original message. If the message was intercepted,
   // will result in deadlock, which means the test failed.
   MPI_Status status;
   int* receiveBuffer = (int*) malloc(length * sizeof(int));
   MPI_Recv(receiveBuffer, length, MPI_INT, myGlobalRank, 1,
            MPIDiff::Get_program_communicator(), &status);
   free(receiveBuffer);

   // Make sure the send has gone through
   MPI_Wait(&request, &status);
   free(testArray);

}

TEST(MPIDiff, ExtraCommunicationBeforeCompare) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank == 1) {
      testArray[99] = 100;
      MPIDiff::Reduce(testArray, length, MPI_INT, 2);
      testArray[99] = 99;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 1);

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_TRUE(failingTags.empty());

   if (myGlobalRank == 0) {
      EXPECT_TRUE(unreadSentTags.empty());
      ASSERT_EQ(unreadReceivedTags.size(), 1);
      EXPECT_EQ(unreadReceivedTags[0], 2);
   }
}

TEST(MPIDiff, ExtraCommunicationBeforeCompareWithFailure) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank == 1) {
      testArray[99] = 100;
      MPIDiff::Reduce(testArray, length, MPI_INT, 2);
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 5);

   free(testArray);

   MPIDiff::Barrier(true);

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 5);
      EXPECT_TRUE(unreadSentTags.empty());
      ASSERT_EQ(unreadReceivedTags.size(), 1);
      EXPECT_EQ(unreadReceivedTags[0], 2);
   }
}

TEST(MPIDiff, ExtraCommunicationAfterCompare) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 1);

   if (myGlobalRank == 0) {
      testArray[0] = -1;
      MPIDiff::Reduce(testArray, length, MPI_INT, 2);
   }

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_EQ(failingTags.size(), 0);

   if (myGlobalRank == 0) {
      ASSERT_EQ(unreadSentTags.size(), 1);
      EXPECT_EQ(unreadSentTags[0], 2);
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, ExtraCommunicationAfterCompareWithFailure) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank == 0) {
      testArray[0] = -1;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 6);

   if (myGlobalRank == 0) {
      testArray[0] = -1;
      MPIDiff::Reduce(testArray, length, MPI_INT, 2);
   }

   free(testArray);

   MPIDiff::Barrier(true);

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 6);
      ASSERT_EQ(unreadSentTags.size(), 1);
      EXPECT_EQ(unreadSentTags[0], 2);
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

TEST(MPIDiff, DivergentCodePaths) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 1);

   if (myGlobalRank == 1) {
      testArray[50] = 60;
      MPIDiff::Reduce(testArray, length, MPI_INT, 2);
      testArray[50] = 50;
   }

   testArray[3] = 10;
   MPIDiff::Reduce(testArray, length, MPI_INT, 3);

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_EQ(failingTags.size(), 0);

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      EXPECT_TRUE(unreadSentTags.empty());
      ASSERT_EQ(unreadReceivedTags.size(), 1);
      EXPECT_EQ(unreadReceivedTags[0], 2);
   }
}

TEST(MPIDiff, DivergentCodePathsWithFailure) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank == 1) {
      testArray[2] = 7;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 7);

   if (myGlobalRank == 1) {
      testArray[50] = 60;
      MPIDiff::Reduce(testArray, length, MPI_INT, 2);
      testArray[50] = 50;
   }

   testArray[3] = 10;

   if (myGlobalRank == 1) {
      testArray[2] = 2;
   }

   MPIDiff::Reduce(testArray, length, MPI_INT, 8);

   free(testArray);

   MPIDiff::Barrier(true);

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 7);
      EXPECT_TRUE(unreadSentTags.empty());
      ASSERT_EQ(unreadReceivedTags.size(), 1);
      EXPECT_EQ(unreadReceivedTags[0], 2);
   }
}

TEST(MPIDiff, RandomDivergence)
{
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, i+1);

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Barrier(true);
   }

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_TRUE(failingTags.empty());
}

TEST(MPIDiff, RandomDivergenceWithFailures)
{
   MPIDiff::Barrier(true);
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] += 1;
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, i+1);

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] -= 1;
      }

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Barrier(true);
   }

   free(testArray);

   MPIDiff::Barrier(true);

   if (myGlobalRank == 0) {
      ASSERT_EQ(failingTags.size(), 2);
      EXPECT_EQ(failingTags[0], 5+1);
      EXPECT_EQ(failingTags[1], 777+1);
   }
}

TEST(MPIDiff, RandomDivergenceNoBarrier)
{
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, i+1);

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }
   }

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_TRUE(failingTags.empty());
}

TEST(MPIDiff, RandomDivergenceNoBarrierWithFailures)
{
   MPIDiff::Barrier(true);
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] += 1;
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, i+1);

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] -= 1;
      }

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }
   }

   free(testArray);

   MPIDiff::Barrier(true);

   if (myGlobalRank == 0) {
      ASSERT_EQ(failingTags.size(), 2);
      EXPECT_EQ(failingTags[0], 5+1);
      EXPECT_EQ(failingTags[1], 777+1);
   }
}

TEST(MPIDiff, RandomDivergenceSameTag)
{
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, 1);

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Barrier(true);
   }

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_TRUE(failingTags.empty());
}

TEST(MPIDiff, RandomDivergenceSameTagWithFailures)
{
   MPIDiff::Barrier(true);
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] += 1;
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, 9);

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] -= 1;
      }

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Barrier(true);
   }

   free(testArray);

   MPIDiff::Barrier(true);

   if (myGlobalRank == 0) {
      ASSERT_EQ(failingTags.size(), 2);
      EXPECT_EQ(failingTags[0], 9);
      EXPECT_EQ(failingTags[1], 9);
   }
}

TEST(MPIDiff, RandomDivergenceSameTagNoBarrier)
{
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, 1);

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }
   }

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_TRUE(failingTags.empty());
}

TEST(MPIDiff, RandomDivergenceSameTagNoBarrierWithFailures)
{
   MPIDiff::Barrier(true);
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] += 1;
      }

      MPIDiff::Reduce(testArray, length, MPI_INT, 9);

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] -= 1;
      }

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }
   }

   free(testArray);

   MPIDiff::Barrier(true);

   if (myGlobalRank == 0) {
      ASSERT_EQ(failingTags.size(), 2);
      EXPECT_EQ(failingTags[0], 9);
      EXPECT_EQ(failingTags[1], 9);
   }
}

TEST(MPIDiff, RandomDivergenceSameTagUsingHashNoBarrier)
{
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int* maxTag, flag;
   MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &maxTag, &flag);

   ASSERT_GT(flag, 0);

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      std::string id = std::string(__FILE__) + ":" + std::to_string(__LINE__);
      int hashValue = std::hash<std::string>{}(id) % *maxTag;
      ASSERT_NE(hashValue, myGlobalRank + 1001);
      MPIDiff::Reduce(testArray, length, MPI_INT, hashValue);

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }
   }

   free(testArray);

   MPIDiff::Barrier(true);

   EXPECT_TRUE(failingTags.empty());
}

TEST(MPIDiff, RandomDivergenceSameTagUsingHashNoBarrierWithFailures)
{
   MPIDiff::Barrier(true);
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int* maxTag, flag;
   MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &maxTag, &flag);

   ASSERT_GT(flag, 0);

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   int hashValue;

   for (int i = 0; i < 1000; ++i) {
      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 3; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] += 1;
      }

      std::string id = std::string(__FILE__) + ":" + std::to_string(__LINE__);
      hashValue = std::hash<std::string>{}(id) % *maxTag; // MPI_TAG_UB is broken
      ASSERT_NE(hashValue, myGlobalRank + 1001);
      tagToInfo.insert({hashValue, id});
      MPIDiff::Reduce(testArray, length, MPI_INT, hashValue);

      if (myGlobalRank == 0 && (i == 5 || i == 777)) {
         testArray[i % length] -= 1;
      }

      // Now change testArray
      for (int j = 0; j < length; ++j) {
         testArray[j] += 1;
      }

      if (rand() % 2 == myGlobalRank) {
         for (int j = 0; j < rand() % 6; ++j) {
            MPIDiff::Reduce(testArray, length, MPI_INT, myGlobalRank + 1001);
         }
      }
   }

   free(testArray);

   MPIDiff::Barrier(true);

   if (myGlobalRank == 0) {
      ASSERT_EQ(failingTags.size(), 2);
      EXPECT_EQ(failingTags[0], hashValue);
      EXPECT_EQ(failingTags[1], hashValue);
   }
}

int main(int argc, char** argv) {
   // Initialize random seed
   srand (time(NULL));

   // Filter out Google Test arguments
   ::testing::InitGoogleTest(&argc, argv);

   // Initialize MPI
   MPI_ERROR_CHECK(MPI_Init(&argc, &argv));

   // Initialize MPIDiff
   int myMultiProgramID = get_my_rank(MPI_COMM_WORLD) % 2;
   MPIDiff::Init(MPI_COMM_WORLD, myMultiProgramID);
   MPIDiff::Set_handler(custom_handler);
   MPIDiff::Set_unread_handler(custom_unread_handler);

   // Add object that will finalize MPI on exit; Google Test owns this pointer
   ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

   // Get the event listener list.
   ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();

   // Remove default listener: the default printer and the default XML printer
   ::testing::TestEventListener *l =
        listeners.Release(listeners.default_result_printer());

   // Adds MPI listener; Google Test owns this pointer
   listeners.Append(
      new GTestMPIListener::MPIWrapperPrinter(l,
                                              MPI_COMM_WORLD)
      );

   // Run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
   // pass and 1 if some test fails.
   int result = RUN_ALL_TESTS();
   (void) result;

   return 0;
}


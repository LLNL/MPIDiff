// std library headers
#include <cmath>
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

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Custom type used to verify custom predicates and to string
///        functors can be used.
///
/////////////////////////////////////////////////////////////////////////
class UserDefinedType {
   public:
      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Constructor
      ///
      /// @param[in] name   The name of this instance of this class
      ///
      /////////////////////////////////////////////////////////////////////////
      UserDefinedType(std::string name) : m_name(name) {}

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Returns the name of this instance of this class
      ///
      /// @return   The name of this instance of this class
      ///
      /////////////////////////////////////////////////////////////////////////
      std::string getName() const { return m_name; }

   private:
      std::string m_name; //<! The name of this instance of this class
};

/////////////////////////////////////////////////////////////////////////
///
/// @brief Helper function to get this process's rank in the given
///        communicator.
///
/// @param[in] communicator The communicator from which to obtain this
///                         process's rank.
///
/////////////////////////////////////////////////////////////////////////
int get_my_rank(MPI_Comm communicator) {
   int rank;
   MPI_ERROR_CHECK(MPI_Comm_rank(communicator, &rank));
   return rank;
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that equal int arrays have no diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, EqualIntArrays) {
   int expectedFailingTag = -1;
   int actualFailingTag = -1;

   int expectedFailingIndex = -1;
   int actualFailingIndex = -1;

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   MPIDiff::Diff(length, testArray, "testArray1",
                 [&] (const int& value1, const int& value2) {
                    static int index = 0;

                    bool equal = value1 == value2;

                    if (!equal) {
                       actualFailingTag = 1;
                       actualFailingIndex = index;
                    }

                    ++index;

                    return equal;
                 });

   if (get_my_rank(MPI_COMM_WORLD) % 2 == 0) {
      EXPECT_EQ(actualFailingTag, expectedFailingTag);
      EXPECT_EQ(actualFailingIndex, expectedFailingIndex);
   }

   free(testArray);
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that if the first element is different,
///        there is a diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, UnequalIntFirstElement) {
   int expectedFailingTag = 2;
   int actualFailingTag = -1;

   int expectedFailingIndex = 0;
   int actualFailingIndex = -1;

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank % 2 == 0) {
      testArray[0] = 1;
   }

   MPIDiff::Diff(length, testArray, "testArray2",
                 [&] (const int& value1, const int& value2) {
                    static int index = 0;

                    bool equal = value1 == value2;

                    if (!equal) {
                       actualFailingTag = 2;
                       actualFailingIndex = index;
                    }

                    ++index;

                    return equal;
                 });

   if (get_my_rank(MPI_COMM_WORLD) % 2 == 0) {
      EXPECT_EQ(actualFailingTag, expectedFailingTag);
      EXPECT_EQ(actualFailingIndex, expectedFailingIndex);
   }

   free(testArray);
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that if a middle element is different,
///        there is a diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, UnequalIntMiddleElement) {
   int expectedFailingTag = 3;
   int actualFailingTag = -1;

   int expectedFailingIndex = 50;
   int actualFailingIndex = -1;

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank % 2 == 1) {
      testArray[50] = -50;
   }

   MPIDiff::Diff(length, testArray, "testArray3",
                 [&] (const int& value1, const int& value2) {
                    static int index = 0;

                    bool equal = value1 == value2;

                    if (!equal) {
                       actualFailingTag = 3;
                       actualFailingIndex = index;
                    }

                    ++index;

                    return equal;
                 });

   if (get_my_rank(MPI_COMM_WORLD) % 2 == 0) {
      EXPECT_EQ(actualFailingTag, expectedFailingTag);
      EXPECT_EQ(actualFailingIndex, expectedFailingIndex);
   }

   free(testArray);
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that if the last element is different,
///        there is a diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, UnequalIntLastElement) {
   int expectedFailingTag = 4;
   int actualFailingTag = -1;

   int expectedFailingIndex = 99;
   int actualFailingIndex = -1;

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   if (myGlobalRank % 2 == 0) {
      testArray[99] = 98;
   }

   MPIDiff::Diff(length, testArray, "testArray4",
                 [&] (const int& value1, const int& value2) {
                    static int index = 0;

                    bool equal = value1 == value2;

                    if (!equal) {
                       actualFailingTag = 4;
                       actualFailingIndex = index;
                    }

                    ++index;

                    return equal;
                 });

   if (get_my_rank(MPI_COMM_WORLD) % 2 == 0) {
      EXPECT_EQ(actualFailingTag, expectedFailingTag);
      EXPECT_EQ(actualFailingIndex, expectedFailingIndex);
   }

   free(testArray);
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that if an element is within the
///        tolerance, there is no diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, FloatingPointInsideTolerance) {
   int expectedFailingTag = -1;
   int actualFailingTag = -1;

   int expectedFailingIndex = -1;
   int actualFailingIndex = -1;

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   double* testArray = (double*) malloc(length * sizeof(double));

   for (int i = 0; i < length; ++i) {
      testArray[i] = (double) i;
   }

   if (myGlobalRank % 2 == 0) {
      testArray[99] -= 1e-12;
   }

   MPIDiff::Diff(length, testArray, "testArray5",
                 [&] (const double& value1, const double& value2) {
                    static int index = 0;

                    bool equal = std::abs(value2 - value1) <= 1e-12;

                    if (!equal) {
                       actualFailingTag = 5;
                       actualFailingIndex = index;
                    }

                    ++index;

                    return equal;
                 });

   if (get_my_rank(MPI_COMM_WORLD) % 2 == 0) {
      EXPECT_EQ(actualFailingTag, expectedFailingTag);
      EXPECT_EQ(actualFailingIndex, expectedFailingIndex);
   }

   free(testArray);
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that if an element is outside the
///        tolerance, there is a diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, FloatingPointOutsideTolerance) {
   int expectedFailingTag = 6;
   int actualFailingTag = -1;

   int expectedFailingIndex = 99;
   int actualFailingIndex = -1;

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length = 100;
   double* testArray = (double*) malloc(length * sizeof(double));

   for (int i = 0; i < length; ++i) {
      testArray[i] = (double) i;
   }

   if (myGlobalRank % 2 == 0) {
      testArray[99] += 1e-11;
   }

   MPIDiff::Diff(length, testArray, "testArray6",
                 [&] (const double& value1, const double& value2) {
                    static int index = 0;

                    bool equal = std::abs(value2 - value1) <= 1e-12;

                    if (!equal) {
                       actualFailingTag = 6;
                       actualFailingIndex = index;
                    }

                    ++index;

                    return equal;
                 });

   if (get_my_rank(MPI_COMM_WORLD) % 2 == 0) {
      EXPECT_EQ(actualFailingTag, expectedFailingTag);
      EXPECT_EQ(actualFailingIndex, expectedFailingIndex);
   }

   free(testArray);
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks if using a custom predicate and to string
///        functor works.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, CustomPredicateAndToString) {
   int expectedFailingTag = -1;
   int actualFailingTag = -1;

   int expectedFailingIndex = -1;
   int actualFailingIndex = -1;

   UserDefinedType customType("value");

   MPIDiff::Diff(1, &customType, "customType",
                 [&] (const UserDefinedType& value1, const UserDefinedType& value2) {
                    static int index = 0;

                    bool equal = value1.getName() == value2.getName();

                    if (!equal) {
                       actualFailingTag = 7;
                       actualFailingIndex = index;
                    }

                    ++index;

                    return equal;
                 },
                 [] (const UserDefinedType& value) {
                    return value.getName();
                 });

   if (get_my_rank(MPI_COMM_WORLD) % 2 == 0) {
      EXPECT_EQ(actualFailingTag, expectedFailingTag);
      EXPECT_EQ(actualFailingIndex, expectedFailingIndex);
   }
}

#if 0
// TODO: Figure out a good way to handle reporting arrays of different sizes.
//       Perhaps provide a handler for checking the sizes as an argument to
//       Diff or perhaps set a handler on the MPIDiff class.

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that if the remote array is shorter
///        than the local array, there is a diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, ShorterIntArray) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length;

   if (myGlobalRank % 2 == 0) {
      length = 100;
   }
   else {
      length = 99;
   }

   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   MPIDiff::Diff(length, testArray, 1, MPIDiffTestHandler<int>{});

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 1);
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}

// TODO: This test case is currently crashing. Add a custom MPI error handler
//       or find a different strategy for handling this case.

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that if the remote array is longer
///        than the local array, there is a diff.
///
/////////////////////////////////////////////////////////////////////////
TEST(MPIDiff, LongerIntArray) {
   failingTags.clear();
   unreadSentTags.clear();
   unreadReceivedTags.clear();

   int myGlobalRank = get_my_rank(MPI_COMM_WORLD);

   int length;

   if (myGlobalRank % 2 == 0) {
      length = 100;
   }
   else {
      length = 101;
   }

   int* testArray = (int*) malloc(length * sizeof(int));

   for (int i = 0; i < length; ++i) {
      testArray[i] = i;
   }

   MPIDiff::Diff(length, testArray, 1, MPIDiffTestHandler<int>{});

   free(testArray);

   MPIDiff::Barrier();

   if (get_my_rank(MPI_COMM_WORLD) == 0) {
      ASSERT_EQ(failingTags.size(), 1);
      EXPECT_EQ(failingTags[0], 1);
      EXPECT_TRUE(unreadSentTags.empty());
      EXPECT_TRUE(unreadReceivedTags.empty());
   }
}
#endif

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


//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef MULTI_PROGRAM_DEBUG_H_
#define MULTI_PROGRAM_DEBUG_H_

// std library headers
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

// other library headers
#include <mpi.h>

inline void mpiErrorCheck(int errorCode, const char* fileName, int lineNumber, bool abort=true) {
   if (errorCode != MPI_SUCCESS) {
      int resultLen;
      char buffer[MPI_MAX_ERROR_STRING];
      MPI_Error_string(errorCode, buffer, &resultLen);

      fprintf(stderr, "[MPIDiff] MPI Error: %s %s %d\n", buffer, fileName, lineNumber);

      if (abort) {
         exit(errorCode);
      }
   }
}

#define MPI_ERROR_CHECK(code) { mpiErrorCheck((code), __FILE__, __LINE__); }

class MPIDiff {
   public:
      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Functor object that converts a value to a string.
      ///
      /////////////////////////////////////////////////////////////////////////
      template <class T>
      class to_string {
         public:
            /////////////////////////////////////////////////////////////////////////
            ///
            /// @author Alan Dayton
            ///
            /// @brief Converts a value to a string.
            ///
            /// @param[in] value   The value to convert to a string
            ///
            /// @return The string representing the value
            ///
            /////////////////////////////////////////////////////////////////////////
            std::string operator() (const T& value) const {
               std::stringstream ss;
               ss.precision(16);
               ss << value;
               return ss.str();
            }
      };

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Tony De Groot, Alan Dayton, Peter Robinson
      ///
      /// @brief Initialize multi-program communication for comparing data through MPI.
      ///
      /// @arg[in]  multiProgramCommunicator Communicator over both codes
      /// @arg[in]  programID Integer identifier for this program
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Init(const MPI_Comm multiProgramCommunicator, int programID);

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Check if MPIDiff has been initialized.
      ///
      /// @return true if Init has been called
      ///
      /////////////////////////////////////////////////////////////////////////
      static bool Initialized();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Set message handler.
      ///
      /// @arg[in] handler Function to call when message tags match
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Set_handler(std::function<void(MPI_Datatype, int,
                                                 const void*, int,
                                                 const void*, int)> handler);

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Set unread message handler.
      ///
      /// @arg[in] handler Function to call when message tags do not match
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Set_unread_handler(std::function<void(MPI_Datatype, int, bool,
                                                        const void*, int)> handler);

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Delayed reduce with debug neighbor. Saves messages until one
      ///        with a matching tag is found from the debug neighbor. Then
      ///        passes both messages to the user provided handler and frees
      ///        the data afterwards. This communication is flexible - messages
      ///        without a matching tag are ignored and flushed as soon as
      ///        one with a matching tag is found or when the memory limit is
      ///        reached, or when Flush is called. Asynchronous.
      ///
      /// @arg[in] buffer   Data to communicate
      /// @arg[in] count    Number of elements in buffer
      /// @arg[in] datatype Type of data in buffer
      /// @arg[in] tag      Integer ID used to correlate messages
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Reduce(const void* buffer, int count, MPI_Datatype datatype, int tag);

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Blocking diff with debug neighbor. Compares elements using
      ///        the std::equal_to<T> function object.
      ///
      /// @arg[in] size   Number of elements in array
      /// @arg[in] data   Data to communicate
      /// @arg[in] key    String used as a key for pairing messages
      ///
      /////////////////////////////////////////////////////////////////////////
      template <class T>
      static inline void Diff(int size, const T* data, const std::string& key) {
         Diff(size, data, key, std::equal_to<T>{}, MPIDiff::to_string<T>{});
      }

      // TODO: Investigate setting a default tolerance for floats and doubles.

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Blocking diff with debug neighbor. Compares elements using
      ///        the given tolerance.
      ///
      /// @arg[in] size        Number of elements in array
      /// @arg[in] data        Data to communicate
      /// @arg[in] key         String used as a key for pairing messages
      /// @arg[in] tolerance   The tolerance for comparing elements of data
      ///
      /////////////////////////////////////////////////////////////////////////
      template <class T>
      static inline void Diff(int size, const T* data, const std::string& key,
                              T tolerance) {
         Diff(size, data, key,
              [=] (const T& value1, const T& value2) {
                 return std::abs(value2 - value1) <= tolerance;
              },
              MPIDiff::to_string<T>{});
      }

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Blocking diff with debug neighbor. Compares elements using
      ///        the given binary predicate.
      ///
      /// @arg[in] size        Number of elements in array
      /// @arg[in] data        Data to communicate
      /// @arg[in] key         String used as a key for pairing messages
      /// @arg[in] predicate   The binary predicate for comparing elements of data
      ///
      /////////////////////////////////////////////////////////////////////////
      template <class T, class BinaryPredicate>
      static inline void Diff(int size, const T* data, const std::string& key,
                              BinaryPredicate predicate) {
         Diff(size, data, key, predicate, MPIDiff::to_string<T>{});
      }

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Blocking diff with debug neighbor. Compares elements using
      ///        the given binary predicate.
      ///
      /// @arg[in] size        Number of elements in array
      /// @arg[in] data        Data to communicate
      /// @arg[in] key         String used as a key for pairing messages
      /// @arg[in] predicate   The binary predicate for comparing elements of data
      /// @arg[in] toString    The function for converting a value to a string
      ///
      /////////////////////////////////////////////////////////////////////////
      template <class T, class BinaryPredicate, class TToString>
      static void Diff(int size, const T* data, const std::string& key,
                       BinaryPredicate predicate, TToString toString) {
         // Check that we are in a valid state
         if (!Initialized()) {
            std::cerr << "[MPIDiff] MPIDiff::Init must be called before MPIDiff::Diff. Unable to perform diffs!" << std::endl;
            return;
         }

         // Set up communication information
         const int partner = Get_partner_debug_rank();
         MPI_Comm comm = Get_debug_communicator();
         std::size_t hashCode = std::hash<std::string>{}(key);
         std::size_t tag = hashCode % Get_max_debug_tag();

         // The actual message will be of the form [keyLength, key, dataLength, data]
         const std::size_t sizeEntryLength = sizeof(std::size_t);
         const std::size_t keyEntryLength = key.size();
         const std::size_t dataEntryLength = size * sizeof(T);
         const std::size_t totalLength = sizeEntryLength + keyEntryLength + sizeEntryLength + dataEntryLength;

         char* message = (char*) malloc(totalLength);

         if (Get_debug_rank() > partner) {
            // Build the message
            std::memcpy(message, &keyEntryLength, sizeEntryLength);
            std::memcpy(message + sizeEntryLength, key.data(), keyEntryLength); 
            std::memcpy(message + sizeEntryLength + keyEntryLength, &dataEntryLength, sizeEntryLength);
            std::memcpy(message + sizeEntryLength + keyEntryLength + sizeEntryLength, data, dataEntryLength);

            // Send the message
            MPI_ERROR_CHECK(MPI_Send(message, totalLength, MPI_CHAR, partner, tag, comm));
         }
         else {
            // Receive the message
            MPI_Status status;
            MPI_ERROR_CHECK(MPI_Recv(message, totalLength, MPI_CHAR, partner, tag, comm, &status));

            // Extract the key from the message
            std::size_t receivedKeySize;
            std::memcpy(&receivedKeySize, message, sizeEntryLength);

            char* receivedKeyData = (char*) malloc(receivedKeySize);
            std::memcpy(receivedKeyData, message + sizeEntryLength, receivedKeySize);
            std::string receivedKey(receivedKeyData, keyEntryLength);

            // Check if there is a hash or tag collision
            if (key != receivedKey) {
               std::size_t receivedHashCode = std::hash<std::string>{}(receivedKey);

               if (hashCode == receivedHashCode) {
                  std::cerr << "[MPIDiff] Hash collision! Both " << key << " and "
                            << receivedKey << " have the same hash value: " << hashCode
                            << std::endl;

                  MPI_Abort(comm, MPI_ERR_TAG);
               }
               else {
                  std::cerr << "[MPIDiff] Tag collision! Both " << key << " and "
                            << receivedKey << " have the same tag value: " << tag
                            << std::endl;

                  MPI_Abort(comm, MPI_ERR_TAG);
               }
            }

            // Extract the data from the message
            std::size_t receivedDataSize;
            std::memcpy(&receivedDataSize, message + sizeEntryLength + receivedKeySize, sizeEntryLength);

            T* receivedData = (T*) malloc(receivedDataSize);
            std::memcpy(receivedData, message + sizeEntryLength + receivedKeySize + sizeEntryLength, receivedDataSize);

            // Perform the diff
            Default_diff(Get_program_id(), size, data,
                         Get_partner_program_id(), size, receivedData,
                         key, predicate, toString);

            // Clean up
            free(receivedData);
            free(receivedKeyData);
         }

         // Clean up
         free(message);
      }

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Checks for any communication to send/receive. If a message
      ///        with a matching tag is found from the debug neighbor, it
      ///        passes both messages to the user provided handler and frees
      ///        the data afterwards.
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Sendrecv();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Finishes all outstanding communication and optionally flushes
      ///        any unmatched messages.
      ///
      /// @brief flush Whether to flush any unmatched messages
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Barrier(bool flush=false);

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Finalize multi-program communication for comparing data through MPI.
      ///        Checks for any remaining receives and does comparisons if necessary.
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Finalize();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Check if MPIDiff has be finalized.
      ///
      /// @return true if Finalize has been called
      ///
      /////////////////////////////////////////////////////////////////////////
      static bool Finalized();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get the multi-program communicator.
      ///
      /// @return the multiprogram communicator
      ///
      /////////////////////////////////////////////////////////////////////////
      static MPI_Comm Get_multi_program_communicator();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my program communicator.
      ///
      /// @return my program communicator
      ///
      /////////////////////////////////////////////////////////////////////////
      static MPI_Comm Get_program_communicator();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get the debug communicator.
      ///
      /// @return the debug communicator
      ///
      /////////////////////////////////////////////////////////////////////////
      static MPI_Comm Get_debug_communicator();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my rank in the multi-program communicator.
      ///
      /// @return my multi program rank
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_multi_program_rank();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my rank in my program communicator.
      ///
      /// @return my program rank
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_program_rank();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my rank in my debug communicator.
      ///
      /// @return my debug rank
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_debug_rank();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my partner's rank in the multi-program communicator.
      ///
      /// @return my partner's multi-program rank
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_partner_multi_program_rank();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my partner's rank in my partner's program communicator.
      ///
      /// @return my partner's program rank
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_partner_program_rank();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my partner's rank in the debug communicator.
      ///
      /// @return my partner's debug rank
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_partner_debug_rank();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my program's ID.
      ///
      /// @return my program's ID
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_program_id();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get my partner's program ID.
      ///
      /// @return my partner's program ID
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_partner_program_id();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get the max debug tag.
      ///
      /// @return the max debug tag
      ///
      /////////////////////////////////////////////////////////////////////////
      static int Get_max_debug_tag();

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Set the output base file name. The local rank number will
      ///        be appended to form the full name of the output file.
      ///
      /// @param[in] outputBaseFileName   The base name of the output file
      ///
      /////////////////////////////////////////////////////////////////////////
      static void Set_output_base_file_name(const std::string& outputBaseFileName);

      static int const SUCCESS = 0;
      static int const INVALID_STATE_FOR_OPERATION = 1;

      static int const DEBUG_NEIGHBOR_WAITING = 0;
      static int const TAG_UB = MPI_TAG_UB - 10;

      struct Message {
         Message(void* buffer, int count, MPI_Datatype datatype, int tag) :
            buffer(buffer),
            count(count),
            datatype(datatype),
            tag(tag)
         {
         }

         void* buffer = nullptr;
         int count = -1;
         MPI_Datatype datatype = MPI_CHAR;
         int tag = -1;
      };

   private:
      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Perform the diff. Compares elements using the given binary predicate.
      ///
      /// @arg[in] program1ID   The ID of the first program
      /// @arg[in] size1        Number of elements in the first array
      /// @arg[in] data1        The first buffer to communicate
      /// @arg[in] program2ID   The ID of the second program
      /// @arg[in] size2        Number of elements in the second array
      /// @arg[in] data2        The second array to communicate
      /// @arg[in] tag          Integer ID used to correlate messages
      /// @arg[in] predicate    The binary predicate for comparing array elements
      ///
      /////////////////////////////////////////////////////////////////////////
      template <class T, class BinaryPredicate, class TToString>
      static void Default_diff(int program1ID, int size1, const T* data1,
                               int program2ID, int size2, const T* data2,
                               std::string key, BinaryPredicate predicate,
                               TToString toString) {
         std::ofstream& s_outputFile = Get_output_file();

         bool firstError = true;

         for (int i = 0; i < size1; ++i) {
            if (!predicate(data1[i], data2[i])) {
               if (firstError) {
                  s_outputFile << "Key " << key << std::endl
                               << "Index, "
                               << "Program " << program1ID << ", "
                               << "Program " << program2ID << std::endl;
                  firstError = false;
               }

               s_outputFile << i << ", "
                            << toString(data1[i]) << ", "
                            << toString(data2[i]) << std::endl;
            }
         }

         if (!firstError) {
            s_outputFile << std::endl;
         }
      }

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Get the output file.
      ///
      /// @return the output file
      ///
      /////////////////////////////////////////////////////////////////////////
      static std::ofstream& Get_output_file();
};

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Blocking diff with debug neighbor. Compares elements using
///        the std::equal_to<T> function object. Uses the array variable
///        name as the key.
///
/// @arg[in] SIZE    Number of elements in array
/// @arg[in] ARRAY   Data to communicate
///
/////////////////////////////////////////////////////////////////////////
#define MPIDIFF(SIZE, ARRAY) MPIDiff::Diff(SIZE, ARRAY, #ARRAY)

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Blocking diff with debug neighbor. Compares elements using
///        the given tolerance. Uses the array variable name as the key.
///
/// @arg[in] SIZE        Number of elements in array
/// @arg[in] ARRAY       Data to communicate
/// @arg[in] TOLERANCE   The tolerance for comparing elements of ARRAY
///
/////////////////////////////////////////////////////////////////////////
#define MPIDIFF_TOLERANCE(SIZE, ARRAY, TOLERANCE) MPIDiff::Diff(SIZE, ARRAY, #ARRAY, TOLERANCE)

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Blocking diff with debug neighbor. Compares elements using
///        the given binary predicate. Uses the array variable name as
///        the key.
///
/// @arg[in] SIZE        Number of elements in array
/// @arg[in] ARRAY       Data to communicate
/// @arg[in] PREDICATE   The binary predicate for comparing elements of ARRAY
///
/////////////////////////////////////////////////////////////////////////
#define MPIDIFF_PREDICATE(SIZE, ARRAY, PREDICATE) MPIDiff::Diff(SIZE, ARRAY, #ARRAY, PREDICATE)

#endif // MULTI_PROGRAM_DEBUG_H_


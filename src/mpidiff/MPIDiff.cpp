//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// Standard library headers
#include <cstdlib>
#include <cstring>
#include <list>
#include <unordered_map>
#include <vector>

// MPIDiff headers
#include "MPIDiff.h"

// Initialize static variables
static bool m_initialized = false;
static bool m_finalized = false;

static MPI_Comm m_multiProgramCommunicator;
static MPI_Comm m_programCommunicator;
static MPI_Comm m_debugCommunicator;

static int m_programID;
static int m_partnerProgramID;

static int m_multiProgramRank;
static int m_programRank;
static int m_debugRank;

static int m_multiProgramNumRanks;
static int m_programNumRanks;
static int m_debugNumRanks;

static int m_partnerMultiProgramRank;
static int m_partnerProgramRank;
static int m_partnerDebugRank;

static int m_maxDebugTag;

static std::list<MPIDiff::Message> m_messages;
static std::list<MPIDiff::Message> m_partnerMessages;

static std::vector<MPI_Request> m_requests;
static std::vector<void*> m_savedBuffers;

static std::function<void(MPI_Datatype, int,
                          const void*, int,
                          const void*, int)> m_handler;

static std::function<void(MPI_Datatype, int, bool,
                          const void*, int)> m_unreadHandler;

static std::size_t m_memoryAllocated = 0;
static std::size_t m_maxMemoryAllocated = 0;

static std::unordered_map<void*, std::size_t> m_pointerMap;

void* allocateMemory(std::size_t bytes) {
   void* pointer = malloc(bytes);

   if (pointer == nullptr) {
      printf("Out of memory!");
      abort();
   }
   else {
      if (m_pointerMap.find(pointer) != m_pointerMap.end()) {
         printf("Memory was deallocated but not removed from the pointer map!");
         abort();
      }
      else {
         m_pointerMap.insert({pointer, bytes});
      }
   }

   m_memoryAllocated += bytes;
   m_maxMemoryAllocated = std::max(m_maxMemoryAllocated, m_memoryAllocated);

   return pointer;
}

void deallocateMemory(void* pointer) {
   auto record = m_pointerMap.find(pointer);

   if (record == m_pointerMap.end()) {
      printf("Attempting to deallocate a pointer not found in the pointer map!");
      abort();
   }
   else {
      m_memoryAllocated -= record->second;
      free(pointer);
      m_pointerMap.erase(record);
   }
}


/////////////////////////////////////////////////////////////////////////
///
/// @author Tony De Groot, Alan Dayton, Peter Robinson
///
/// @brief Initialize multi-program communication for comparing data through MPI.
///
/// @arg[in] multiProgramCommunicator Communicator over both codes
/// @arg[in] programID Integer identifier for this program
///
/////////////////////////////////////////////////////////////////////////
void MPIDiff::Init(const MPI_Comm multiProgramCommunicator, int programID) {
   // Check that MPI has been initialized
   {
      int flag;
      MPI_Initialized(&flag);

      if (flag == 0) {
         fprintf(stderr, "[MPIDiff] MPI_Init must be called before any MPIDiff method!\n");
         abort();
      }
   }

   // Check arguments
   if (multiProgramCommunicator == MPI_COMM_NULL) {
      fprintf(stderr, "[MPIDiff] Must pass a valid communicator to MPIDiff::Init!\n");
      abort();
   }

   // Duplicate the multi program communicator
   MPI_ERROR_CHECK(MPI_Comm_dup(multiProgramCommunicator, &m_multiProgramCommunicator));

   // Save the program ID
   m_programID = programID;

   // Get needed info
   MPI_ERROR_CHECK(MPI_Comm_rank(m_multiProgramCommunicator, &m_multiProgramRank));
   MPI_ERROR_CHECK(MPI_Comm_size(m_multiProgramCommunicator, &m_multiProgramNumRanks));

   // Gather all program IDs
   int* allProgramIDs = (int*) allocateMemory(m_multiProgramNumRanks * sizeof(int));
   MPI_ERROR_CHECK(MPI_Allgather(&m_programID, 1, MPI_INT, allProgramIDs, 1, MPI_INT,
                                 m_multiProgramCommunicator));

   // Make a list of ranks with my program ID
   int m_programNumRanks = 0;
   int* ranksWithMyProgramID = (int*) allocateMemory(m_multiProgramNumRanks * sizeof(int));

   bool multiplePrograms = false;
   
   for (int i = 0; i < m_multiProgramNumRanks; ++i) {
      if (allProgramIDs[i] == m_programID) {
         ranksWithMyProgramID[m_programNumRanks++] = i;
      }
      else {
         multiplePrograms = true;
      }
   }

   if (!multiplePrograms) {
      fprintf(stderr, "[MPIDiff] MPIDiff only found one program ID. There must be two!\n");
      abort();
   }

   // Create a group from the multi program communicator
   MPI_Group multiProgramGroup;
   MPI_ERROR_CHECK(MPI_Comm_group(m_multiProgramCommunicator, &multiProgramGroup));

   // Create a group from the ranks with my program ID
   MPI_Group programGroup;
   MPI_ERROR_CHECK(MPI_Group_incl(multiProgramGroup, m_programNumRanks,
                                  ranksWithMyProgramID, &programGroup));

   // Create a communicator for my program
   MPI_ERROR_CHECK(MPI_Comm_create(m_multiProgramCommunicator, programGroup,
                                   &m_programCommunicator));

   // Get my program rank (already know the number of ranks)
   MPI_ERROR_CHECK(MPI_Comm_rank(m_programCommunicator, &m_programRank));

   // Gather all program ranks
   int* allProgramRanks = (int*) allocateMemory(m_multiProgramNumRanks * sizeof(int));
   MPI_ERROR_CHECK(MPI_Allgather(&m_programRank, 1, MPI_INT, allProgramRanks, 1,
                                 MPI_INT, m_multiProgramCommunicator));

   // Try to find the multi program rank of my debug neighbor.
   // My debug neighbor should have the same local rank as me,
   // but will not be in the same spot in the multi program array as me.
   for (int i = 0; i < m_multiProgramNumRanks; ++i) {
      if (allProgramRanks[i] == m_programRank && i != m_multiProgramRank) {
         m_partnerMultiProgramRank = i;
         break;
      }
   }

   // Set my partner's program ID
   m_partnerProgramID = allProgramIDs[m_partnerMultiProgramRank];

   // Clean up memory
   deallocateMemory(allProgramRanks);

   // Now create the debug communicator
   int debugRanks[2] = { std::min(m_multiProgramRank, m_partnerMultiProgramRank),
                         std::max(m_multiProgramRank, m_partnerMultiProgramRank) };

   MPI_Group debugGroup;
   MPI_ERROR_CHECK(MPI_Group_incl(multiProgramGroup, 2, debugRanks, &debugGroup));
   MPI_ERROR_CHECK(MPI_Comm_create(m_multiProgramCommunicator, debugGroup,
                                   &m_debugCommunicator));

   // Now get my debug rank
   MPI_ERROR_CHECK(MPI_Comm_rank(m_debugCommunicator, &m_debugRank));

   if (m_debugRank == 0) {
      m_partnerDebugRank = 1;
   }
   else {
      m_partnerDebugRank = 0;
   }

   // Get the max tag that can be used for the debug communicator
   int* maxTag, flag;
   MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &maxTag, &flag);

   if (flag > 0) {
      m_maxDebugTag = *maxTag;
   }
   else {
      fprintf(stderr, "[MPIDiff] Unable to get the max tag!\n");
      abort();
   }

   MPI_ERROR_CHECK(MPI_Group_free(&debugGroup));
   MPI_ERROR_CHECK(MPI_Group_free(&programGroup));
   MPI_ERROR_CHECK(MPI_Group_free(&multiProgramGroup));

   // Clean up memory
   deallocateMemory(ranksWithMyProgramID);
   deallocateMemory(allProgramIDs);

   // Set state to initialized
   m_initialized = true;

   // Set default output file
   Set_output_base_file_name("mpidiff");

   return;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Check if MPIDiff has been initialized.
///
/// @arg[out] flag Set to true if MPIDiff has been initialized
///
/////////////////////////////////////////////////////////////////////////
bool MPIDiff::Initialized() {
   return m_initialized;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Set message handler.
///
/// @arg[in] handler Function to call when message tags match
///
/////////////////////////////////////////////////////////////////////////
void MPIDiff::Set_handler(std::function<void(MPI_Datatype, int,
                                             const void*, int,
                                             const void*, int)> handler) {
   m_handler = handler;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Set unread message handler.
///
/// @arg[in] handler Function to call when message tags do not match
///
/////////////////////////////////////////////////////////////////////////
void MPIDiff::Set_unread_handler(std::function<void(MPI_Datatype, int, bool,
                                                    const void*, int)> handler) {
   m_unreadHandler = handler;
}

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
void MPIDiff::Reduce(const void* buffer, int count, MPI_Datatype datatype, int tag) {
   // Check that we are in a valid state
   if (!m_initialized) {
      fprintf(stderr, "[MPIDiff] MPIDiff::Init must be called before MPIDiff::Reduce. Expect undefined behavior!");
   }

   // Check arguments
   if (tag == DEBUG_NEIGHBOR_WAITING) {
      fprintf(stderr, "[MPIDiff] Invalid tag value %d. Tag must be {1, MPIDiff::Get_max_debug_tag()}. Expect undefined behavior!", tag);
   }

   // Send/Receive
   Sendrecv();

   if (m_multiProgramRank > m_partnerMultiProgramRank) {
      // Copy the buffer since we are doing an Isend and we are not sure when it is
      // safe to use it again.
      int typeSize;
      MPI_ERROR_CHECK(MPI_Type_size(datatype, &typeSize));
      void* saveBuffer = allocateMemory(typeSize * count);
      memcpy(saveBuffer, buffer, typeSize * count);

      // Post a send
      MPI_Request sendRequest;
      MPI_ERROR_CHECK(MPI_Isend(saveBuffer, count, datatype, m_partnerDebugRank, tag,
                                m_debugCommunicator, &sendRequest));

      m_requests.push_back(sendRequest);
      m_savedBuffers.push_back(saveBuffer);
   }
   else {
      // Check if the given buffer matches any received message
      auto it = m_partnerMessages.begin();

      for (; it != m_partnerMessages.end(); ++it) {
         Message message = *it;

         if (tag == message.tag) {
            if (m_handler) {
               m_handler(datatype, tag,
                         buffer, count,
                         message.buffer, message.count);

               deallocateMemory(message.buffer);
            }

            break;
         }
      }

      // If we didn't reach the end of the list, we found a match. Flush
      // everything up to and including the match (hence the preincrement).
      if (it != m_partnerMessages.end()) {
         for (auto it2 = m_partnerMessages.begin(); it2 != it; ++it2) {
            Message message = *it2;

            if (message.buffer) {
               if (m_unreadHandler) {
                  m_unreadHandler(message.datatype, message.tag, true,
                                  message.buffer, message.count);
               }

               deallocateMemory(message.buffer);
            }
         }

         m_partnerMessages.erase(m_partnerMessages.begin(), ++it);

         for (auto message : m_messages) {
            if (message.buffer) {
               if (m_unreadHandler) {
                  m_unreadHandler(message.datatype, message.tag, false,
                                  message.buffer, message.count);
               }

               deallocateMemory(message.buffer);
            }
         }

         m_messages.clear();
      }
      else {
         int typeSize;
         MPI_ERROR_CHECK(MPI_Type_size(datatype, &typeSize));
         void* saveBuffer = allocateMemory(typeSize * count);
         memcpy(saveBuffer, buffer, typeSize * count);
         m_messages.emplace_back(saveBuffer, count, datatype, tag);
      }
   }
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
void MPIDiff::Sendrecv() {
   // Check that we are in a valid state
   if (!m_initialized) {
      fprintf(stderr, "[MPIDiff] MPIDiff::Init must be called before MPIDiff::Reduce. Expect undefined behavior!");
   }

   // The lower rank collects all the data. The higher rank sends it (see Reduce).
   if (m_multiProgramRank < m_partnerMultiProgramRank) {
      int flag = 1;

      while (flag == 1) {
         // Probe to see the tag and length of the message
         MPI_Status status;
         MPI_ERROR_CHECK(MPI_Iprobe(m_partnerDebugRank, MPI_ANY_TAG, m_debugCommunicator,
                                    &flag, &status));

         if (flag == 1) {
            // Receive the message
            int receiveTag = status.MPI_TAG;

            if (receiveTag == 0) {
               fprintf(stderr, "[MPIDiff] Should not have received a tag of 0 here!\n");
            }

            int receiveCount;
            MPI_ERROR_CHECK(MPI_Get_count(&status, MPI_CHAR, &receiveCount));

            char* receiveBuffer = (char*) allocateMemory(receiveCount);
            MPI_ERROR_CHECK(MPI_Recv(receiveBuffer, receiveCount, MPI_CHAR,
                                     m_partnerDebugRank, receiveTag,
                                     m_debugCommunicator, MPI_STATUS_IGNORE));

            // Now check if the received messages matches any saved message
            auto it = m_messages.begin();

            for (; it != m_messages.end(); ++it) {
               Message message = *it;

               if (message.tag == receiveTag) {
                  if (m_handler) {
                     m_handler(message.datatype, receiveTag,
                               message.buffer, message.count,
                               receiveBuffer, receiveCount);
                  }

                  deallocateMemory(receiveBuffer);
                  deallocateMemory(message.buffer);
                  break;
               }
            }

            // If we didn't reach the end of the list, we found a match. Flush
            // everything up to and including the match (hence the preincrement).
            if (it != m_messages.end()) {
               for (auto it2 = m_messages.begin(); it2 != it; ++it2) {
                  Message message = *it2;

                  if (message.buffer) {
                     if (m_unreadHandler) {
                        m_unreadHandler(message.datatype, message.tag, false,
                                        message.buffer, message.count);
                     }

                     deallocateMemory(message.buffer);
                  }
               }

               m_messages.erase(m_messages.begin(), ++it);

               for (auto message : m_partnerMessages) {
                  if (message.buffer) {
                     if (m_unreadHandler) {
                        m_unreadHandler(message.datatype, message.tag, true,
                                        message.buffer, message.count);
                     }

                     deallocateMemory(message.buffer);
                  }
               }

               m_partnerMessages.clear();
            }
            else {
               m_partnerMessages.emplace_back(receiveBuffer, receiveCount, MPI_CHAR, receiveTag);
            }
         }
      }
   }
   else {
      // Check if any sent messages have been received so that memory can be freed
      int outcount;
      int* array_of_indices = (int*) allocateMemory(m_requests.size() * sizeof(int));
      MPI_ERROR_CHECK(MPI_Testsome(m_requests.size(), m_requests.data(), &outcount,
                                   array_of_indices, MPI_STATUSES_IGNORE));

      for (int i = 0; i < outcount; ++i) {
         int index = array_of_indices[i];

         if (m_requests[index] != MPI_REQUEST_NULL) {
            MPI_Request_free(&m_requests[index]);
            m_requests[index] = MPI_REQUEST_NULL;
         }

         if (m_savedBuffers[index]) {
            deallocateMemory(m_savedBuffers[index]);
            m_savedBuffers[index] = nullptr;
         }
      }

      deallocateMemory(array_of_indices);
   }
}

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
void MPIDiff::Barrier(bool flush) {
   if (!m_initialized) {
      fprintf(stderr, "[MPIDiff] MPIDiff::Init must be called before MPIDiff::Reduce. Expect undefined behavior!");
   }

   MPI_ERROR_CHECK(MPI_Barrier(m_debugCommunicator));

   // The lower rank collects all the data. The higher rank sends it (see Reduce).
   if (m_multiProgramRank < m_partnerMultiProgramRank) {
      bool receivedTokenFromPartner = false;

      while (!receivedTokenFromPartner) {
         // Probe to see the tag and length of the message
         MPI_Status status;
         MPI_ERROR_CHECK(MPI_Probe(m_partnerDebugRank, MPI_ANY_TAG, m_debugCommunicator,
                                   &status));

         // Receive the message
         int receiveTag = status.MPI_TAG;

         if (receiveTag == MPIDiff::DEBUG_NEIGHBOR_WAITING) {
            char receiveBuffer;
            MPI_ERROR_CHECK(MPI_Recv(&receiveBuffer, 1, MPI_CHAR, m_partnerDebugRank,
                                     receiveTag, m_debugCommunicator, MPI_STATUS_IGNORE));

            receivedTokenFromPartner = true;
            break;
         }
         else {
            int receiveCount;
            MPI_ERROR_CHECK(MPI_Get_count(&status, MPI_CHAR, &receiveCount));

            char* receiveBuffer = (char*) allocateMemory(receiveCount);
            MPI_ERROR_CHECK(MPI_Recv(receiveBuffer, receiveCount, MPI_CHAR,
                                     m_partnerDebugRank, receiveTag,
                                     m_debugCommunicator, MPI_STATUS_IGNORE));

            // Now check if the received messages matches any saved message
            auto it = m_messages.begin();

            for (; it != m_messages.end(); ++it) {
               Message message = *it;

               if (message.tag == receiveTag) {
                  if (m_handler) {
                     m_handler(message.datatype, receiveTag,
                               message.buffer, message.count,
                               receiveBuffer, receiveCount);
                  }

                  deallocateMemory(receiveBuffer);
                  deallocateMemory(message.buffer);
                  break;
               }
            }

            // If we didn't reach the end of the list, we found a match. Flush
            // everything up to and including the match (hence the preincrement).
            if (it != m_messages.end()) {
               for (auto it2 = m_messages.begin(); it2 != it; ++it2) {
                  Message message = *it2;

                  if (message.buffer) {
                     if (m_unreadHandler) {
                        m_unreadHandler(message.datatype, message.tag, false,
                                        message.buffer, message.count);
                     }

                     deallocateMemory(message.buffer);
                  }
               }

               m_messages.erase(m_messages.begin(), ++it);

               for (auto message : m_partnerMessages) {
                  if (message.buffer) {
                     if (m_unreadHandler) {
                        m_unreadHandler(message.datatype, message.tag, true,
                                        message.buffer, message.count);
                     }

                     deallocateMemory(message.buffer);
                  }
               }

               m_partnerMessages.clear();
            }
            else {
               m_partnerMessages.emplace_back(receiveBuffer, receiveCount, MPI_CHAR, receiveTag);
            }
         }
      }

      if (flush) {
         for (auto message : m_messages) {
            if (message.buffer) {
               if (m_unreadHandler) {
                  m_unreadHandler(message.datatype, message.tag, false,
                                  message.buffer, message.count);
               }

               deallocateMemory(message.buffer);
            }
         }

         m_messages.clear();
         
         for (auto message : m_partnerMessages) {
            if (message.buffer) {
               if (m_unreadHandler) {
                  m_unreadHandler(message.datatype, message.tag, true,
                                  message.buffer, message.count);
               }

               deallocateMemory(message.buffer);
            }
         }

         m_partnerMessages.clear();
      }
   }
   else {
      // Send the token to let the other process know they are done
      char value = 1;
      MPI_Request request;
      MPI_ERROR_CHECK(MPI_Isend(&value, 1, MPI_CHAR, m_partnerDebugRank,
                                MPIDiff::DEBUG_NEIGHBOR_WAITING,
                                m_debugCommunicator, &request));

      m_requests.push_back(request);

      // Make sure other process has received all messages
      MPI_ERROR_CHECK(MPI_Waitall(m_requests.size(), m_requests.data(), MPI_STATUSES_IGNORE));

      for (auto request : m_requests) {
         if (request != MPI_REQUEST_NULL) {
            MPI_Request_free(&request);
         }
      }

      m_requests.clear();

      for (auto buffer : m_savedBuffers) {
         if (buffer) {
            deallocateMemory(buffer);
         }
      }

      m_savedBuffers.clear();
   }

   MPI_ERROR_CHECK(MPI_Barrier(m_debugCommunicator));
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Finalize multi-program communication for comparing data through MPI.
///        Checks for any remaining receives and does comparisons if necessary.
///
/////////////////////////////////////////////////////////////////////////
void MPIDiff::Finalize() {
   if (!m_initialized) {
      fprintf(stderr, "[MPIDiff] MPIDiff::Init must be called before MPIDiff::Reduce. Expect undefined behavior!");
   }

   Barrier(true);
   m_finalized = true;
   MPI_ERROR_CHECK(MPI_Comm_free(&m_debugCommunicator));
   MPI_ERROR_CHECK(MPI_Comm_free(&m_programCommunicator));
   MPI_ERROR_CHECK(MPI_Comm_free(&m_multiProgramCommunicator));
   printf("[MPIDiff] Max memory used: %lu bytes\n", m_maxMemoryAllocated);
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Check if MPIDiff has be finalized.
///
/// @return true if Finalize has been called
///
/////////////////////////////////////////////////////////////////////////
bool MPIDiff::Finalized() {
   return m_finalized;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get the multiprogram communicator.
///
/// @return the multiprogram communicator
///
/////////////////////////////////////////////////////////////////////////
MPI_Comm MPIDiff::Get_multi_program_communicator() {
   return m_multiProgramCommunicator;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get the program communicator.
///
/// @return the program communicator
///
/////////////////////////////////////////////////////////////////////////
MPI_Comm MPIDiff::Get_program_communicator() {
   return m_programCommunicator;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get the debug communicator.
///
/// @return the debug communicator
///
/////////////////////////////////////////////////////////////////////////
MPI_Comm MPIDiff::Get_debug_communicator() {
   return m_debugCommunicator;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my rank in the multiprogram communicator.
///
/// @return my multi program rank
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_multi_program_rank() {
   return m_multiProgramRank;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my rank in my program communicator.
///
/// @return my program rank
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_program_rank() {
   return m_programRank;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my rank in my debug communicator.
///
/// @return my debug rank
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_debug_rank() {
   return m_debugRank;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my partner's rank in the multi-program communicator.
///
/// @return my partner's multi-program rank
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_partner_multi_program_rank() {
   return m_partnerMultiProgramRank;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my partner's rank in my partner's program communicator.
///
/// @return my partner's program rank
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_partner_program_rank() {
   return m_partnerProgramRank;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my partner's rank in the debug communicator.
///
/// @return my partner's debug rank
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_partner_debug_rank() {
   return m_partnerDebugRank;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my program's ID.
///
/// @return my program's ID
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_program_id() {
   return m_programID;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get my partner's program ID.
///
/// @return my partner's program ID
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_partner_program_id() {
   return m_partnerProgramID;
}

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Get the max debug tag.
///
/// @return the max debug tag
///
/////////////////////////////////////////////////////////////////////////
int MPIDiff::Get_max_debug_tag() {
   return m_maxDebugTag;
}

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
void MPIDiff::Set_output_base_file_name(const std::string& outputBaseFileName) {
   if (!m_initialized) {
      fprintf(stderr, "[MPIDiff] MPIDiff::Init must be called before any other call to MPIDiff. Unable to complete operation!\n");
   }

   // Close the old file
   std::ofstream& s_outputFile = Get_output_file();
   s_outputFile.close();

   // Open the new file
   s_outputFile.open(outputBaseFileName + "." + std::to_string(m_programRank) + ".txt");
   s_outputFile.precision(16);
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
std::ofstream& MPIDiff::Get_output_file() {
   static std::ofstream s_outputFile;
   return s_outputFile;
}

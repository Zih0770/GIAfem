cmake_minimum_required (VERSION 3.5)

project(GIAfem LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(MFEM_LIBRARIES "/home/sssou/local/mfem/lib/libmfem.so" "/usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so")
set(MFEM_INCLUDE "/home/sssou/local/mfem/include")
include_directories(${MFEM_INCLUDE})

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_library(giafem_lib src/giafem.cpp)
target_include_directories(giafem_lib PUBLIC include ${MFEM_INCLUDE})
target_link_libraries(giafem_lib PUBLIC ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(ex1 ex1.cpp)
target_link_libraries(ex1 PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(SG_PW SG_PW.cpp)
target_link_libraries(SG_PW PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


#add_subdirectory(src)





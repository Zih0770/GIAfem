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

include_directories(/usr/include/suitesparse)

add_library(giafem_lib lib/giafem.cpp)
target_include_directories(giafem_lib PUBLIC include ${MFEM_INCLUDE})
target_link_libraries(giafem_lib PUBLIC ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(ex16 demos/ex16.cpp)
target_link_libraries(ex16 PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(test demos/test.cpp)
target_link_libraries(test PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(test_N_L2 demos/test_N_L2.cpp)
target_link_libraries(test_N_L2 PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(test_interpolation demos/test_interpolation.cpp)
target_link_libraries(test_interpolation PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(test_field demos/test_field.cpp)
target_link_libraries(test_field PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(test_time_oper demos/test_time_oper.cpp)
target_link_libraries(test_time_oper PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(test_field_2 demos/test_field_2.cpp)
target_link_libraries(test_field_2 PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(test_ve demos/test_ve.cpp)
target_link_libraries(test_ve PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(SG_PW demos/SG_PW.cpp)
target_link_libraries(SG_PW PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(SG_PW_theta demos/SG_PW_theta.cpp)
target_link_libraries(SG_PW_theta PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(SG_PW_p demos/SG_PW_p.cpp)
target_link_libraries(SG_PW_p PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(elastostatic demos/elastostatic.cpp)
target_link_libraries(elastostatic PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

#add_executable(viscoelastic_beam demos/viscoelastic_beam.cpp)
#target_link_libraries(viscoelastic_beam PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(beam demos/beam.cpp)
target_link_libraries(beam PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(beam_t demos/beam_t.cpp)
target_link_libraries(beam_t PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

add_executable(MatrixLinearForm demos/MatrixLinearForm.cpp)
target_link_libraries(MatrixLinearForm PRIVATE giafem_lib ${MFEM_LIBRARIES} MPI::MPI_CXX)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


#add_subdirectory(lib)





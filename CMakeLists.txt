set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required (VERSION 3.5)

project(GIAfem LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED on)
#find_package(OpenMP REQUIRED)

message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

set(MFEM_LIBRARIES "/home/sssou/local/mfem/lib/libmfem.so" "/home/sssou/source/petsc/arch-opt/lib/libmpi.so" "~/source/hypre/build/lib/libHYPRE.so" "/home/sssou/source/mfemElasticity/installed/lib/libmfemElasticity.a")
set(MFEM_INCLUDE "/home/sssou/local/mfem/include" /home/sssou/source/mfemElasticity/installed/include)
include_directories(${MFEM_INCLUDE})

# CUDA
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

find_path(RAJA_INCLUDE_DIR RAJA/RAJA.hpp HINTS /home/sssou/source/raja/installed/include)
find_library(RAJA_LIBRARY RAJA HINTS /home/sssou/source/raja/installed/lib)
include_directories(${RAJA_INCLUDE_DIR})

find_path(AMGX_INCLUDE_DIR amgx_c.h HINTS /home/sssou/source/amgx//installed/include)
find_library(AMGX_LIBRARY amgxsh HINTS /home/sssou/source/amgx/installed/lib)
include_directories(${AMGX_INCLUDE_DIR})

set(SUNDIALS_DIR "/home/sssou/source/sundials/installed")
find_path(SUNDIALS_INCLUDE_DIR sundials/sundials_config.h HINTS ${SUNDIALS_DIR}/include)
find_library(SUNDIALS_LIBRARY sundials_cvode HINTS ${SUNDIALS_DIR}/lib)
include_directories(${SUNDIALS_INCLUDE_DIR})

set(GINKGO_DIR "/home/sssou/source/ginkgo/installed")
find_path(GINKGO_INCLUDE_DIR ginkgo/ginkgo.hpp HINTS ${GINKGO_DIR}/include)
find_library(GINKGO_LIBRARY ginkgo HINTS ${GINKGO_DIR}/lib)
include_directories(${GINKGO_INCLUDE_DIR})

set(PETSC_DIR "/home/sssou/source/petsc")
set(PETSC_ARCH "arch-opt")
include_directories("${PETSC_DIR}/${PETSC_ARCH}/include")
include_directories("${PETSC_DIR}/include")
link_directories("${PETSC_DIR}/${PETSC_ARCH}/lib")

find_package(Boost REQUIRED)

set(MPI_CXX_LIBRARIES "${PETSC_DIR}/${PETSC_ARCH}/lib/libmpi.so")
set(MPI_CXX_INCLUDE_PATH "${PETSC_DIR}/${PETSC_ARCH}/include")
#find_package(MPI REQUIRED)
#include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

#include_directories(/usr/include/suitesparse)

add_library(giafem_lib lib/giafem.cpp)
target_include_directories(giafem_lib PUBLIC include ${MFEM_INCLUDE})
target_link_libraries(giafem_lib PUBLIC ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(test demos/test.cpp)
target_link_libraries(test PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(ex demos/ex.cpp)
target_link_libraries(ex PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(ex1 demos/ex1.cpp)
target_link_libraries(ex1 PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(ex1p demos/ex1p.cpp)
target_link_libraries(ex1p PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(ex2 demos/ex2.cpp)
target_link_libraries(ex2 PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(ex2p demos/ex2p.cpp)
target_link_libraries(ex2p PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(ex3 demos/ex3.cpp)
#target_link_libraries(ex3 PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(ex3p demos/ex3p.cpp)
target_link_libraries(ex3p PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(test_interp tests/test_interp.cpp)
target_link_libraries(test_interp PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(test_integ tests/test_integ.cpp)
target_link_libraries(test_integ PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(test_coeff tests/test_coeff.cpp)
target_link_libraries(test_coeff PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(test_ode demos/test_ode.cpp)
#target_link_libraries(test_ode PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(test_ode_l2 demos/test_ode_l2.cpp)
#target_link_libraries(test_ode_l2 PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(test_ode_tau demos/test_ode_tau.cpp)
#target_link_libraries(test_ode_tau PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(test_N_L2 demos/test_N_L2.cpp)
#target_link_libraries(test_N_L2 PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(test_N_alp demos/test_N_alp.cpp)
#target_link_libraries(test_N_alp PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(test_N_tau demos/test_N_tau.cpp)
#target_link_libraries(test_N_tau PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(test_interpolation demos/test_interpolation.cpp)
#target_link_libraries(test_interpolation PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})


add_executable(test_time_oper demos/test_time_oper.cpp)
target_link_libraries(test_time_oper PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(test_ve demos/test_ve.cpp)
target_link_libraries(test_ve PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(viscoelastic_sphere demos/viscoelastic_sphere.cpp)
target_link_libraries(viscoelastic_sphere PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(SG_PW demos/SG_PW.cpp)
target_link_libraries(SG_PW PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(SG_PW_theta demos/SG_PW_theta.cpp)
target_link_libraries(SG_PW_theta PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(SG_PW_p demos/SG_PW_p.cpp)
target_link_libraries(SG_PW_p PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(elastostatic demos/elastostatic.cpp)
target_link_libraries(elastostatic PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

#add_executable(viscoelastic_beam demos/viscoelastic_beam.cpp)
#target_link_libraries(viscoelastic_beam PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(beam demos/beam.cpp)
target_link_libraries(beam PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

add_executable(beam_t demos/beam_t.cpp)
target_link_libraries(beam_t PRIVATE giafem_lib ${MFEM_LIBRARIES} ${MPI_CXX_LIBRARIES})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(/home/sssou/local/gmsh/include)
link_directories(/home/sssou/local/gmsh/lib)

set(MESH_GENERATORS
    mesh/cpp/geomesh.cpp
    mesh/cpp/geomesh_Earth.cpp
    mesh/cpp/geomesh_f.cpp
)

foreach(mesh_src ${MESH_GENERATORS})
    get_filename_component(exec_name ${mesh_src} NAME_WE)
    add_executable(${exec_name} ${mesh_src})
    target_link_libraries(${exec_name} gmsh)
endforeach()



#add_subdirectory(lib)





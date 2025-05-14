add_executable(ex1 ex1.cpp)
target_link_libraries(ex1  ${MFEM_LIBRARIES} MPI::MPI_CXX)
target_include_directories(ex1 PRIVATE ${MFEM_INCLUDE})

#add_executable(ex1p ex1p.cpp)
#target_link_libraries(ex1p  ${MFEM_LIBRARIES} MPI::MPI_CXX)
#target_include_directories(ex1p PRIVATE ${MFEM_INCLUDE})

add_executable(test1 test1.cpp)
target_link_libraries(test1  ${MFEM_LIBRARIES} MPI::MPI_CXX)
target_include_directories(test1 PRIVATE ${MFEM_INCLUDE})

cmake_minimum_required(VERSION 2.6)
project(usort)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -std=c99")


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} $ {OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(include)

add_executable(usort src/main.cpp)
target_link_libraries(usort ${MPI_LIBRARIES} m)

if(MPI_COMPILE_FLAGS)
  set_target_properties(usort PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(usort PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

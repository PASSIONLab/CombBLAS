cmake_minimum_required(VERSION 3.3)

set(CombBLAS_VERSION 1.16.0)

project(CombBLAS VERSION ${CombBLAS_VERSION} LANGUAGES C CXX)


# Main CombBLAS library
add_library(CombBLAS src/CommGrid.cpp src/mmio.c src/MPIType.cpp src/MPIOp.cpp src/MemoryPool.cpp src/hash.cpp)

# require c++14
if("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES) # check if cmake knows about c++14
  target_compile_features(CombBLAS PUBLIC cxx_std_14)
else()
  target_compile_options(CombBLAS PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>)
endif()

# set include directories
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psort-1.0/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/CombBLAS)


# MPI and OpenMP dependencies
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

if(TARGET MPI::MPI_CXX) # Use target if available (cmake >= 3.9)
  target_link_libraries(CombBLAS PUBLIC MPI::MPI_CXX)
elseif()
  target_compile_options(CombBLAS PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
  target_link_libraries(CombBLAS PUBLIC MPI_CXX_LINK_LIBRARIES "${MPI_CXX_LINKFLAGS}")
  target_include_directories(CombBLAS PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif()

if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
  # Set THREADED if OpenMP is found
  target_compile_definitions(CombBLAS PUBLIC THREADED)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(CombBLAS PUBLIC OpenMP::OpenMP_CXX)
  else()
    target_compile_options(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
    target_link_libraries(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
  endif()
endif()

ADD_SUBDIRECTORY( usort )
target_link_libraries(CombBLAS PUBLIC Usortlib)

ADD_SUBDIRECTORY( graph500-1.2/generator )
target_link_libraries(CombBLAS PUBLIC GraphGenlib)


# Set up exported configuration
# This allows CombBLAS to be installed in two ways:
#   1. In /usr/local (or whatever prefix is specified)
#   2. Exporting the current build directory. This allows a user to make
#      modifications to CombBLAS and have the changes automatically recompiled for
#      dependent projects.
# Either way, we need to create a CombBLASConfig.cmake.
set(ConfigPackageLocation lib/cmake/CombBLAS)

# Generate version number header
include(GenerateExportHeader)
generate_export_header(CombBLAS)
set_property(TARGET CombBLAS PROPERTY VERSION ${CombBLAS_VERSION})

# installation
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY psort-1.0/include/ DESTINATION include)
install(TARGETS CombBLAS EXPORT CombBLASTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
  VERSION ${CombBLAS_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# The following commands allow for option 2.
export(EXPORT CombBLASTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASTargets.cmake"
  NAMESPACE CombBLAS::
)
configure_file(cmake/CombBLASConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfig.cmake"
  COPYONLY
)

# Allow for option 2
install(EXPORT CombBLASTargets
  FILE
    CombBLASTargets.cmake
  NAMESPACE
    CombBLAS::
  DESTINATION
    ${ConfigPackageLocation}
)
export(PACKAGE CombBLAS)
install(
  FILES
    cmake/CombBLASConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)

# Testing
ENABLE_TESTING()
INCLUDE(CTest)

ADD_SUBDIRECTORY( ReleaseTests )
ADD_SUBDIRECTORY( Applications )
ADD_SUBDIRECTORY( Ordering )
ADD_SUBDIRECTORY( Applications/SpMSpV-IPDPS2017 )
ADD_SUBDIRECTORY( 3DSpGEMM )


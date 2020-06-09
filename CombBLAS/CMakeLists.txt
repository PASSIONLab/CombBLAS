cmake_minimum_required(VERSION 3.3)
project(CombBLAS VERSION 1.16.0 LANGUAGES C CXX)

# require c++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

# Main CombBLAS library
add_library(CombBLAS src/CommGrid.cpp src/mmio.c src/MPIType.cpp src/MPIOp.cpp src/MemoryPool.cpp src/hash.cpp)

# require c++14 in CombBLAS interface
if("cxx_std_14" IN_LIST CMAKE_CXX_COMPILE_FEATURES) # Use language feature if available (CMake >= 3.8)
  target_compile_features(CombBLAS PUBLIC cxx_std_14)
else()
  # Use C++14-specific features for CMake >= 3.3, which will force the appropriate language level flags.
  target_compile_features(CombBLAS PUBLIC cxx_return_type_deduction)
endif()

# set include directories
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psort-1.0/include> $<INSTALL_INTERFACE:include>)
target_include_directories(CombBLAS PRIVATE include/CombBLAS)

# MPI and OpenMP dependencies
find_package(MPI REQUIRED)
find_package(OpenMP)

if(TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
  target_link_libraries(CombBLAS PUBLIC MPI::MPI_CXX)
else()
  target_compile_options(CombBLAS PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
  target_link_libraries(CombBLAS PUBLIC "${MPI_CXX_LIBRARIES}" "${MPI_CXX_LINKFLAGS}")
  target_include_directories(CombBLAS PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif()

if(TARGET OpenMP::OpenMP_CXX) # Use target if available (CMake >= 3.9)
  target_compile_definitions(CombBLAS PUBLIC THREADED)
  target_link_libraries(CombBLAS PUBLIC OpenMP::OpenMP_CXX)
elseif(OPENMP_FOUND)
  target_compile_definitions(CombBLAS PUBLIC THREADED)
  target_compile_options(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
  target_link_libraries(CombBLAS PUBLIC "${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(usort)
target_link_libraries(CombBLAS PUBLIC Usortlib)

add_subdirectory(graph500-1.2/generator)
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
set_target_properties(CombBLAS PROPERTIES VERSION ${CombBLAS_VERSION})

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
enable_testing()
include(CTest)

add_subdirectory(ReleaseTests)
add_subdirectory(Applications)
add_subdirectory(Applications/Ordering)
add_subdirectory(Applications/BipartiteMatchings)
add_subdirectory(Applications/SpMSpV-IPDPS2017)
add_subdirectory(3DSpGEMM)

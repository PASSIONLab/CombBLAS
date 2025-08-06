cmake_minimum_required(VERSION 3.20...4.0)
project(CombBLAS VERSION 2.0.1 LANGUAGES C CXX)

# require c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(GNUInstallDirs)

# Main CombBLAS library
add_library(CombBLAS src/CommGrid.cpp src/mmio.c src/MPIType.cpp src/MPIOp.cpp src/MemoryPool.cpp src/hash.cpp)


# set include directories
target_include_directories(CombBLAS PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/psort-1.0/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE include/CombBLAS
)

# MPI dependencies
find_package(MPI REQUIRED)
if(TARGET MPI::MPI_CXX)
    target_link_libraries(CombBLAS PUBLIC MPI::MPI_CXX)
endif()

# OpenMP dependencies
find_package(OpenMP)
option(USE_OPENMP "default on, but can be turned off by users" ON)

if(TARGET OpenMP::OpenMP_CXX)
if(USE_OPENMP)
    target_compile_definitions(CombBLAS PUBLIC THREADED)
    target_link_libraries(CombBLAS PUBLIC OpenMP::OpenMP_CXX)
endif()
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
set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/cmake/CombBLAS)

# Generate version number header
include(GenerateExportHeader)
generate_export_header(CombBLAS)
set_target_properties(CombBLAS PROPERTIES VERSION ${CombBLAS_VERSION})

# installation
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY psort-1.0/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS CombBLAS EXPORT CombBLASTargets)

include(CMakePackageConfigHelpers)

configure_package_config_file(cmake/CombBLASConfig.cmake
        "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfig.cmake"
        INSTALL_DESTINATION ${ConfigPackageLocation}
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfigVersion.cmake"
    VERSION ${CombBLAS_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(EXPORT CombBLASTargets
        FILE
        CombBLASTargets.cmake
        NAMESPACE
        CombBLAS::
        DESTINATION
        ${ConfigPackageLocation}
)

# The following command allow for option 2.
export(EXPORT CombBLASTargets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASTargets.cmake"
       NAMESPACE CombBLAS::
       )
install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CombBLAS/CombBLASConfig.cmake"
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
add_subdirectory(Applications/Incremental)
add_subdirectory(3DSpGEMM)

# Top level directory has the include files

## Add timing flag for timing breakdown prints
## https://stackoverflow.com/questions/44284275/passing-compiler-options-in-cmake
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DTIMING")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTIMING")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fpermissive")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

ADD_EXECUTABLE(summa_test summa_test.cpp)
ADD_EXECUTABLE(msbfs msbfs.cpp)
#ADD_EXECUTABLE(summa_test summa_test.cpp)

TARGET_LINK_LIBRARIES( summa_test CombBLAS )
TARGET_LINK_LIBRARIES( msbfs CombBLAS )


# Following portions taken from $PETSC_DIR/share/petsc/CMakeLists.txt
# Resource refered from: https://petsc.org/main/manual/getting_started/#sec-writing-application-codes
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

add_executable(petsc_test petsc_test.c)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
# Linking with CombBLAS to avoid error due to not finding OpenMP
target_link_libraries(petsc_test PkgConfig::PETSC CombBLAS)

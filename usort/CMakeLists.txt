# Top level directory has the include files

add_library(Usortlib src/parUtils.cpp src/binUtils.cpp)
target_include_directories(Usortlib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE include/usort
)

if(TARGET MPI::MPI_CXX) # Use target if available (CMake >= 3.9)
  target_link_libraries(Usortlib PUBLIC MPI::MPI_CXX)
else()
  target_compile_options(Usortlib PUBLIC "${MPI_CXX_COMPILE_FLAGS}")
  target_link_libraries(Usortlib PUBLIC "${MPI_CXX_LIBRARIES}" "${MPI_CXX_LINKFLAGS}")
  target_include_directories(Usortlib PUBLIC "${MPI_CXX_INCLUDE_PATH}")
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS Usortlib EXPORT CombBLASTargets)


# Top level directory has the include files

ADD_LIBRARY( Usortlib src/parUtils.cpp src/binUtils.cpp )
target_include_directories(Usortlib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(Usortlib PRIVATE include/usort)


target_compile_definitions(Usortlib PUBLIC THREADED)
if(TARGET OpenMP::OpenMP_CXX)
	target_link_libraries(Usortlib PUBLIC OpenMP::OpenMP_CXX)
else()
	target_compile_options(Usortlib PUBLIC "${OpenMP_CXX_FLAGS}")
	target_link_libraries(Usortlib PUBLIC "${OpenMP_CXX_FLAGS}")
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS Usortlib EXPORT CombBLASTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)


# Top level directory has the include files

ADD_LIBRARY( GraphGenlib btrd_binomial_distribution.c splittable_mrg.c mrg_transitions.c graph_generator.c permutation_gen.c make_graph.c utils.c scramble_edges.c)
target_include_directories(GraphGenlib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE include/graph500/generator
)
if(CMAKE_C_COMPILER_ID STREQUAL "Intel")
  target_compile_options(GraphGenlib PRIVATE "-restrict")
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS GraphGenlib EXPORT CombBLASTargets)

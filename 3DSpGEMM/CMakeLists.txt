add_executable(mpipspgemm mpipspgemm.cpp)
target_link_libraries(mpipspgemm CombBLAS)

add_executable(test_mpipspgemm test_mpipspgemm.cpp)
target_link_libraries(test_mpipspgemm CombBLAS)


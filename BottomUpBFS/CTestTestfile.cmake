# CMake generated Testfile for 
# Source directory: /Users/abuluc/SVNLocal/Bottomup-CombBLAS/combblas
# Build directory: /Users/abuluc/SVNLocal/Bottomup-CombBLAS/combblas
# 
# This file includes the relevent testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(Multiplication_Test "mpirun" "-n" "4" "ReleaseTests/MultTest" "TESTDATA/rmat_scale16_A.txt" "TESTDATA/rmat_scale16_B.txt" "TESTDATA/rmat_scale16_productAB.txt" "TESTDATA/x_65536_halfdense.txt" "TESTDATA/y_65536_halfdense.txt")
ADD_TEST(Reduction_Test "mpirun" "-n" "4" "ReleaseTests/ReduceTest" "TESTDATA/sprand10000" "TESTDATA/sprand10000_sumcols" "TESTDATA/sprand10000_sumrows")
ADD_TEST(Iterator_Test "mpirun" "-n" "4" "ReleaseTests/IteratorTest" "TESTDATA" "sprand10000")
ADD_TEST(Transpose_Test "mpirun" "-n" "4" "ReleaseTests/TransposeTest" "TESTDATA" "betwinput_scale16" "betwinput_transposed_scale16")
ADD_TEST(Indexing_Test "mpirun" "-n" "4" "ReleaseTests/IndexingTest" "TESTDATA" "B_100x100.txt" "B_10x30_Indexed.txt" "rand10outta100.txt" "rand30outta100.txt")
ADD_TEST(FindSparse_Test "mpirun" "-n" "4" "ReleaseTests/FindSparse" "TESTDATA" "findmatrix.txt")
ADD_TEST(BetwCent_Test "mpirun" "-n" "4" "Applications/betwcent" "TESTDATA/SCALE16BTW-TRANSBOOL/" "10" "64")
ADD_TEST(G500_Test "mpirun" "-n" "4" "Applications/graph500" "Force" "20")
SUBDIRS(ReleaseTests)
SUBDIRS(Applications)
SUBDIRS(graph500-1.2/generator)

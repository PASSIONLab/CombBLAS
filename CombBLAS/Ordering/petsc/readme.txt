before compile
---------------

module load gcc
module load cray-petsc

run (see examples in convert.sh and  batch.sh)
------
mperpute: read a matrix market file (and permute it) and save as binary file
ex18: is the solver that solves the matrix stored in the binary file
I set the exact solution to a vector with all ones and computed the right-hand-side vector via SpMV.

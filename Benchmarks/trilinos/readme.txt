
trilinosMatSquare.cpp 
======================
What: reads a matrix market file and square it using EpetraExt::MatrixMatrix::Multiply

Requirement:
Please load the cray-trilinos module before compiling the code.
#module load cray-trilinos


Compile:
Use the makefile in the folder to compile
#make trilinosMatSquare

running:

srun -N 4 -n 16 ./trilinosMatSquare  cage12.mtx

or, use the example script in the folder 



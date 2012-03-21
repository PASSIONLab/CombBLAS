#!/bin/bash

for proc in 1 4 9 16 25 36 49
do
	echo "============================== $proc procs: ==============================="
	mpirun -np $proc python roofline_spmv.py
done

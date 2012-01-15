#!/bin/bash

for proc in 1 4 9 16 25 36 49
do
	mpirun -np $proc python roofline.py
done

#!/bin/sh

for f in *.gp
do
	gnuplot < $f
done

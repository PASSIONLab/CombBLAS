#!/bin/bash

f=$1

echo
echo "Python SR / Python Filter"
echo
grep "mean_PythonSR_PythonFilter_OTFtime" $f
echo
grep "min_PythonSR_PythonFilter_OTFtime" $f
echo
grep "max_PythonSR_PythonFilter_OTFtime" $f
echo
grep "stddev_PythonSR_PythonFilter_OTFtime" $f

echo
echo "C++ SR / Python filter"
echo
grep "mean_C++SR_PythonFilter_OTFtime" $f
echo
grep "min_C++SR_PythonFilter_OTFtime" $f
echo
grep "max_C++SR_PythonFilter_OTFtime" $f
echo
grep "stddev_C++SR_PythonFilter_OTFtime" $f

echo
echo "C++ SR / SEJITS filter"
echo
grep "mean_C++SR_SejitsFilter_OTFtime" $f
echo
grep "min_C++SR_SejitsFilter_OTFtime" $f
echo
grep "max_C++SR_SejitsFilter_OTFtime" $f
echo
grep "stddev_C++SR_SejitsFilter_OTFtime" $f
echo

echo
echo "SEJITS SR / SEJITS filter"
echo
grep "mean_SejitsSR_SejitsFilter_OTFtime" $f
echo
grep "min_SejitsSR_SejitsFilter_OTFtime" $f
echo
grep "max_SejitsSR_SejitsFilter_OTFtime" $f
echo
grep "stddev_SejitsSR_SejitsFilter_OTFtime" $f
echo

echo
echo "C++ SR / Python filter materialized"
echo
grep "mean_C++SR_PythonFilter_Mattime" $f
echo
grep "min_C++SR_PythonFilter_Mattime" $f
echo
grep "max_C++SR_PythonFilter_Mattime" $f
echo
grep "stddev_C++SR_PythonFilter_Mattime" $f
echo

echo
echo "materialization time"
grep "Materialized" $f

echo
echo "SEJITS time"
grep "Created SEJITS filter" $f
echo 
echo "TODO: Sejits SR create time"

echo
echo "Stats"
grep "vertices and" $f
grep "edges survived the filter" $f

echo
echo
grep "Total runtime" $f

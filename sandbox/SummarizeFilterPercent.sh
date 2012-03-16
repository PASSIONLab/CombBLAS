#!/bin/bash

f=$1

echo
echo "OTF TEPS:"
grep "harmonic_mean_OTF" $f

echo
echo "OTF PEPS:"
grep "harmonic_mean_IncFiltered" $f

echo
echo "OTF mean iteration time"
grep "mean_OTFtime" $f

echo
echo "Pure TEPS on materialized graph excl. materialization time"
grep "harmonic_mean_Mat" $f

echo
echo "TEPS on materialized graph incl. materialization time"
grep "harmonic_mean_PlusMatTime" $f

echo
echo "materialization time"
grep "Materialized" $f

echo
echo "Materialized mean iteration time"
grep "mean_Mattime" $f

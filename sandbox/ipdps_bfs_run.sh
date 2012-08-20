#/bin/bash

for p in 36 25 16 9 4 1
do
	for scale in 20
	do
		echo "mpirun -np $p python TwitterFilterPercent-combined.py $scale sso cso cpm cpo pso ppo > result_ipdps_bfs_${scale}_${p}.txt"
	done
done

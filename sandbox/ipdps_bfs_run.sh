#/bin/bash

for p in 9 4 1 #36 25 16 9 4 1
do
	for scale in 22
	do
		command="mpirun -np $p python TwitterFilterPercent-combined.py $scale sso cso cpm cpo pso ppo > result_ipdps_bfs_${scale}_${p}.txt"
		echo $command
		mpirun -np $p python TwitterFilterPercent-combined.py $scale sso cso cpm cpo pso ppo > result_ipdps_bfs_${scale}_${p}.txt
	done
done

# HipMCL: High-performance Markov Clustering

This is the HIP port of the GPU-based Markov Clustering. For a brief introduction to HipMCL, refer to this wiki page [HipMCL wiki](https://bitbucket.org/azadcse/hipmcl/wiki/Home).

## Build with accelerator support

The HipMCL application requires a makefile to be built. A makefile is provided under Applications named ```Makefile_mcl_gpu-frontier```. To build it, run
```
make -f Makefile_mcl_gpu-frontier
```
Don't use the cmake file in the project directory as it has issues with thread support.


## Run HipMCL

The options for base HipMCL are:
- ```-M <string>```: Input file name (required).
- ```--matrix-market```: If provided, the input file is in the matrix market format (default: the file is in labeled triples format).
- ```-base <0|1>```:  : Index of the first vertex (optional) ```[default: 1]```. This parameter is used if ```--matrix-market``` option is provided.
- ```--o <string>```:  : Output file name to list the clusters (optional, default: ```<input-file-name>.hipmcl```).
- ```-I <integer>```:  : Inflation parameter (required).
- ```-p <real>```:  : Cutoff for pruning (optional) ```[default: 0.0001]```.
- ```-pct <real> ```: Recovery percentage (optional) ```[default: 90]```.
- ```-R <integer>```: Recovery number (optional) ```[default: 1400]```.
- ```-S <integer>```: Selection number (optional) ```[default: 1100]```.
- ```--preprune```:  If provided, apply prune/select/recovery before the first iteration (needed when dense columns are present) ```[default: off]```.
- ```-rand <0|1>```: Randomly permute vertices for load balance (optional) ```[default: 0 (don't permute)]```.
- ```--remove-isolated```: If provided, remove isolated vertices ```[default: off]```.
- ```phases <integer>```: Number of phases (optional) ```[default: 1]```.
- ```-per-process-mem <integer>```: Available memory per process in GB (optional) ```[default: 0]```.
- ```--32bit-local-index```: If provided, use 32-bit local indices ```[default: off]```.
- ```--show```: If provided, show matrices after major steps (optional) ```[default: off]```.

For more information about the parameters for the base HipMCL, refer to [HipMCL wiki](https://bitbucket.org/azadcse/hipmcl/wiki/Home).

The options for the accelerator-supported HipMCL are:
- ```-lspgemm <nsparse|cpu|hybrid>```: SpGEMM algorithm to run. The ```nsparse``` option enables and uses the accelerator while the ```cpu``` option uses the CPU. The hybrid option makes a selection between those according to certain criteria ```[default: hybrid]```.
- ```--nrounds <integer>```: Number of samples to use in probabilistic memory estimation ```[default: 10]```.





## Notes for running HipMCL on Frontier 

The necessary modules for running HipMCL on Frontier are as follows:
* module swap PrgEnv-cray PrgEnv-gnu
* module load cmake rocm amd-mixed craype-accel-amd-gfx90a

Example runs:
- ```srun -n 4 -c 14 --gpus-per-task 2 --cpu-bind=threads --threads-per-core=1 -m block:cyclic --gpu-bind=closest ./mcl_gpu -M Renamed_vir_vs_vir_30_50length.indexed.mtx -I 2 --matrix-market -base 0```
- ```srun -n 16 -c 14 --gpus-per-task 2 --cpu-bind=threads --threads-per-core=1 -m block:cyclic --gpu-bind=closest ./mcl_gpu -M Renamed_arch_vs_arch_30_50length.indexed.mtx -I 2 --matrix-market -base 0 --32bit-local-index --preprune -lspgemm hybrid --nrounds 5 -per-process-mem 400```


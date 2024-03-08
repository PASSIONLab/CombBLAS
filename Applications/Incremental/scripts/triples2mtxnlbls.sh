#!/bin/bash -l

#SBATCH -q debug 
#SBATCH -C cpu
##SBATCH -A m1982 # Aydin's project (PASSION: Scalable Graph Learning for Scientific Discovery)
##SBATCH -A m2865 # Exabiome project
##SBATCH -A m4012 # Ariful's project (GraphML: Intelligent Premitives for Graph Machine Learning)
#SBATCH -A m4293 # Sparsitute project (A Mathematical Institute for Sparse Computations in Science and Engineering)

#SBATCH -t 0:30:00

#SBATCH -N 2
#SBATCH -J triples2mtxnlbls 
#SBATCH -o slurm.triples2mtxnlbls.o%j

SYSTEM=perlmutter_cpu
CORE_PER_NODE=128 # Never change. Specific to the system
PER_NODE_MEMORY=256 # Never change. Specific to the system
N_NODE=2
PROC_PER_NODE=8
N_PROC=$(( $N_NODE * $PROC_PER_NODE ))
THREAD_PER_PROC=$(( $CORE_PER_NODE / $PROC_PER_NODE ))
PER_PROC_MEM=$(( $PER_NODE_MEMORY / $PROC_PER_NODE - 2)) #2GB margin of error
export OMP_NUM_THREADS=$THREAD_PER_PROC

BINARY=$HOME/Codes/CombBLAS/_build/Applications/Incremental/triples2mtxnlbls

INPUT_DIR="${CFS}/m1982/taufique/matrix_generation/img_isolates_genomes"
M11_NAME="archaea_vs_archaea_id50_cov80"
#M11_NAME="ref_vs_ref"
M21_NAME="uniparc_active_p1_vs_archaea_id50_cov80"
#M21_NAME="meta_vs_ref"
M22_NAME="uniparc_active_p1_vs_uniparc_active_p1_id50_cov80"
#M22_NAME="meta_vs_ref"
M12_NAME="archaea_vs_uniparc_active_p1_id50_cov80"
#M12_NAME="ref_vs_meta"
TRIPLES_M11="$INPUT_DIR"/"$M11_NAME"."triples"
TRIPLES_M22="$INPUT_DIR"/"$M22_NAME"."triples"
TRIPLES_M21="$INPUT_DIR"/"$M21_NAME"."triples"
MTX_M11="$INPUT_DIR"/"$M11_NAME"."mtx"
MTX_M22="$INPUT_DIR"/"$M22_NAME"."mtx"
MTX_M21="$INPUT_DIR"/"$M21_NAME"."mtx"
MTX_M12="$INPUT_DIR"/"$M12_NAME"."mtx"
LBL_M11="$INPUT_DIR"/"$M11_NAME"."lbl"
LBL_M22="$INPUT_DIR"/"$M22_NAME"."lbl"

if [ -f $TRIPLES_M11 ]; then 
    echo [Y] TRIPLES_M11: $TRIPLES_M11; 
else
    echo [N] TRIPLES_M11: $TRIPLES_M11; 
fi

if [ -f $TRIPLES_M21 ]; then 
    echo [Y] TRIPLES_M21: $TRIPLES_M21; 
else
    echo [N] TRIPLES_M21: $TRIPLES_M21; 
fi

if [ -f $TRIPLES_M22 ]; then 
    echo [Y] TRIPLES_M22: $TRIPLES_M22; 
else
    echo [N] TRIPLES_M22: $TRIPLES_M22; 
fi

if [ -f $MTX_M11 ]; then 
    echo [Y] MTX_M11: $MTX_M11; 
    rm -rf ${MTX_M11}; 
else
    echo [N] MTX_M11: $MTX_M11; 
fi

if [ -f $MTX_M12 ]; then 
    echo [Y] MTX_M12: $MTX_M12; 
    rm -rf ${MTX_M12}; 
else
    echo [N] MTX_M12: $MTX_M12; 
fi

if [ -f $MTX_M21 ]; then 
    echo [Y] MTX_M21: $MTX_M21; 
    rm -rf ${MTX_M21}; 
else
    echo [N] MTX_M21: $MTX_M21; 
fi

if [ -f $MTX_M22 ]; then 
    echo [Y] MTX_M22: $MTX_M22; 
    rm -rf ${MTX_M22}; 
else
    echo [N] MTX_M22: $MTX_M22; 
fi

if [ -f $LBL_M11 ]; then 
    echo [Y] LBL_M11: $LBL_M11; 
    rm -rf ${LBL_M11}; 
else
    echo [N] LBL_M11: $LBL_M11; 
fi

if [ -f $LBL_M22 ]; then 
    echo [Y] LBL_M22: $LBL_M22; 
    rm -rf ${LBL_M22}; 
else
    echo [N] LBL_M22: $LBL_M22; 
fi

srun -N $N_NODE -n $N_PROC -c $THREAD_PER_PROC --ntasks-per-node=$PROC_PER_NODE --cpu-bind=cores $BINARY \
    --triples-m11 "$TRIPLES_M11" \
    --triples-m22 "$TRIPLES_M22" \
    --triples-m21 "$TRIPLES_M21" \
    --mtx-m11 "$MTX_M11" \
    --mtx-m22 "$MTX_M22" \
    --mtx-m12 "$MTX_M12" \
    --mtx-m21 "$MTX_M21" \
    --lbl-m11 "$LBL_M11" \
    --lbl-m22 "$LBL_M22" \

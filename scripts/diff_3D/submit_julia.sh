#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

project_dir=/scratch/lraess/PTsolvers/PseudoTransientDiffusion
app_dir=scripts/diff_3D

export HOME2=${project_dir}
export JULIA_PROJECT=${HOME2}/${app_dir}
export JULIA_DEPOT_PATH=${HOME2}/julia_depot

export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
export IGG_CUDAAWARE_MPI=1

export PS_THREAD_BOUND_CHECK=1
export JULIA_NUM_THREADS=4

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2
# module load openmpi/gcc83-316-c112
module load openmpi/gcc83-314-c112

julia_=$(which julia)


RES=$1
U_GPU=$2
D_VIZ=$3
D_SAVE=$4
D_SAVE_VIZ=$5
NAME=$6

USE_GPU=$U_GPU DO_VIZ=$D_VIZ DO_SAVE=$D_SAVE DO_SAVE_VIZ=$D_SAVE_VIZ NX=$RES NY=$RES NZ=$RES $julia_ -O3 --check-bounds=no "$NAME".jl

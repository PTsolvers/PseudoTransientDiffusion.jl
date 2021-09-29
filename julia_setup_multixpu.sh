#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

STARTUP=true

project_dir=/scratch/lraess/PTsolvers/PseudoTransientDiffusion
app_dir=scripts/diff_3D

export HOME2=${project_dir}
cd ${HOME2}

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2
module load openmpi/gcc83-316-c112

export JULIA_PROJECT=${HOME2}/${app_dir}
export JULIA_DEPOT_PATH=${HOME2}/julia_depot

export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_BINARY=system
export JULIA_CUDA_USE_BINARYBUILDER=false
export IGG_CUDAAWARE_MPI=1

export PS_THREAD_BOUND_CHECK=0
export JULIA_NUM_THREADS=4

# Only the first time
if [ "$STARTUP" = true ]; then

    cd ${app_dir}

    julia --project -e 'using Pkg; pkg"activate ."; pkg"add ImplicitGlobalGrid"; pkg"add CUDA"; pkg"add MPI"; pkg"add ParallelStencil"; pkg"add Plots"; pkg"add MAT"'
    julia --project -e 'using Pkg; pkg"activate ."; pkg"update"'
    
    cd ${HOME2}
fi

# Every time
cd ${app_dir}

julia --project -e 'using Pkg; pkg"instantiate"; pkg"build MPI"'
julia --project -e 'using Pkg; pkg"precompile"'

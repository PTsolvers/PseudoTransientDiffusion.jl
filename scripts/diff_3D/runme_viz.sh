#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2
# module load openmpi/gcc83-316-c112
module load openmpi/gcc83-314-c112

mpirun_=$(which mpirun)

RESOL=( 128 )
declare -a RUN=( "diff_3D_lin3_multixpu" "diff_3D_linstep3_multixpu" "diff_3D_nonlin3_multixpu" )

USE_GPU=true

DO_VIZ=false

DO_SAVE=false

DO_SAVE_VIZ=true

# Read the array values with space
for name in "${RUN[@]}"; do

    if [ "$DO_SAVE" = "true" ]; then

        FILE=../../output/out_"$name".txt
        
        if [ -f "$FILE" ]; then
            echo "Systematic results (file $FILE) already exists. Remove to continue."
            exit 0
        else 
            echo "Launching systematics (saving results to $FILE)."
        fi
    fi

    for i in "${RESOL[@]}"; do

        $mpirun_ -np 8 -rf gpu_rankfile_node40 ./submit_julia.sh $i $USE_GPU $DO_VIZ $DO_SAVE $DO_SAVE_VIZ $name
    
    done

done

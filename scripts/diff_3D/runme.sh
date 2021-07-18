#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

mpirun_=$(which mpirun)

RESOL=( 32 64 128 256 512 )

declare -a RUN=( "diff_3D_lin" "diff_3D_lin2" "diff_3D_linstep" "diff_3D_linstep2" "diff_3D_nonlin" "diff_3D_nonlin2" )

USE_GPU=true

DO_VIZ=false

DO_SAVE=true

# Read the array values with space
for name in "${RUN[@]}"; do

    if [ "$DO_SAVE" = "true" ]; then
        FILE=../../output/out_$name.txt
        if [ -f "$FILE" ]; then
            echo "Systematic results (file $FILE) already exists. Remove to continue."
            exit 0
        else 
            echo "Launching systematics (saving results to $FILE)."
        fi
    fi

    for i in "${RESOL[@]}"; do

        $mpirun_ -np 8 -rf gpu_rankfile_node40 ./submit_julia.sh $i $USE_GPU $DO_VIZ $DO_SAVE $name
    
    done

done

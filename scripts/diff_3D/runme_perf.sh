#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2
# module load openmpi/gcc83-316-c112
module load openmpi/gcc83-314-c112

mpirun_=$(which mpirun)

RESOL1=( 32 64 128 256 512 )

RESOL2=512

NP=( 1 2 4 8 )

declare -a RUN=( "diff_3D_nonlin3_perf" "diff_3D_nonlin3_multixpu_perf" )

USE_GPU=true

DO_VIZ=false

DO_SAVE=true

DO_SAVE_VIZ=false

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

    if [ "$name" = "${RUN[0]}" ]; then

        for i in "${RESOL1[@]}"; do

            for ie in {1..5}; do
            
                echo "== Running script $name, resol=$i (test $ie)"
                ./submit_julia.sh $i $USE_GPU $DO_VIZ $DO_SAVE $DO_SAVE_VIZ $name

            done

        done

    else 

        for np in "${NP[@]}"; do

            for ie in {1..5}; do

                echo "== Running script $name, resol=$RESOL2, nprocs=$np (test $ie)"
                $mpirun_ -np $np -rf gpu_rankfile_node40 ./submit_julia.sh $RESOL2 $USE_GPU $DO_VIZ $DO_SAVE $DO_SAVE_VIZ $name
    
            done

        done

    fi

done

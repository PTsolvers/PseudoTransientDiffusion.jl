#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2

julia_=$(which julia)

RESOL_1D=( 32 64 128 256 512 1024 2048 4096 8192 16384 )
declare -a RUN_1D=( "diff_1D_lin3" "diff_1D_linstep3" "diff_1D_nonlin3" )

USE_GPU=true

DO_VIZ=false

DO_SAVE=true

DO_SAVE_VIZ=false

# Read the array values with space
for name in "${RUN_1D[@]}"; do

    if [ "$DO_SAVE" = "true" ]; then

        FILE=../output/out_"$name".txt
    
        if [ -f "$FILE" ]; then
            echo "Systematic results (file $FILE) already exists. Remove to continue."
            exit 0
        else 
            echo "Launching systematics (saving results to $FILE)."
        fi
    fi

    for i in "${RESOL_1D[@]}"; do
        USE_GPU=$USE_GPU DO_VIZ=$DO_VIZ DO_SAVE=$DO_SAVE DO_SAVE_VIZ=$DO_SAVE_VIZ NX=$i $julia_ --project -O3 --check-bounds=no "$name".jl
    done
done

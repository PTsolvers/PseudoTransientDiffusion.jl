#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

module purge > /dev/null 2>&1
module load julia
module load cuda/11.2

julia_=$(which julia)

RESOL_2D=( 512 )
declare -a RUN_2D=( "diff_2D_lin" "diff_2D_linstep" "diff_2D_nonlin" )

USE_GPU=true

DO_VIZ=false

DO_SAVE=false

DO_SAVE_VIZ=true

# Read the array values with space
for name in "${RUN_2D[@]}"; do

    if [ "$DO_SAVE" = "true" ]; then

        FILE=../output/out_"$name".txt
    
        if [ -f "$FILE" ]; then
            echo "Systematic results (file $FILE) already exists. Remove to continue."
            exit 0
        else 
            echo "Launching systematics (saving results to $FILE)."
        fi
    fi

    for i in "${RESOL_2D[@]}"; do

        USE_GPU=$USE_GPU DO_VIZ=$DO_VIZ DO_SAVE=$DO_SAVE DO_SAVE_VIZ=$DO_SAVE_VIZ NX=$i NY=$i $julia_ --project -O3 --check-bounds=no "$name".jl
    
    done
done

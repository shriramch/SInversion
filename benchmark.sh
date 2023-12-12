#!/bin/bash

current_time=$(date +"%Y%m%d_%H%M%S")
echo "Start time: ${current_time}"
OUTPUT_FOLDER="output_logs_${current_time}"
# rm -rf $OUTPUT_FOLDER
rm *.o
mkdir -p $OUTPUT_FOLDER

NUMRUNS=100   # num_of_runs
SYMMETRIC=0  # matrix is symmtric or not
M_VALUES=(2048, 8192, 32768, 131072, 524288, 2097152) # matrix size 
B_VALUES=(16, 64, 128, 256, 512) # block size 

CODESETS=("rgf1" "rgf2" "rgf1_cuda" "rgf2_cuda")

for codeset in "${CODESETS[@]}"; do
    for m_value in "${M_VALUES[@]}"; do
        for b_value in "${B_VALUES[@]}"; do
            echo "*******************************"

            temp=$((2 * b_value))
            result=$((m_value % temp))
            if [  $result -ne 0 ]; then
                echo " Invalid block size: : m=$m_value, b=$b_value"
                break
            fi

            echo "  Running with m_value: $m_value"
            echo "  Running with b_value: $b_value"
            echo "  Running codeset: $codeset"
            make "$codeset"


            if [[ $codeset == "rgf1"* ]]; then
                mpirun -np 1 ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 > run.txt
            elif [[ $codeset == "rgf2"* ]]; then
                mpirun -np 2 ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 > run.txt
            fi
            if [[ $codeset == "rgf1" ]]; then
                mkdir "${OUTPUT_FOLDER}/${m_value}_${b_value}"
            fi
            for file in lsb.DPHPC_Project*; do
                mv "$file" "$OUTPUT_FOLDER/${m_value}_${b_value}/data_${codeset}.${file: -2}"
            done
        done
    done
done

rm test
current_time=$(date +"%Y%m%d_%H%M%S")
echo "End time: ${current_time}"
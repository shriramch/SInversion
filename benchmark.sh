#!/bin/bash

OUTPUT_FOLDER="output_logs"
mkdir -p $OUTPUT_FOLDER

NUMRUNS=10   #num_of_runs
SYMMETRIC=0  #matrix is symmtric or not
M_VALUES=(8 16 32) #matrix size 
B_VALUES=(2 4 8) #block size 

CODESETS=("rgf1" "rgf2" "rgf1_cuda" "rgf2_cuda") 
# CODESETS=("rgf1" "rgf2" "rgf1_cuda" "rgf2_cuda") 


for codeset in "${CODESETS[@]}"; do
    for m_value in "${M_VALUES[@]}"; do
        for b_value in "${B_VALUES[@]}"; do

            temp=$((2 * b_value))
            result=$((m_value % temp))

            if [  $result -ne 0 ]; then
                echo "**************************"
                echo " matrix size : $m_value"
                echo " invalid block size. : $b_value"
                break
            fi

            echo "*******************************"
            echo "  Running with M_VALUE: $m_value"
            echo "  Running with B_VALUE: $b_value"
            echo "  Running codeset: $codeset"
            make "$codeset"


            if [[ $codeset == "rgf1"* ]]; then
                mpirun -np 1 ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 > run.txt
            elif [[ $codeset == "rgf2"* ]]; then
                mpirun -np 2 ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 > run.txt
            fi

            for file in lsb.DPHPC_Project*; do
                mv "$file" "$OUTPUT_FOLDER/data_${m_value}_${b_value}_${codeset}_${file}"
            done

        done
    done
done
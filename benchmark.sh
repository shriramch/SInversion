#!/bin/bash

current_time=$(date +"%Y%m%d_%H%M%S")
echo "Start time: ${current_time}"
OUTPUT_FOLDER="output_logs_${current_time}"
rm *.o
mkdir -p $OUTPUT_FOLDER

NUMRUNS=100                                                # num_of_runs
SYMMETRIC=0                                                # matrix is symmtric or not

benchmark () {
    for codeset in "${CODESETS[@]}"; do
        for m_value in "${M_VALUES[@]}"; do
            for b_value in "${B_VALUES[@]}"; do
                echo "*******************************"

                temp=$((2 * b_value))
                result=$((m_value % temp))
                if [  $result -ne 0 ]; then
                    echo " Invalid block size: m=$m_value, b=$b_value"
                    break
                fi

                echo "  Running with m_value: $m_value"
                echo "  Running with b_value: $b_value"
                echo "  Running codeset: $codeset"
                make "$codeset"

                if [[ $codeset == "rgf1"* ]]; then
                    mpirun -np 1 ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 >> run.txt
                elif [[ $codeset == "rgf2"* ]]; then
                    mpirun -np 2 ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 >> run.txt
                fi
                if [[ $codeset == "rgf1_cuda" ]]; then
                    mkdir "${OUTPUT_FOLDER}/${m_value}_${b_value}"
                fi
                for file in lsb.DPHPC_Project*; do
                    mv "$file" "$OUTPUT_FOLDER/${m_value}_${b_value}/data_${codeset}.${file: -2}"
                done
            done
        done
    done
}

M_VALUES=(1024 2048 4096 8192 16384 32768 65536 131072)    # matrix size 
B_VALUES=(8 16 32 64 128)                                  # block size 
CODESETS=("rgf1_cuda" "rgf2_cuda" "rgf1" "rgf2")           # codesets

echo "*******************************"
echo "Round 1 Benchmark"
benchmark

M_VALUES=(16384 32768)                                     # matrix size 
B_VALUES=(32 128 512 2048 8192)                            # block size 
CODESETS=("rgf1_cuda" "rgf2_cuda")                         # codesets

echo "*******************************"
echo "Round 2 Benchmark"
benchmark

rm test
current_time=$(date +"%Y%m%d_%H%M%S")
echo "*******************************"
echo "End time: ${current_time}"

#!/bin/bash

current_time=$(date +"%Y%m%d_%H%M%S")
echo "Start time: ${current_time}"
OUTPUT_FOLDER="output_nsys_logs_${current_time}"
# rm -rf $OUTPUT_FOLDER
rm *.o
mkdir -p $OUTPUT_FOLDER

NUMRUNS=10   # num_of_runs
SYMMETRIC=0  # matrix is symmtric or not

M_VALUES=(16384) # matrix size, skip 524288 
B_VALUES=(256 512 1024 2048) # block size 


CODESETS=("rgf1_cuda")

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
                nsys profile ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 >> run.txt
            elif [[ $codeset == "rgf2"* ]]; then
                nsys profile ./test -m "$m_value" -b "$b_value" -n "$NUMRUNS" -s "$SYMMETRIC" -o 1 >> run.txt
            fi
            if [[ $codeset == "rgf1"* ]]; then
                mkdir "${OUTPUT_FOLDER}/${m_value}_${b_value}"
            fi
            for file in report*; do
                mv "$file" "$OUTPUT_FOLDER/${m_value}_${b_value}/data_${codeset}.${file}"
            done
        done

    done
done

rm test
current_time=$(date +"%Y%m%d_%H%M%S")
echo "End time: ${current_time}"

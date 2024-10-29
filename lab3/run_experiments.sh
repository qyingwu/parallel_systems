#!/bin/bash

# Define the path to the executable and input files
EXEC="./myprogram"
INPUT_FILES=("coarse.txt")  # Add "fine.txt" if needed

# Define the ranges for comp-workers
COMP_WORKER_VALUES=(1 2 4 6 7 8 9 10 12 14 16 18 20 24 28 32 36 48 56 64 100 128)

# Loop through each input file to create separate results files
for INPUT_FILE in "${INPUT_FILES[@]}"; do
  # Define the output file based on the input file name
  OUTPUT_FILE="experiment_results_${INPUT_FILE%.txt}.csv"
  echo "comp-workers, implementation, hashTime, groupTime, compareTreeTime, totalExecutionTime" > $OUTPUT_FILE
  
  # Loop through each value of comp-workers
  for COMP_WORKERS in "${COMP_WORKER_VALUES[@]}"; do
    # Run the program and capture output
    echo "Running with comp-workers=$COMP_WORKERS on $INPUT_FILE"
    RUN_OUTPUT=$($EXEC -input="./input/$INPUT_FILE" -comp-workers=$COMP_WORKERS 2>&1)

    # Debug log of entire output
    echo "Full output for comp-workers=$COMP_WORKERS:" >> debug_log.txt
    echo "$RUN_OUTPUT" >> debug_log.txt
    echo "------------------------------" >> debug_log.txt

    # Determine which implementation was used
    if echo "$RUN_OUTPUT" | grep -q "sequentialImpl"; then
      IMPLEMENTATION="sequentialImpl"
    elif echo "$RUN_OUTPUT" | grep -q "channelImpl"; then
      IMPLEMENTATION="channelImpl"
    elif echo "$RUN_OUTPUT" | grep -q "mutexImpl"; then
      IMPLEMENTATION="mutexImpl"
    elif echo "$RUN_OUTPUT" | grep -q "semaphoreImpl"; then
      IMPLEMENTATION="semaphoreImpl"
    else
      IMPLEMENTATION="Unknown"
    fi

    # Extract specific performance metrics from output, setting empty values if missing
    HASH_TIME=$(echo "$RUN_OUTPUT" | grep "hashTime" | awk '{print $2}')
    GROUP_TIME=$(echo "$RUN_OUTPUT" | grep "groupTime" | awk '{print $2}')
    COMPARE_TREE_TIME=$(echo "$RUN_OUTPUT" | grep "compareTreeTime" | awk '{print $2}')
    TOTAL_EXECUTION_TIME=$(echo "$RUN_OUTPUT" | grep "totalExecutionTime" | awk '{print $2}')

    # Provide default values if metrics are missing
    HASH_TIME=${HASH_TIME:-"N/A"}
    GROUP_TIME=${GROUP_TIME:-"N/A"}
    COMPARE_TREE_TIME=${COMPARE_TREE_TIME:-"N/A"}
    TOTAL_EXECUTION_TIME=${TOTAL_EXECUTION_TIME:-"N/A"}

    # Append results to output file in CSV format
    echo "$COMP_WORKERS, $IMPLEMENTATION, $HASH_TIME, $GROUP_TIME, $COMPARE_TREE_TIME, $TOTAL_EXECUTION_TIME" >> $OUTPUT_FILE

    # Add a brief delay to ensure all processes complete
    sleep 0.5
  done

  echo "Experiment for $INPUT_FILE completed. Results saved to $OUTPUT_FILE."
done

echo "All experiments completed."

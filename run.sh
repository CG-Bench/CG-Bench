#!/bin/bash

# Get command line arguments
TASK_MODE=${1:-""}
MODEL_NAME=${2:-""}
MODEL_SIZE=${3:-""}
NUM_SEGMENT=${4:-32}
SUB=${5:-true}
SUB_TIME=${6:-true}
FRAME_TIME=${7:-true}

# Check if required arguments are provided
if [ -z "$TASK_MODE" ] || [ -z "$MODEL_NAME" ] || [ -z "$MODEL_SIZE" ]; then
    echo "Error: Required arguments missing"
    echo "Usage: $0 TASK_MODE MODEL_NAME MODEL_SIZE [NUM_SEGMENT] [SUB] [SUB_TIME] [FRAME_TIME]"
    exit 1
fi

# Adjust NUM_SEGMENT for clue_acc task if needed
if [ "$TASK_MODE" = "clue_acc" ] && [ "$NUM_SEGMENT" -gt 32 ]; then
    NUM_SEGMENT=32
fi

# Determine method based on TASK_MODE
if [ "$TASK_MODE" = "clue_acc" ]; then
    METHOD="interval"
else
    METHOD="global"
fi

# Run extract_frames.py
python ./run/extract_frames.py --method "$METHOD" --num_segment "$NUM_SEGMENT"

# Check if extract_frames.py executed successfully
if [ $? -ne 0 ]; then
    echo "Error: extract_frames.py failed"
    exit 1
fi

# Run run_api.py
python ./run/run_api.py \
    --task_mode "$TASK_MODE" \
    --model_name "$MODEL_NAME" \
    --model_size "$MODEL_SIZE" \
    --num_segment "$NUM_SEGMENT" \
    --sub "$SUB" \
    --sub_time "$SUB_TIME" \
    --frame_time "$FRAME_TIME"

# Check if run_api.py executed successfully
if [ $? -ne 0 ]; then
    echo "Error: run_api.py failed"
    exit 1
fi
#!/bin/bash
set -e  # exit on error

# ---- User Configurable Variables ----
CONFIG_PATH="configs/train/trainer.yaml"
TOKENIZER_CONFIG="configs/model/tokenizer.yaml"
MODEL_CONFIG="configs/model/model_config.yaml"
DATALOADER_CONFIG="configs/data/dataloader.yaml"
CHECKPOINT_PATH=""   # set to checkpoint file path if resuming, else leave empty
DEVICE="cpu"        # "cuda" | "cpu"
DEBUG="false"        # "true" enables debug logging

# ---- Derived Args ----
RESUME_ARG=""
if [[ -n "$CHECKPOINT_PATH" ]]; then
    RESUME_ARG="--resume $CHECKPOINT_PATH"
fi

# ---- Run Training ----
echo "Starting training..."
echo "Config: $CONFIG_PATH"
echo "Device: $DEVICE"
echo "Resume: ${CHECKPOINT_PATH:-None}"
echo "Debug: $DEBUG"

python -m src.training.cli \
    --config $CONFIG_PATH \
    --config_tokenizer $TOKENIZER_CONFIG \
    --config_model $MODEL_CONFIG \
    --config_dataloader $DATALOADER_CONFIG \
    --device $DEVICE \
    $RESUME_ARG \
    $( [[ "$DEBUG" == "true" ]] && echo "--debug" )
echo "Training script finished."
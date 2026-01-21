#!/bin/bash

# Configuration
MODE="asms_elastic"
BETA=0.9
LAM=1.0
H_PEAK=0.1
TAU=0.85
BETA_UP=0.95
LAMBDA_DOWN=2.5
NAME="ASMS_Elastic"
GEN_LEN=512

echo "Starting Experiments with Name: $NAME"

# ==========================================
# Countdown Experiments
# ==========================================
echo "Running Countdown Experiments..."

# Block Size 2
# Steps = 512 / 2 = 256
echo "Countdown Block 2"
python tests/run_countdown_asms.py \
    --mode $MODE \
    --beta $BETA \
    --lam $LAM \
    --h_peak $H_PEAK \
    --tau $TAU \
    --beta_up $BETA_UP \
    --lambda_down $LAMBDA_DOWN \
    --name "$NAME" \
    --block_length 2 \
    --gen_length $GEN_LEN \
    --steps 256

# Block Size 8
# Steps = 512 / 8 = 64
echo "Countdown Block 8"
python tests/run_countdown_asms.py \
    --mode $MODE \
    --beta $BETA \
    --lam $LAM \
    --h_peak $H_PEAK \
    --tau $TAU \
    --beta_up $BETA_UP \
    --lambda_down $LAMBDA_DOWN \
    --name "$NAME" \
    --block_length 8 \
    --gen_length $GEN_LEN \
    --steps 64

# Block Size 32
# Steps = 512 / 32 = 16
echo "Countdown Block 32"
python tests/run_countdown_asms.py \
    --mode $MODE \
    --beta $BETA \
    --lam $LAM \
    --h_peak $H_PEAK \
    --tau $TAU \
    --beta_up $BETA_UP \
    --lambda_down $LAMBDA_DOWN \
    --name "$NAME" \
    --block_length 32 \
    --gen_length $GEN_LEN \
    --steps 16

# ==========================================
# Math500 Experiments
# ==========================================
echo "Running Math500 Experiments..."

# Block Size 4
# Steps = 512 / 4 = 128
echo "Math500 Block 4"
python tests/run_math500_asms.py \
    --mode $MODE \
    --beta $BETA \
    --lam $LAM \
    --h_peak $H_PEAK \
    --tau $TAU \
    --beta_up $BETA_UP \
    --lambda_down $LAMBDA_DOWN \
    --name "$NAME" \
    --block_length 4 \
    --gen_length $GEN_LEN \
    --steps 128

# Block Size 8
# Steps = 512 / 8 = 64
echo "Math500 Block 8"
python tests/run_math500_asms.py \
    --mode $MODE \
    --beta $BETA \
    --lam $LAM \
    --h_peak $H_PEAK \
    --tau $TAU \
    --beta_up $BETA_UP \
    --lambda_down $LAMBDA_DOWN \
    --name "$NAME" \
    --block_length 8 \
    --gen_length $GEN_LEN \
    --steps 64

# Block Size 32
# Steps = 512 / 32 = 16
echo "Math500 Block 32"
python tests/run_math500_asms.py \
    --mode $MODE \
    --beta $BETA \
    --lam $LAM \
    --h_peak $H_PEAK \
    --tau $TAU \
    --beta_up $BETA_UP \
    --lambda_down $LAMBDA_DOWN \
    --name "$NAME" \
    --block_length 32 \
    --gen_length $GEN_LEN \
    --steps 16

echo "All experiments completed."

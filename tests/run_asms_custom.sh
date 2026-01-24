#!/bin/bash

# ==========================================
# Kinetic-Only (Countdown)
# ==========================================
echo "Running Kinetic-Only (Countdown) Block 2"
python tests/run_countdown_asms.py \
    --mode asms \
    --beta 0.85 \
    --lam 0.5 \
    --tau 0.85 \
    --h_peak 0.1 \
    --block_length 2 \
    --gen_length 512 \
    --steps 256 \
    --name "Kinetic-Only" \
    --no_semantic

# ==========================================
# Soft-Elastic (Math500)
# ==========================================
echo "Running Soft-Elastic (Math500) Block 2"
python tests/run_math500_asms.py \
    --mode asms_elastic \
    --beta 0.8 \
    --beta_up 0.95 \
    --lambda_down 1.2 \
    --lam 0.5 \
    --tau 0.85 \
    --h_peak 0.1 \
    --block_length 2 \
    --gen_length 512 \
    --steps 256 \
    --name "Soft-Elastic"

# ==========================================
# High-Trust Kinetic (Math500)
# ==========================================
echo "Running High-Trust Kinetic (Math500) Block 2"
python tests/run_math500_asms.py \
    --mode asms \
    --beta 0.8 \
    --lam 0.5 \
    --tau 0.75 \
    --h_peak 0.1 \
    --block_length 2 \
    --gen_length 512 \
    --steps 256 \
    --name "High-Trust Kinetic" \
    --no_semantic

echo "All experiments completed."

echo "Running Standard (Countdown) Block 2"
python tests/run_countdown_asms.py \
    --mode asms \
    --beta 0.85 \
    --lam 0.5 \
    --tau 0.85 \
    --h_peak 0.1 \
    --block_length 2 \
    --gen_length 512 \
    --steps 256 \
    --name "Standard" \

echo "All experiments completed."

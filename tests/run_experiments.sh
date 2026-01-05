#!/bin/bash

# ==========================================
# ASMS EXPERIMENTAL GRID
# ==========================================

echo "Starting ASMS Benchmark Suite..."

# --- SET 1: BASELINES ---
# The standard we must beat. RCR is the current SOTA for reasoning.
# ------------------------
echo "Running Baselines..."
python benchmark_asms.py --mode lcr --name "Baseline_LCR"
python benchmark_asms.py --mode rcr --name "Baseline_RCR"


# --- SET 2: THE GOLDEN ZONE (Core Hypothesis) ---
# Derived from Theory.md. Balanced stability vs. agility.
# High Beta (0.8) = Smoother ride (less flicker).
# High Lambda (1.0) = Stronger error correction.
# ------------------------
echo "Running Golden Zone (Stability Tests)..."

# Config A: Balanced (The Default)
python benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.1 --tau 0.85 --name "ASMS_Golden_Balanced"

# Config B: Agile Corrector (Lower momentum mass, harder correction kick)
# Use this if the model is too "sluggish" to fix errors.
python benchmark_asms.py --mode asms --beta 0.6 --lam 1.0 --h_peak 0.1 --tau 0.85 --name "ASMS_Golden_Agile"


# --- SET 3: STRESS TEST (Aggressive) ---
# Use this if RCR beats ASMS. We need to force the model to "un-stick".
# Beta 0.9 = Heavy flywheel (resists noise).
# Lambda 1.5 = Massive kick (overpowers confidence when dropping).
# ------------------------
echo "Running Stress Tests..."

python benchmark_asms.py --mode asms --beta 0.9 --lam 1.5 --h_peak 0.1 --tau 0.85 --name "ASMS_Stress_Heavy"


# --- SET 4: ABLATION (Sensitivity) ---
# Test if the "Flicker Zone" (h_peak) logic is actually correct.
# If h_peak=0.3 performs better, your theory about low-entropy flicker is wrong.
# ------------------------
echo "Running Ablation..."

python benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.3 --tau 0.85 --name "ASMS_Ablation_HighEntropy"

echo "All experiments completed."

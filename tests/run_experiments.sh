#!/bin/bash

# ==========================================
# ASMS EXPERIMENTAL GRID
# ==========================================

echo "Starting ASMS Benchmark Suite..."

# --- SET 1: BASELINES ---
# The standard we must beat. RCR is the current SOTA for reasoning.
# ------------------------
echo "Running Baselines..."
python tests/benchmark_asms.py --mode lcr --name "Baseline_LCR"
python tests/benchmark_asms.py --mode rcr --name "Baseline_RCR"


# --- SET 2: THE GOLDEN ZONE (Core Hypothesis) ---
# Derived from Theory.md. Balanced stability vs. agility.
# High Beta (0.8) = Smoother ride (less flicker).
# High Lambda (1.0) = Stronger error correction.
# ------------------------
echo "Running Golden Zone (Stability Tests)..."

# Config A: Balanced (The Default)
python tests/benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.1 --tau 0.85 --name "ASMS_Golden_Balanced"

# Config B: Agile Corrector (Lower momentum mass, harder correction kick)
# Use this if the model is too "sluggish" to fix errors.
python tests/benchmark_asms.py --mode asms --beta 0.6 --lam 1.0 --h_peak 0.1 --tau 0.85 --name "ASMS_Golden_Agile"


# --- SET 3: STRESS TEST (Aggressive) ---
# Use this if RCR beats ASMS. We need to force the model to "un-stick".
# Beta 0.9 = Heavy flywheel (resists noise).
# Lambda 1.5 = Massive kick (overpowers confidence when dropping).
# ------------------------
echo "Running Stress Tests..."

python tests/benchmark_asms.py --mode asms --beta 0.9 --lam 1.5 --h_peak 0.1 --tau 0.85 --name "ASMS_Stress_Heavy"


# --- SET 4: ABLATION (Sensitivity) ---
# Test if the "Flicker Zone" (h_peak) logic is actually correct.
# If h_peak=0.3 performs better, your theory about low-entropy flicker is wrong.
# ------------------------
echo "Running Ablation..."

python tests/benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.3 --tau 0.85 --name "ASMS_Ablation_HighEntropy"


# --- SET 5: MANEUVER B (Entropy Alignment Sweep) ---
# Sweep h_peak with fixed Golden Balanced settings (beta=0.8, lam=0.5, tau=0.85).
# This tests whether the "Flicker Zone" peak at ~0.1 is optimal or if other entropy
# thresholds produce better results. Tests: 0.05 (lower), 0.2 (higher), 0.3 (very high).
# (0.1 is already covered in Golden Zone Balanced)
# ------------------------
echo "Running Maneuver B (Entropy Sweep)..."

python tests/benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.05 --tau 0.85 --name "ASMS_Entropy_0.05"
python tests/benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.2 --tau 0.85 --name "ASMS_Entropy_0.2"
python tests/benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.3 --tau 0.85 --name "ASMS_Entropy_0.3"


# --- SET 6: ELASTIC MODE (Asymmetric Momentum) ---
# Tests the "easy up, hard down" hypothesis.
# beta_up controls rising confidence, lambda_down controls falling.
# ------------------------
echo "Running Elastic Mode Tests..."

# Config A: Default Elastic (Balanced asymmetry)
python tests/benchmark_asms.py --mode asms_elastic --beta 0.8 --lam 0.5 --h_peak 0.1 --tau 0.85 --beta_up 0.9 --lambda_down 1.5 --name "Elastic_Default"

# Config B: Aggressive Elastic (Harder punishment for drops)
python tests/benchmark_asms.py --mode asms_elastic --beta 0.8 --lam 0.5 --h_peak 0.1 --tau 0.85 --beta_up 0.9 --lambda_down 2.0 --name "Elastic_Aggressive"

# Config C: Soft Elastic (Gentler asymmetry)
python tests/benchmark_asms.py --mode asms_elastic --beta 0.8 --lam 0.5 --h_peak 0.1 --tau 0.85 --beta_up 0.7 --lambda_down 1.2 --name "Elastic_Soft"

# Config D: Extreme Elastic (Maximum stubbornness without RCR lock-in)
python testsbenchmark_asms.py --mode asms_elastic --beta 0.9 --lam 1.0 --h_peak 0.1 --tau 0.85 --beta_up 0.95 --lambda_down 2.5 --name "Elastic_Extreme"


# --- SET 7: KINETIC-ONLY MODE (No Semantics) ---
# Tests pure momentum trajectory without embedding cosine similarity.
# Useful when embedding space is anisotropic (similarity ~0.9 for all pairs).
# ------------------------
echo "Running Kinetic-Only (No Semantics) Tests..."

# Kinetic-Only ASMS (no embedding similarity)
python tests/benchmark_asms.py --mode asms --beta 0.8 --lam 0.5 --h_peak 0.1 --tau 0.85 --no_semantic --name "Kinetic_Balanced"

# Kinetic-Only Elastic (no embedding similarity + asymmetric momentum)
python tests/benchmark_asms.py --mode asms_elastic --beta 0.8 --lam 0.5 --beta_up 0.9 --lambda_down 1.5 --h_peak 0.1 --tau 0.85 --no_semantic --name "Kinetic_Elastic"

# Kinetic-Only Aggressive (high lambda for harder correction)
python tests/benchmark_asms.py --mode asms --beta 0.9 --lam 1.5 --h_peak 0.1 --tau 0.85 --no_semantic --name "Kinetic_Aggressive"

echo "All experiments completed."

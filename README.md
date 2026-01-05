# Adaptive Semantic-Momentum Sampling (ASMS)

**ASMS** is a kinetic control algorithm for discrete diffusion models (such as LLaDA) that eliminates "Temporal Oscillation" (flickering) during generation. By applying momentum to the confidence trajectory and gating it with semantic similarity and entropy, ASMS achieves stable, high-quality generation.

## Features
- **Semantic Hysteresis**: Distinguishes between refinement and instability using cosine similarity.
- **Entropy-Gated Decay**: Dynamically adjusts momentum decay based on prediction uncertainty.
- **Breakout Threshold**: Allows high-confidence corrections to bypass momentum inertia.

## Installation
```bash
pip install -e .
```

## Usage
ASMS is implemented as an enhanced sampling function compatible with LLaDA models.

```python
import torch
from transformers import AutoTokenizer
from asms import LLaDAModelLM, sample

# Load Model
model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)

# Generate
prompt = "Your prompt here"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Run ASMS
out, _, _, _ = sample(
    model, 
    input_ids, 
    mask_id=126336, 
    steps=64, 
    asms=True,      # Enable ASMS
    beta_base=0.8,  # Momentum decay
    h_peak=0.1,     # Flicker Zone (Normalized Entropy)
    lambda_mom=0.5  # Momentum weight
)

print(tokenizer.batch_decode(out, skip_special_tokens=True)[0])
```

## Structure
- `src/asms/`: Core package
    - `models/llada/`: LLaDA model definition
    - `generation.py`: ASMS implementation (`sample`)
- `tests/`: Verification scripts

## Theory
See [theory.md](theory.md) for the mathematical formulation.

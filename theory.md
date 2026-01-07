# Adaptive Semantic-Momentum Sampling (ASMS): Theoretical Formulation

## 1. Problem Statement: Temporal Oscillation & Hallucination Lock-in
Discrete Diffusion Models (DDMs) like LLaDA generate text by iteratively refining a sequence. Two opposing failure modes plague standard sampling strategies:

1.  **Temporal Oscillation (Flickering):** The model wavers between valid candidates (e.g., "happy" vs. "glad") across steps. Standard Low-Confidence Remasking (LCR) fails to dampen this because it is memoryless.
2.  **Stubbornness (Hallucination Lock-in):** Running Confidence Remasking (RCR) solves flickering by taking the maximum historical confidence ($S_t = \max(C_t, S_{t-1})$). However, this creates "Stubbornness"—if the model hallucinates with high confidence early on, RCR ignores subsequent drops in confidence, locking in the error.

## 2. Solution: ASMS Control Loop
ASMS treats the sampling process as a **Kinetic Control Problem**. We apply momentum to the confidence trajectory, modulated by semantic stability and entropy, to achieve "Elastic Stability"—resisting noise while yielding to strong negative evidence.

### 2.1. Semantic Hysteresis (The "Soft Reset")
Standard momentum in continuous space ($v_t = \gamma v_{t-1} + \eta \nabla$) fails in discrete text because token identities change discontinuously. We define "Semantic Consistency" $\mathcal{S}$ as the cosine similarity between the embeddings of consecutive tokens:

$$\mathcal{S}_t = \text{CosSim}(\mathbf{E}(x_t), \mathbf{E}(x_{t-1}))$$

The Momentum Update Rule becomes:

$$d_t = \Delta C_t + \beta \cdot \mathcal{S}_t \cdot d_{t-1}$$

where $\Delta C_t = C_t - C_{t-1}$.

### 2.1.1. Kinetic-Only Ablation ("Efficiency Mode")
If embedding computation is too costly, or if the embedding space is anisotropic, we can disable semantic gating ($\mathcal{S}_t = 1$).
* **Performance Note:** Experiments show this achieves parity with Semantic mode in balanced configurations (74% accuracy) but is fragile under asymmetric pressure (Elastic Mode), dropping to 72% accuracy due to the lack of semantic protection for valid synonyms.

### 2.2. Gamma-Skewed Entropy Decay (Refined Anti-Stubbornness)
Oscillation typically occurs at **low normalized entropy** (binary conflicts, $\bar{H} \approx 0.1$). High entropy indicates broad confusion where momentum should be disabled. We define the decay factor $\beta_t$ using a skewed Beta-like distribution:

$$\beta_t = Z \cdot \beta_{base} \cdot \bar{H}^\gamma \cdot (1 - \bar{H})$$

Using $H_{peak} \approx 0.1$ as the "Flicker Zone":
$$\gamma \approx 0.11, \quad Z \approx 1.41$$

## 3. Elastic Mode: Active Punishment
*This section has been updated to reflect the "Active Punishment" implementation.*

### 3.1. Motivation
Standard momentum applies equal inertia whether confidence is rising or falling. RCR applies infinite inertia (max-pooling) only when rising. **Elastic Mode** bridges these by actively punishing drops in confidence.

### 3.2. Asymmetric Active Momentum
Instead of just decaying the history, we asymmetrically scale the **current change** ($\Delta C_t$). This creates an "Active Punishment" mechanism:

$$d_t = \mathbf{\alpha(\Delta C_t)} \cdot \Delta C_t + \mathbf{\kappa(\Delta C_t)} \cdot \beta_t \cdot \mathcal{S}_t \cdot d_{t-1}$$

Where the coefficients depend on the direction of change:

| Condition | $\Delta C_t > 0$ (Rising) | $\Delta C_t \leq 0$ (Falling) |
| :--- | :--- | :--- |
| **Input Scale** $\alpha$ | $1.0$ (Trust the rise) | $\lambda_{down} \approx 2.5$ (Amplify the drop) |
| **Buffer Scale** $\kappa$ | $\beta_{up} \approx 0.95$ (Keep history) | $0.5 \cdot \beta_{up}$ (Dampen history) |

### 3.3. The "Penalty Box" Effect
By setting $\lambda_{down} \gg 1$ (e.g., 2.5), a small drop in raw confidence (e.g., -0.1) becomes a massive drop in the momentum score (-0.25).
* **Result:** The token's score tanks, pushing it to the bottom of the sorting queue.
* **Benefit:** This solves RCR's stubbornness. If the model doubts a token even slightly, Elastic Mode flushes it out immediately, preventing lock-in.

## 4. Implementation Strategy: Precision Monotonicity
While standard diffusion (MaskGit) relies on "Iterative Correction" (unmasking and re-masking), ASMS achieves superior results (78% vs 60%) using **Precision Monotonicity**.

* **Adaptive Sorting:** Instead of correcting mistakes, ASMS focuses on **ordering** commitments correctly.
* **The Mechanism:**
    1.  The "Penalty Box" (Elastic Mode) ensures unstable tokens (hallucinations) have very low scores.
    2.  These tokens are forced to the back of the unmasking queue.
    3.  They remain masked until the very end, when maximum context is available to resolve the ambiguity.
* **Conclusion:** In reasoning tasks (GSM8K), preventing early errors via strict sorting is superior to trying to "erase" errors later.

## 5. Summary of Algorithm
1.  **Compute Raw Confidence:** $C_t = P(x_t | x_t^{masked})$.
2.  **Compute Similarity:** $\mathcal{S}_t = \text{CosSim}(x_t, x_{t-1})$.
3.  **Calculate Delta:** $\Delta C_t = C_t - C_{t-1}$.
4.  **Apply Elastic Scales:** Amplify $\Delta C_t$ by $\lambda_{down}$ if negative.
5.  **Update Momentum:** $d_t = \alpha \Delta C + \kappa \beta \mathcal{S} d_{t-1}$.
6.  **Score & Sort:** $S_t = C_t + \lambda d_t$.
7.  **Unmask:** Unmask the top-k highest scoring tokens (Monotonic) OR Re-mask the bottom-k (Iterative).

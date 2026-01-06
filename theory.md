# Adaptive Semantic-Momentum Sampling (ASMS): Theoretical Formulation

## 1. Problem Statement: Temporal Oscillation
Discrete Diffusion Models (DDMs) like LLaDA generate text by iteratively refining a sequence of tokens from a masked state. A common failure mode is **Temporal Oscillation** (flickering), where the model wavers between two or more valid candidates across steps.

Let $x_t^{(i)}$ be the token at position $i$ at step $t$. Oscillation occurs when:
$$x_t^{(i)} \neq x_{t-1}^{(i)} \neq x_{t-2}^{(i)}$$

Standard Low-Confidence Remasking (LCR) fails to dampen this because it is memoryless. Running Confidence Remasking (RCR) addresses this by taking the maximum historical confidence:
$$S_t^{(i)} = \max(C_t^{(i)}, S_{t-1}^{(i)})$$
However, RCR suffers from "Stubbornness"—once a high confidence is observed, it locks in, even if the token is wrong (hallucination).

## 2. Solution: ASMS Control Loop
ASMS treats the sampling process as a **Kinetic Control Problem**. Instead of max-pooling, we apply momentum to the confidence trajectory, modulated by semantic stability and entropy.

### 2.1. Semantic Hysteresis (The "Soft Reset")
Standard momentum in continuous space: $v_t = \gamma v_{t-1} + \eta \nabla$.
In discrete space, "velocity" is ill-defined because token identities change discontinuously. We define "Semantic Consistency" $\mathcal{S}$ as the cosine similarity between the embeddings of consecutive tokens:

$$\mathcal{S}_t = \text{CosSim}(\mathbf{E}(x_t), \mathbf{E}(x_{t-1}))$$

where $\mathbf{E}(\cdot)$ is the embedding function.

The Momentum Update Rule becomes:
$$d_t = (C_t - C_{t-1}) + \beta \cdot \mathcal{S}_t \cdot d_{t-1}$$

### 2.1.1. Kinetic-Only Ablation ("No Semantics")
Anisotropy in LLM embedding spaces can weaken the signal-to-noise ratio of Cosine Similarity ($\mathcal{S}_t$). If $\mathcal{S}_t \approx 0.9$ for all token pairs, it acts merely as a constant damping factor rather than a semantic gate.

The **Kinetic-Only** mode ($\mathcal{S}_t = 1$) isolates the pure momentum trajectory:
$$d_t = (C_t - C_{t-1}) + \beta_t \cdot d_{t-1}$$

If oscillation is driven primarily by confidence instability rather than semantic drift, this mode may improve performance and speed by bypassing embedding lookups.

### 2.2. Gamma-Skewed Entropy Decay (Refined Anti-Stubbornness)
To precisely target "Temporal Oscillation" without locking in confident errors, we use a skewed momentum schedule. Oscillation typically occurs at **low normalized entropy** (binary/ternary conflicts, $\bar{H} \approx 0.1$), whereas high entropy indicates broad confusion.

We define the decay factor $\beta_t$ using a skewed Beta-like distribution:

$$\beta_t = Z \cdot \beta_{base} \cdot \bar{H}^\gamma \cdot (1 - \bar{H})$$

Using $H_{peak} \approx 0.1$ as the "Flicker Zone":
$$\gamma = \frac{H_{peak}}{1 - H_{peak}} \approx 0.11$$
$$Z = \frac{1}{H_{peak}^\gamma (1 - H_{peak})} \approx 1.41$$

*   **Logic**:
    *   **Zero Entropy ($\bar{H} \to 0$)**: $\beta \to 0$. Confident hallucinations are not stabilized.
    *   **Flicker Zone ($\bar{H} \approx 0.1$)**: $\beta \to \beta_{base}$. Max stability for synonyms.
    *   **High Entropy ($\bar{H} \to 1$)**: $\beta \to 0$. No momentum for confusion.

### 2.3. Breakout Threshold (Trust Region)
If the model is overwhelmingly confident ($C_t > \tau$), we trust the current prediction absolutely, ignoring momentum. This acts as a "Trust Region" optimization.

Final Score Formulation:
$$S_t = \begin{cases} 
C_t & \text{if } C_t > \tau \\
C_t + \lambda \cdot d_t & \text{otherwise}
\end{cases}$$

## 3. Summary of Algorithm
1.  **Compute Raw Confidence**: $C_t = P(x_t | x_t^{masked})$.
2.  **Compute Entropy**: $\bar{H}_t$.
3.  **Compute Similarity**: $\mathcal{S}$.
4.  **Update Momentum**: Using Gamma-Skewed $\beta_t$.
5.  **Compute Remasking Score**: $S_t = C_t + \lambda d_t$.
6.  **Mask**: Mask $N(t)$ tokens with lowest $S_t$.

## 4. Elastic Mode: Asymmetric Momentum

### 4.1. Motivation
RCR (Running Confidence Remasking) is "stubborn"—it only allows confidence to rise via max-pooling. Standard ASMS is "symmetric"—momentum applies equally whether confidence is rising or falling.

**Elastic Mode** bridges these: confidence can rise easily (like RCR) but is punished when falling (unlike RCR's lock-in).

### 4.2. Asymmetric Momentum Update
We decouple the momentum coefficient based on the direction of confidence change:

$$d_t = \Delta C_t + \alpha(\Delta C_t) \cdot \beta_t \cdot \mathcal{S}_t \cdot d_{t-1}$$

where the direction-dependent coefficient is:

$$\alpha(\Delta C_t) = \begin{cases} 
\beta_{up} & \text{if } \Delta C_t > 0 \text{ (rising)} \\
\lambda_{down} & \text{if } \Delta C_t \leq 0 \text{ (falling)}
\end{cases}$$

### 4.3. Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| $\beta_{up}$ | 0.9 | Momentum coefficient when confidence is rising. Higher = smoother upward trajectory. |
| $\lambda_{down}$ | 1.5 | Momentum coefficient when confidence is falling. Higher = stronger resistance to drops. |

### 4.4. Intuition
*   **$\beta_{up} = 0.9$**: When confidence rises, we trust it and allow smooth accumulation.
*   **$\lambda_{down} = 1.5$**: When confidence drops, we amplify the negative momentum, making it "expensive" to lose confidence. This prevents oscillation without the permanent lock-in of RCR.

### 4.5. Comparison
| Method | Rising Confidence | Falling Confidence | Failure Mode |
|--------|-------------------|--------------------|--------------|
| RCR | Locks in (max-pool) | Ignored | Stubbornness (hallucination lock-in) |
| ASMS | $\beta_t$ | $\beta_t$ | Can oscillate if $\beta$ too low |
| **ASMS Elastic** | $\beta_{up} \cdot \beta_t$ | $\lambda_{down} \cdot \beta_t$ | Balanced (easy up, hard down) |

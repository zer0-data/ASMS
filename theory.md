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

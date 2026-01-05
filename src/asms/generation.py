import torch
import numpy as np
import torch.nn.functional as F
import torch.distributions as dists
from asms.models.llada.generate import add_gumbel_noise, get_num_transfer_tokens_maskgit

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

@torch.no_grad()
def sample(model, prompt, mask_id, prompt_mask=None, steps=64, gen_length=128, block_length=32, temperature=0.,
                 conf_alg='random', mode="linear", rcr=False, top_p=None, top_k=None,
                 # ASMS arguments
                 asms=False, beta_base=0.8, h_peak=0.1, lambda_mom=0.5, sim_thresh=0.5, breakout_thresh=0.85,
                 # Optimization
                 return_intermediates=False):
    '''
    ASMS Generation Function.
    '''
    if prompt_mask == None:
        prompt_mask = torch.ones_like(prompt) 
    # Use mixed precision for faster computation
    with torch.amp.autocast("cuda", enabled=True):
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        attn_mask = torch.ones_like(x)
        attn_mask[:, :prompt_mask.shape[1]] = prompt_mask.clone()
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        intermediate_inputs = []
        intermediate_results = []
        intermediate_confidence = []
        # Adjust steps if needed
        steps_per_block = max(1, steps // num_blocks)
        overtime_confidence = torch.zeros_like(x, dtype=torch.float32)
        
        # ASMS state initialization
        if asms:
            momentum_buffer = torch.zeros_like(x, dtype=torch.float32)
            prev_confidence = torch.zeros_like(x, dtype=torch.float32)
            # Precompute embedding shape for initialization
            input_embeddings = model.get_input_embeddings().weight
            embed_dim = input_embeddings.shape[1]
            vocab_size = input_embeddings.shape[0]
            max_entropy = np.log(vocab_size)
            prev_x0_embeddings = torch.zeros((x.shape[0], x.shape[1], embed_dim), device=x.device, dtype=torch.float32)
            
            # Precompute Gamma-Skewed Parameters
            # Gamma calculation for peak at h_peak
            # Peak of x^gamma * (1-x) is at gamma / (gamma + 1)
            # h_peak = gamma / (gamma + 1) => gamma = h_peak / (1 - h_peak)
            gamma = h_peak / (1 - h_peak)
            # Z normalization factor to make peak amplitude 1.0
            # max_val = h_peak^gamma * (1 - h_peak)
            z_factor = 1.0 / ( (h_peak ** gamma) * (1 - h_peak) )

        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = (x[:, start_idx:end_idx] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens_maskgit(block_mask_index, steps_per_block, mode=mode)

            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                if return_intermediates:
                    intermediate_inputs.append(x.clone().cpu()[:, -gen_length:])
                # Handle classifier-free guidance more efficiently
                logits = model(x, attention_mask=attn_mask).logits
                # Apply Gumbel noise for sampling, commented for now as Dream 7B crashes with gumbel noise
                # logits = logits(logits, temperature)
                if temperature > 0:
                    logits = logits / temperature
                if top_p is not None and top_p < 1:
                    logits = top_p_logits(logits, top_p)
                if top_k is not None:
                    logits = top_k_logits(logits, top_k)
                # x0 = torch.argmax(logits_with_noise, dim=-1)
                # Handle remasking strategy
                if conf_alg == 'random' and not asms:
                    p = torch.rand(x.shape, device=x.device)
                    log_p = None
                else:
                    # Use log_softmax for numerical stability
                    log_p = F.log_softmax(logits, dim=-1)
                    p = log_p.exp()
                    
                if temperature > 0:
                    try:
                        x0 = dists.Categorical(probs=p).sample()
                        confidence = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
                    except:
                        confidence, x0 = p.max(dim=-1)
                else:
                    confidence, x0 = p.max(dim=-1)
                
                # ASMS Logic
                if asms:
                    # Semantic Hysteresis
                    current_embeddings = model.get_input_embeddings()(x0) # (B, L, D)
                    # Normalize embeddings once for efficient cosine similarity
                    current_embeddings_norm = F.normalize(current_embeddings, p=2, dim=-1)

                    # Check for first step to avoid Division by Zero and Momentum Kick
                    is_first_step = (i == 0)

                    if is_first_step:
                        # Initialize state without applying momentum
                        momentum_updated = torch.zeros_like(confidence)
                        beta_t = 0.0 # Not used but good for tracing
                        final_score = confidence
                    else:
                        # Standard ASMS Logic
                        # Optimized cosine similarity: dot product of normalized vectors
                        similarity = torch.sum(current_embeddings_norm * prev_x0_embeddings, dim=-1) # (B, L)
                        
                        # Gamma-Skewed Entropy-Gated Decay
                        # Use precomputed log_p for stability
                        entropy = -torch.sum(p * log_p, dim=-1) # (B, L)
                        # max_entropy precomputed outside loop
                        norm_entropy = entropy / max_entropy # (B, L)
                        
                        # Gamma-Skewed Beta Calculation
                        # beta_t = beta_base * Z * (H^gamma) * (1 - H)
                        beta_t = beta_base * z_factor * (norm_entropy.pow(gamma)) * (1 - norm_entropy)
                        
                        # Handle numerical instability or NaNs if any
                        beta_t = torch.nan_to_num(beta_t, nan=0.0)
                        
                        # Momentum Update
                        delta_C = confidence - prev_confidence
                        momentum_updated = delta_C + beta_t * similarity * momentum_buffer
                    
                    # Update State (Always happens)
                    momentum_buffer = momentum_updated.clone()
                    prev_confidence = confidence.clone()
                    # Store normalized embeddings for next step's similarity calculation
                    prev_x0_embeddings = current_embeddings_norm.clone()
                    
                    # Compute Remasking Score
                    # If confidence > breakout, trust region activated
                    if not is_first_step:
                        score = confidence + lambda_mom * momentum_updated
                        final_score = torch.where(confidence > breakout_thresh, confidence, score)
                        # Use final_score as confidence for masking selection
                        confidence = final_score
                    
                    # Use final_score as confidence for masking selection
                    confidence = final_score

                if not rcr and return_intermediates:
                    intermediate_confidence.append(confidence.clone().cpu()[:, -gen_length:])
                if conf_alg == 'entropy':
                    # Use precomputed log_p
                    confidence = torch.sum(p * log_p, dim=-1)
                elif conf_alg == "topk_margin":
                    sorted_probs, _ = torch.sort(p, dim=-1, descending=True)
                    # Extract top1 and top2 probabilities
                    top1_probs = sorted_probs[:, :, 0]
                    top2_probs = sorted_probs[:, :, 1]
                    # Calculate confidence as top1 - top2
                    confidence = top1_probs - top2_probs
                
                # Ensure we don't process tokens beyond the current block
                confidence[:, end_idx:] = -np.inf
                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                if return_intermediates:
                    intermediate_results.append(x0.clone().cpu()[:, -gen_length:])
                # valid_token_mask = x0 != 198
                # confidence = torch.where(torch.logical_and(mask_index, valid_token_mask), x0_p, torch.tensor(-np.inf, device=x0.device))
                confidence = torch.where(mask_index, confidence, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if rcr and not asms:
                        _, select_indices = torch.topk(confidence[j], k=num_transfer_tokens[j, i:].sum().item())
                        x[j, select_indices] = x0[j, select_indices]
                        overtime_confidence[j, select_indices] = confidence[j, select_indices].clone()
                        # if (x[j,:] == mask_id).sum() <= 0:
                        if i != (steps_per_block - 1):
                            overtime_conf_wo_zeros = \
                                torch.where(overtime_confidence == 0.0, 1.0, overtime_confidence)[j]
                            num_tokens_to_mask = num_transfer_tokens[j, i + 1:].sum().item()
                            _, mask_select_indices = torch.topk(overtime_conf_wo_zeros, k=num_tokens_to_mask,
                                                                largest=False)
                            if len(mask_select_indices) == 0:
                                break
                            x[j, mask_select_indices] = mask_id
                    else:
                        if num_tokens > 0:
                            _, select_indices = torch.topk(confidence[j], k=num_tokens)
                            x[j, select_indices] = x0[j, select_indices]
                if rcr and return_intermediates:
                    intermediate_confidence.append(overtime_confidence.clone().cpu()[:, -gen_length:])
        return x[:, -gen_length:], intermediate_results, intermediate_confidence, intermediate_inputs

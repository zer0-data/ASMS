import torch
import torch.nn.functional as F
import numpy as np
import time
import re
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

def extract_answer(text):
    """
    Extracts the answer after '####' in the text.
    """
    if "####" in text:
        text = text.split("####")[1]
    
    # Clean and extract last number
    text = text.strip().replace(",", "")
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)
    if numbers:
        return numbers[-1]
    return ""

def calculate_flicker(intermediate_results, check_last_n=16):
    """
    Calculates the average number of tokens that flipped in the last n steps.
    intermediate_results: list of tensors (1, L)
    """
    if not intermediate_results or len(intermediate_results) < 2:
        return 0.0
    
    # We only care about the last n steps
    steps_to_check = intermediate_results[-check_last_n:]
    if len(steps_to_check) < 2:
        return 0.0
        
    flickers = 0
    total_checks = 0
    
    for i in range(1, len(steps_to_check)):
        prev = steps_to_check[i-1]
        curr = steps_to_check[i]
        
        # Measure flips (tokens that changed)
        # Assuming batch size 1 for this benchmark
        diff = (prev != curr).sum().item()
        flickers += diff
        total_checks += 1
        
    return flickers / total_checks

def gamma_func(r, mode="cosine", total_num=512):
    if mode == "linear":
        mask_ratio = 1 - r
    elif mode == "cosine":
        mask_ratio = np.cos(r * np.pi / 2)
    elif "pow" in mode:
        exponent = float(mode.replace("pow", ""))
        mask_ratio = 1 - r ** exponent
    elif mode == "log":
        mask_ratio = -np.log2(r) / np.log2(total_num)
    elif mode == "exp":
        mask_ratio = 1 - np.exp2(-np.log2(total_num) * (1-r))
    else:
        raise NotImplementedError
    mask_ratio = np.clip(mask_ratio, 1e-6, 1)
    return mask_ratio

def get_num_transfer_tokens_maskgit(mask_index, steps, mode="linear"):
    total_num = mask_index.sum(dim=1, keepdim=True)
    ratios = [[gamma_func((t+1) / steps, mode=mode, total_num=total_num_item.item()) for t in range(steps)] for total_num_item in total_num[:, 0]]
    num_transfer_tokens = total_num.expand((total_num.shape[0], steps))
    mask_ratios = torch.tensor(ratios).to(mask_index.device)
    num_transfer_tokens = total_num - torch.floor(num_transfer_tokens * mask_ratios)
    
    # Calculate diff to get number of tokens to transfer PER STEP
    num_transfer_tokens_diff = torch.cat([num_transfer_tokens[:, 0:1], num_transfer_tokens[:, 1:] - num_transfer_tokens[:, :-1]], dim=1)
    return num_transfer_tokens_diff.to(torch.int64)

@torch.no_grad()
def generate_lcr(model, input_ids, mask_id, steps=64, gen_length=128, block_length=32):
    """
    Baseline LCR (MaskGit) generation.
    Correct Logic:
    1. Pass input to model.
    2. Predict all tokens x0.
    3. Fill masks with predictions.
    4. Calculate confidence of the FILLED tokens (x).
    5. Mask the tokens with lowest confidence.
    """
    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=input_ids.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()
    prompt_len = input_ids.shape[1]
    
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)
    
    intermediates = []

    for num_block in range(num_blocks):
        start_idx = prompt_len + num_block * block_length
        end_idx = prompt_len + (num_block + 1) * block_length
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            logits = model(x).logits
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            
            # 1. Update belief: Fill masked tokens with current best guess
            curr_block_mask = (x[:, start_idx:end_idx] == mask_id)
            for j in range(x.shape[0]):
                block_indices = torch.where(curr_block_mask[j])[0] + start_idx
                x[j, block_indices] = x0[j, block_indices]
            
            # Capture intermediate state
            intermediates.append(x.clone())
            
            # 2. Schedule
            progress = (i + 1) / steps_per_block
            r = progress
            mask_ratio = np.cos(r * np.pi / 2) 
            mask_ratio = np.clip(mask_ratio, 0, 1)
            
            block_size = end_idx - start_idx
            num_to_mask = int(block_size * mask_ratio)
            
            if i == steps_per_block - 1:
                num_to_mask = 0
            
            # 3. Re-masking
            # CRITICAL FIX: Calculate confidence of the ACTUAL tokens in x
            # We want P(x_token | masked_input)
            # x now contains mixed content: old tokens + new x0 fills
            # We must evaluate if the old tokens are still supported by the model
            
            x_confidence = torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1)
            scores = x_confidence[:, start_idx:end_idx] # (B, block_len)
            
            if num_to_mask > 0:
                # Mask the 'num_to_mask' tokens with LOWEST confidence
                _, indices = torch.topk(scores, k=num_to_mask, largest=False, dim=-1)
                
                for j in range(x.shape[0]):
                    mask_pos = indices[j] + start_idx
                    x[j, mask_pos] = mask_id
            
    flicker = calculate_flicker(intermediates)
    return x, flicker

@torch.no_grad()
def generate_rcr(model, input_ids, mask_id, steps=64, gen_length=128, block_length=32):
    """
    Baseline RCR (Recursive Consistency) generation.
    Correct Logic:
    1. Maintain overtime_confidence (Running Max).
    2. At each step, predict x0.
    3. Fill masks.
    4. Calculate confidence of FILLED tokens (x).
    5. Update history: overtime_confidence = max(overtime_confidence, x_confidence).
    6. Re-mask based on schedule using the RUNNING MAX confidence.
    """
    x = torch.full((1, input_ids.shape[1] + gen_length), mask_id, dtype=torch.long, device=input_ids.device)
    x[:, :input_ids.shape[1]] = input_ids.clone()
    prompt_len = input_ids.shape[1]
    
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)
    
    overtime_confidence = torch.zeros_like(x, dtype=torch.float32)
    intermediates = []

    for num_block in range(num_blocks):
        start_idx = prompt_len + num_block * block_length
        end_idx = prompt_len + (num_block + 1) * block_length
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            logits = model(x).logits
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            
            # Fill current block completely with new predictions
            curr_block_mask = (x[:, start_idx:end_idx] == mask_id)
            for j in range(x.shape[0]):
                block_indices = torch.where(curr_block_mask[j])[0] + start_idx
                x[j, block_indices] = x0[j, block_indices]
                
            intermediates.append(x.clone())
            
            # CRITICAL FIX: Update Running Max Confidence using ACTUAL token confidence
            x_confidence = torch.gather(probs, -1, x.unsqueeze(-1)).squeeze(-1)
            current_conf_block = x_confidence[:, start_idx:end_idx]
            
            overtime_confidence[:, start_idx:end_idx] = torch.maximum(
                overtime_confidence[:, start_idx:end_idx], 
                current_conf_block.float()
            )
            
            # Schedule
            progress = (i + 1) / steps_per_block
            r = progress
            mask_ratio = np.cos(r * np.pi / 2) 
            mask_ratio = np.clip(mask_ratio, 0, 1)
            block_size = end_idx - start_idx
            num_to_mask = int(block_size * mask_ratio)
            
            if i == steps_per_block - 1:
                num_to_mask = 0
                
            # Re-mask using RUNNING MAX confidence
            scores = overtime_confidence[:, start_idx:end_idx]
            
            if num_to_mask > 0:
                _, indices = torch.topk(scores, k=num_to_mask, largest=False, dim=-1)
                for j in range(x.shape[0]):
                    mask_pos = indices[j] + start_idx
                    x[j, mask_pos] = mask_id

    flicker = calculate_flicker(intermediates)
    return x, flicker

def main():
    print("Loading LLaDA...")
    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    try:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Loading GSM8K...")
    data = load_dataset("gsm8k", "main", split="test")
    subset = data.select(range(5)) # Run 5 samples for test

    mask_id = 126336 # Default for LLaDA
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id

    correct_lcr = 0
    flicker_lcr_sum = 0
    correct_rcr = 0
    flicker_rcr_sum = 0
    
    for i, example in enumerate(tqdm(subset)):
        question = example["question"]
        answer = extract_answer(example["answer"])
        
        messages = [{"role": "user", "content": question + "\nPlease answer step by step and finish your answer with '#### <final answer>' on the last line."}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.cuda()
        
        # LCR
        out_lcr, f_lcr = generate_lcr(model, input_ids, mask_id)
        txt_lcr = tokenizer.decode(out_lcr[0, input_ids.shape[1]:], skip_special_tokens=True)
        pred_lcr = extract_answer(txt_lcr)
        if pred_lcr == answer:
            correct_lcr += 1
        flicker_lcr_sum += f_lcr
            
        # RCR
        out_rcr, f_rcr = generate_rcr(model, input_ids, mask_id)
        txt_rcr = tokenizer.decode(out_rcr[0, input_ids.shape[1]:], skip_special_tokens=True)
        pred_rcr = extract_answer(txt_rcr)
        if pred_rcr == answer:
            correct_rcr += 1
        flicker_rcr_sum += f_rcr

        print(f"Sample {i}: Truth={answer}")
        print(f"  LCR: Correct={pred_lcr==answer}, Flicker={f_lcr:.2f}, Pred={pred_lcr}")
        print(f"  RCR: Correct={pred_rcr==answer}, Flicker={f_rcr:.2f}, Pred={pred_rcr}")

    print("-" * 40)
    print(f"LCR Accuracy: {correct_lcr}/5 | Avg Flicker: {flicker_lcr_sum/5:.2f}")
    print(f"RCR Accuracy: {correct_rcr}/5 | Avg Flicker: {flicker_rcr_sum/5:.2f}")

if __name__ == "__main__":
    main()

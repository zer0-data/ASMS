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
    Baseline LCR (MaskGit) generation
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
        
        # Initial mask for this block
        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_maskgit(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            logits = model(x).logits
            confidence = F.softmax(logits, dim=-1)
            x0 = torch.argmax(confidence, dim=-1)
            
            # Capture what the model thinks the FULL sequence is right now
            current_full_pred = torch.where(mask_index, x0, x)
            intermediates.append(current_full_pred.clone())
            
            x0_confidence, _ = torch.max(confidence, dim=-1)
            current_confidence = torch.where(mask_index, x0_confidence, torch.tensor(-1.0, device=x.device))
            
            for j in range(x.shape[0]):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    block_conf = current_confidence[j, start_idx:end_idx]
                    k = min(k, (x[j, start_idx:end_idx] == mask_id).sum().item())
                    
                    if k > 0:
                        _, topk_indices = torch.topk(block_conf, k=k)
                        global_indices = topk_indices + start_idx
                        x[j, global_indices] = x0[j, global_indices]
                        
    flicker = calculate_flicker(intermediates)
    return x, flicker

@torch.no_grad()
def generate_rcr(model, input_ids, mask_id, steps=64, gen_length=128, block_length=32):
    """
    Baseline RCR (Recursive Consistency) generation
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
        
        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens_maskgit(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            logits = model(x).logits
            confidence = F.softmax(logits, dim=-1)
            x0 = torch.argmax(confidence, dim=-1)
            x0_confidence, _ = torch.max(confidence, dim=-1)
            
            # Capture full prediction before we enact schedule
            current_full_pred = torch.where(mask_index, x0, x)
            intermediates.append(current_full_pred.clone())
            
            curr_block_mask = (x[:, start_idx:end_idx] == mask_id)
            
            # 1. Fill everything in block
            for j in range(x.shape[0]):
                block_indices = torch.where(curr_block_mask[j])[0] + start_idx
                x[j, block_indices] = x0[j, block_indices]
                overtime_confidence[j, block_indices] = x0_confidence[j, block_indices]
                
                # 2. Mask out lowest confidence to meet schedule
                # Target revealed count = sum of num_transfer_tokens up to this step
                target_revealed_count = num_transfer_tokens[j, :i+1].sum().item()
                
                block_conf = overtime_confidence[j, start_idx:end_idx]
                
                if target_revealed_count < (end_idx - start_idx):
                    num_to_mask = (end_idx - start_idx) - target_revealed_count
                    _, low_conf_indices = torch.topk(block_conf, k=num_to_mask, largest=False)
                    global_mask_indices = low_conf_indices + start_idx
                    x[j, global_mask_indices] = mask_id

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

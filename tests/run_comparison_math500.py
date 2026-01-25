import torch
import numpy as np
import torch.nn.functional as F
import re
import json
import time
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from asms import LLaDAModelLM, sample
import argparse

# ==========================================
# 1. MATH-500 Utilities
# ==========================================

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

def extract_answer_model_output(text):
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(answer_str):
    if not answer_str:
        return ""
    clean = answer_str.replace('$', '').replace('\\', '')
    clean = "".join(clean.split())
    clean = clean.strip('.')
    return clean

def check_equality_heuristic(predicted, gold):
    if predicted is None:
        return False
    norm_pred = normalize_answer(predicted)
    norm_gold = normalize_answer(str(gold))
    return norm_pred == norm_gold

# ==========================================
# 2. Trusted RCR Implementation (from test_baselines_simple.py)
# ==========================================

def calculate_flicker(intermediate_results, check_last_n=16):
    if not intermediate_results or len(intermediate_results) < 2:
        return 0.0
    steps_to_check = intermediate_results[-check_last_n:]
    if len(steps_to_check) < 2:
        return 0.0
    flickers = 0
    total_checks = 0
    for i in range(1, len(steps_to_check)):
        prev = steps_to_check[i-1]
        curr = steps_to_check[i]
        diff = (prev != curr).sum().item()
        flickers += diff
        total_checks += 1
    return flickers / total_checks

@torch.no_grad()
def generate_rcr(model, input_ids, mask_id, steps=64, gen_length=128, block_length=32):
    """
    Baseline RCR (Recursive Consistency) generation.
    Trusted implementation from test_baselines_simple.py.
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
            logits = model(x).logits
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            
            # Fill current block
            curr_block_mask = (x[:, start_idx:end_idx] == mask_id)
            for j in range(x.shape[0]):
                block_indices = torch.where(curr_block_mask[j])[0] + start_idx
                x[j, block_indices] = x0[j, block_indices]
                
            intermediates.append(x.clone())
            
            # Update Running Max Confidence
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

# ==========================================
# 3. Main Script
# ==========================================

def main():
    print("Loading Model...")
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use LLaDAModelLM for compatibility with 'sample' (Identity Gating)
    # It works with generate_rcr too as it wraps/extends the HF model
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    mask_id = 126336
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to run")
    args = parser.parse_args()

    print("Loading MATH-500 Subset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.select(range(args.num_examples)) # User defined samples

    results = []
    
    # Parameters for Bohr-Kinetic Sampling
    asms_params = {
        "asms": True,
        "rcr": True, # ENABLE IONIZATION (Bohr-Kinetic)
        "elastic": True,
        "identity_gating": True,
        "beta_base": 0.8,
        "lambda_mom": 0.5,
        "h_peak": 0.1,
        "breakout_thresh": 0.85,
        "lambda_down": 1.5,
        "beta_up": 0.9,
        "semantic": True 
    }
    
    print("\nRunning Comparison: Trusted RCR vs Bohr-Kinetic")
    print("="*60)
    
    metrics = {"rcr": {"corr": 0, "flicker": 0}, "bohr_kinetic": {"corr": 0, "flicker": 0}}
    
    for idx, example in enumerate(tqdm(dataset)):
        question = example.get("problem", "")
        gold_answer = example.get("answer", "")
        if not gold_answer: gold_answer = example.get("solution", "")
        
        prompt_content = QUERY_TEMPLATE.format(Question=question)
        messages = [{"role": "user", "content": prompt_content}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        
        sample_res = {"id": idx, "question": question, "gold": gold_answer}
        
        # 1. Trusted RCR
        try:
            # start = time.time()
            out_rcr, f_rcr = generate_rcr(model, input_ids, mask_id, steps=256, gen_length=512, block_length=128)
            txt_rcr = tokenizer.decode(out_rcr[0, input_ids.shape[1]:], skip_special_tokens=True)
            pred_rcr = extract_answer_model_output(txt_rcr)
            corr_rcr = check_equality_heuristic(pred_rcr, gold_answer)
            
            sample_res["rcr"] = {"pred": pred_rcr, "corr": corr_rcr, "flicker": f_rcr, "txt": txt_rcr}
            metrics["rcr"]["corr"] += corr_rcr
            metrics["rcr"]["flicker"] += f_rcr
        except Exception as e:
            print(f"RCR Error: {e}")
            sample_res["rcr"] = {"error": str(e)}

        # 2. Bohr-Kinetic (Orbital ASMS)
        try:
            # start = time.time()
            # Aligned block_length to 128 to match RCR and allow Warm-up/Freeze cycle to work
            out_bk, inter_bk, _, _ = sample(
                model, input_ids, mask_id=mask_id, 
                steps=256, gen_length=512, block_length=128,
                return_intermediates=True,
                **asms_params
            )
            f_bk = calculate_flicker(inter_bk)
            txt_bk = tokenizer.decode(out_bk[0], skip_special_tokens=True)
            pred_bk = extract_answer_model_output(txt_bk)
            corr_bk = check_equality_heuristic(pred_bk, gold_answer)
            
            sample_res["bohr_kinetic"] = {"pred": pred_bk, "corr": corr_bk, "flicker": f_bk, "txt": txt_bk}
            metrics["bohr_kinetic"]["corr"] += corr_bk
            metrics["bohr_kinetic"]["flicker"] += f_bk
        except Exception as e:
            print(f"Bohr-Kinetic Error: {e}")
            sample_res["bohr_kinetic"] = {"error": str(e)}
            
        results.append(sample_res)
        
        # Live Update
        # print(f"Sample {idx}: RCR={corr_rcr} (F={f_rcr:.2f}) | Identity={corr_ig} (F={f_ig:.2f})")

    # Save
    with open("compare_rcr_identity.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nFinal Results (N={args.num_examples}):")
    print(f"RCR:      Acc={metrics['rcr']['corr']}/{args.num_examples}, Avg Flicker={metrics['rcr']['flicker']/args.num_examples:.2f}")
    print(f"Bohr-Kinetic: Acc={metrics['bohr_kinetic']['corr']}/{args.num_examples}, Avg Flicker={metrics['bohr_kinetic']['flicker']/args.num_examples:.2f}")

if __name__ == "__main__":
    main()

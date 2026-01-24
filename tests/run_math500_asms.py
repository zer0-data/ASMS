import torch
import argparse
import time
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
from asms import LLaDAModelLM, sample
from tqdm import tqdm

# --- MATH-500 Prompt & Extraction Logic (Adapted from Starter Code) ---

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"

def extract_answer_model_output(text):
    """
    Extracts the answer from the model output using the expected pattern.
    """
    match = re.search(ANSWER_PATTERN, text)
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(answer_str):
    """
    Simple normalization for math answers to facilitate exact match checking.
    Removes latex dollars, spaces, and handles basic numeric formatting.
    """
    if not answer_str:
        return ""
    # Remove latex delimiters
    clean = answer_str.replace('$', '').replace('\\', '')
    # Remove whitespace
    clean = "".join(clean.split())
    # Remove trailing punctuation often added by models
    clean = clean.strip('.')
    return clean

def check_equality_heuristic(predicted, gold):
    """
    Heuristic check for equality. 
    In a real benchmark, this might be an LLM call.
    """
    if predicted is None:
        return False
    
    norm_pred = normalize_answer(predicted)
    norm_gold = normalize_answer(str(gold)) # gold might be int/float
    
    return norm_pred == norm_gold

# --- Metric Utilities ---

def calculate_flicker(intermediate_results, check_last_n=16):
    """
    Calculates the average number of tokens that flipped in the last n steps.
    """
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

def save_summary_txt(args, agg_metrics, configurations):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"{args.name}_math500_summary_{timestamp}.txt"
    
    with open(txt_filename, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"ASMS MATH-500 Benchmark Summary\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("HYPERPARAMETER CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Experiment Name: {args.name}\n")
        f.write(f"Mode: {args.mode.upper()}\n")
        pass # Add more config details if needed
        f.write("\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20} | {'Accuracy (%)':<15} | {'Avg Flicker':<15} | {'Avg Time (s)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for cfg_name, metrics in agg_metrics.items():
            if metrics["total"] > 0:
                acc = (metrics["correct"] / metrics["total"]) * 100
                avg_flicker = metrics["flicker_sum"] / metrics["total"]
                avg_time = metrics["time_sum"] / metrics["total"]
                f.write(f"{cfg_name:<20} | {acc:<15.2f} | {avg_flicker:<15.2f} | {avg_time:<15.2f}\n")
            else:
                f.write(f"{cfg_name:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}\n")
        
        f.write("="*80 + "\n")
    
    return txt_filename

# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(description="Run ASMS on MATH-500 Benchmark")
    parser.add_argument("--mode", type=str, choices=["lcr", "rcr", "asms", "asms_elastic"], required=True)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--h_peak", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.85)
    
    # Elastic mode
    parser.add_argument("--beta_up", type=float, default=0.9)
    parser.add_argument("--lambda_down", type=float, default=1.5)
    
    # Kinetic-only
    parser.add_argument("--no_semantic", action="store_true")
    
    # Identity Gating
    parser.add_argument("--identity_gating", action="store_true", help="Enable strict Identity-Gated Momentum (Debounce)")
    
    parser.add_argument("--name", type=str, default="experiment")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to run (default: all)")
    parser.add_argument("--steps", type=int, default=64, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=512, help="Generation length (longer for math)")
    parser.add_argument("--block_length", type=int, default=32)
    
    args = parser.parse_args()

    # 1. Setup Model
    print("Loading Model and Tokenizer...")
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "./cache"
    
    try:
        model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, 
                                             cache_dir=cache_dir, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Dataset
    print("Loading MATH-500 Dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading dataset from HuggingFace: {e}")
        return

    if args.num_examples:
        dataset = dataset.select(range(min(args.num_examples, len(dataset))))

    # 3. Configure Method
    current_params = {"rcr": False, "asms": False}
    if args.mode == "rcr":
        current_params["rcr"] = True
    elif args.mode == "asms":
        current_params["asms"] = True
        current_params["beta_base"] = args.beta
        current_params["lambda_mom"] = args.lam
        current_params["h_peak"] = args.h_peak
        current_params["breakout_thresh"] = args.tau
        current_params["semantic"] = not args.no_semantic
    elif args.mode == "asms_elastic":
        current_params["asms"] = True
        current_params["elastic"] = True
        current_params["beta_base"] = args.beta
        current_params["lambda_mom"] = args.lam
        current_params["h_peak"] = args.h_peak
        current_params["breakout_thresh"] = args.tau
        current_params["beta_up"] = args.beta_up
        current_params["lambda_down"] = args.lambda_down
        current_params["lambda_down"] = args.lambda_down
        current_params["semantic"] = not args.no_semantic
        current_params["identity_gating"] = args.identity_gating

    configurations = [{"name": args.name, "params": current_params}]
    
    results_log = []
    agg_metrics = {cfg["name"]: {"correct": 0, "total": 0, "flicker_sum": 0.0, "time_sum": 0.0} for cfg in configurations}

    mask_id = 126336
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id

    print("\nStarting MATH-500 Benchmark...")
    
    for idx, example in enumerate(tqdm(dataset, desc="Benchmarking")):
        # Map fields
        # HuggingFaceH4/MATH-500 fields: problem, solution, answer, subject, level, unique_id
        # We need 'question' and 'gold_answer' (using 'answer' field for short answer if available, or 'solution'?)
        # Looking at user prompt: "problem, solution, answer, subject, level, unique_id"
        # Usually 'answer' is the short final answer, 'solution' is the full trace.
        
        question = example.get("problem", "")
        # Try to use the concise 'answer' field if it exists, otherwise might need extraction from solution
        # But wait, looking at the starter code, the gold is the full solution?
        # Starter code: "gold_answer = str(data_row.get('Answer', 'N/A'))"
        # And "Answer" in starter DataFrame (from kaggle) seemed to be the FULL solution.
        # But checking equality: "expression1: ... Expression 2: ..."
        # The starter code's equality checker can handle "Expression 1: $2$" vs "Expression 2: 2".
        # Let's assume 'answer' from HF dataset is the concise answer.
        gold_answer = example.get("answer", "")
        if not gold_answer:
            # Fallback to solution if answer is empty (unlikely for MATH)
            gold_answer = example.get("solution", "")

        # Prepare Prompt
        prompt_content = QUERY_TEMPLATE.format(Question=question)
        
        # Format for Chat Model
        messages = [{"role": "user", "content": prompt_content}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        
        sample_result = {"id": idx, "question": question, "gold_answer": gold_answer}

        for config in configurations:
            cfg_name = config["name"]
            params = config["params"]
            
            start_time = time.time()
            try:
                out, intermediate_results, _, _ = sample(
                    model, 
                    input_ids, 
                    mask_id=mask_id, 
                    steps=args.steps, 
                    gen_length=args.gen_length, 
                    block_length=args.block_length,
                    return_intermediates=True,
                    **params
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                
                # Extract Answer
                # The model *should* output "Answer: ..."
                prediction = extract_answer_model_output(generated_text)
                
                # Check correctness
                is_correct = check_equality_heuristic(prediction, gold_answer)
                
                avg_flicker = calculate_flicker(intermediate_results, check_last_n=16)
                
                sample_result[cfg_name] = {
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "time": elapsed_time,
                    "flicker": avg_flicker,
                    "full_output": generated_text
                }
                
                agg_metrics[cfg_name]["total"] += 1
                if is_correct:
                    agg_metrics[cfg_name]["correct"] += 1
                agg_metrics[cfg_name]["flicker_sum"] += avg_flicker
                agg_metrics[cfg_name]["time_sum"] += elapsed_time
                
            except Exception as e:
                print(f"Error in {cfg_name} for sample {idx}: {e}")
                sample_result[cfg_name] = {"error": str(e)}

        results_log.append(sample_result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{args.name}_results_{timestamp}.json"
    
    with open(json_filename, "w") as f:
        json.dump(results_log, f, indent=4)

    txt_file = save_summary_txt(args, agg_metrics, configurations)

    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Accuracy (%)':<15} | {'Avg Flicker':<15} | {'Avg Time (s)':<15}")
    print("-" * 80)
    for cfg_name, metrics in agg_metrics.items():
        if metrics["total"] > 0:
            acc = (metrics["correct"] / metrics["total"]) * 100
            avg_flicker = metrics["flicker_sum"] / metrics["total"]
            avg_time = metrics["time_sum"] / metrics["total"]
            print(f"{cfg_name:<20} | {acc:<15.2f} | {avg_flicker:<15.2f} | {avg_time:<15.2f}")
    print("="*80)
    print(f"Detailed results saved to {json_filename}")
    print(f"Summary saved to {txt_file}")

if __name__ == "__main__":
    main()

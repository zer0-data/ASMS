import torch
import argparse
import time
import json
import re
import numpy as np
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer
from asms import LLaDAModelLM, sample
from tqdm import tqdm

def extract_answer(text):
    """
    Extracts the answer after '####' in the text.
    """
    if "####" in text:
        return text.split("####")[1].strip().replace(",", "")
    return ""

def calculate_flicker(intermediate_results, check_last_n=16):
    """
    Calculates the average number of tokens that flipped in the last n steps.
    """
    if not intermediate_results or len(intermediate_results) < 2:
        return 0.0
    
    # intermediate_results is a list of tensors (B, L)
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

def save_summary_txt(args, agg_metrics, configurations):
    """
    Saves a summary txt file with hyperparameter config and metrics.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"{args.name}_summary_{timestamp}.txt"
    
    with open(txt_filename, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"ASMS Benchmark Summary\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Hyperparameter Config
        f.write("HYPERPARAMETER CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Experiment Name: {args.name}\n")
        f.write(f"Mode: {args.mode.upper()}\n")
        if args.mode == "asms":
            f.write(f"  - Beta (Momentum Decay): {args.beta}\n")
            f.write(f"  - Lambda (Momentum Weight): {args.lam}\n")
            f.write(f"  - H_Peak (Flicker Zone): {args.h_peak}\n")
            f.write(f"  - Tau (Breakout Threshold): {args.tau}\n")
        f.write("\n")
        
        # Results Summary
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["lcr", "rcr", "asms"], required=True)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--h_peak", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--name", type=str, default="experiment")
    args = parser.parse_args()

    # 1. Setup
    print("Loading Model and Tokenizer...")
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use cache_dir as seen in previous scripts
    cache_dir = "./cache"
    
    try:
        model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, 
                                             cache_dir=cache_dir, device_map=device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading GSM8K Dataset...")
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    subset = dataset.select(range(50)) # First 50 samples

    # Construct SINGLE config from CLI args
    current_params = {"rcr": False, "asms": False}
    
    if args.mode == "rcr":
        current_params["rcr"] = True
    elif args.mode == "asms":
        current_params["asms"] = True
        current_params["beta_base"] = args.beta
        current_params["lambda_mom"] = args.lam
        current_params["h_peak"] = args.h_peak
        current_params["breakout_thresh"] = args.tau

    configurations = [{"name": args.name, "params": current_params}]

    results_log = []
    
    # Metrics aggregators
    agg_metrics = {cfg["name"]: {"correct": 0, "total": 0, "flicker_sum": 0.0, "time_sum": 0.0} for cfg in configurations}

    print("\nStarting Battle Mode: ASMS vs Baselines...")
    mask_id = 126336 # Standard mask ID for LLaDA
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id

    for idx, example in enumerate(tqdm(subset, desc="Benchmarking")):
        question = example["question"]
        ground_truth = extract_answer(example["answer"])
        
        # Prepare Prompt
        messages = [{"role": "user", "content": question}]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        
        sample_result = {"id": idx, "ground_truth": ground_truth}

        for config in configurations:
            cfg_name = config["name"]
            params = config["params"]
            
            # Run Generation
            start_time = time.time()
            try:
                # Force return_intermediates=True for flicker measurement
                out, intermediate_results, _, _ = sample(
                    model, 
                    input_ids, 
                    mask_id=mask_id, 
                    steps=64, 
                    gen_length=128, 
                    block_length=32,
                    return_intermediates=True,
                    **params
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
                prediction = extract_answer(generated_text)
                
                # Check correctness
                # Simple loose checking implies filtering out non-numeric characters might be needed, 
                # but for GSM8K exact string match of the number is standard.
                # We'll normalize by removing formatting
                is_correct = (prediction == ground_truth)
                
                # Calculate Flicker
                avg_flicker = calculate_flicker(intermediate_results, check_last_n=16)
                
                # Log specific results
                sample_result[cfg_name] = {
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "time": elapsed_time,
                    "flicker": avg_flicker
                }
                
                # Update Aggregates
                agg_metrics[cfg_name]["total"] += 1
                if is_correct:
                    agg_metrics[cfg_name]["correct"] += 1
                agg_metrics[cfg_name]["flicker_sum"] += avg_flicker
                agg_metrics[cfg_name]["time_sum"] += elapsed_time
                
            except Exception as e:
                print(f"Error in {cfg_name} for sample {idx}: {e}")
                sample_result[cfg_name] = {"error": str(e)}

        results_log.append(sample_result)

    # Save detailed logs
    with open("results_comparison.json", "w") as f:
        json.dump(results_log, f, indent=4)

    # Save summary txt file with config and metrics
    txt_file = save_summary_txt(args, agg_metrics, configurations)

    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Accuracy (%)':<15} | {'Avg Flicker':<15} | {'Avg Time (s)':<15}")
    print("-" * 80)
    
    for cfg_name, metrics in agg_metrics.items():
        if metrics["total"] > 0:
            acc = (metrics["correct"] / metrics["total"]) * 100
            avg_flicker = metrics["flicker_sum"] / metrics["total"]
            avg_time = metrics["time_sum"] / metrics["total"]
            print(f"{cfg_name:<20} | {acc:<15.2f} | {avg_flicker:<15.2f} | {avg_time:<15.2f}")
        else:
             print(f"{cfg_name:<20} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
    print("="*80)
    print(f"Detailed results saved to results_comparison.json")
    print(f"Summary saved to {txt_file}")

if __name__ == "__main__":
    main()

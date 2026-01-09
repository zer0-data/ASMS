import torch
import argparse
import time
import json
import re
import os
import sys
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer
from asms import LLaDAModelLM, sample
from tqdm import tqdm

# Add current directory to path to import parser_helper if needed
# Assuming script is run from project root, tests/ is available?
# If run as python tests/run_countdown_asms.py, we can import from tests.parser_helper if root is in path or add tests to path
sys.path.append(os.path.join(os.getcwd(), 'tests'))

try:
    from parser_helper import remove_boxed, last_boxed_only_string
except ImportError:
    # Fallback if import fails (e.g. running from root without package structure)
    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return string

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]
        return retval

    def remove_boxed(s):
        if "\\boxed " in s:
            left = "\\boxed "
            return s[len(left) :]
        left = "\\boxed{"
        if s.startswith(left) and s.endswith("}"):
            return s[len(left) : -1]
        return s

# --- Countdown Prompt & Validation Logic (Adapted from tests/countdown.py) ---

CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. "
    "You may use the operations +, -, *, and / as needed, but each number must be used exactly once. "
    "Think step-by-step and provide your final expression inside \\boxed{} tags without including an equals sign or the target number. "
    "For example: \\boxed{a + b * c}"
)

def validate_countdown(generated_text, numbers, target):
    """
    Validates the generated equation against the numbers and target.
    """
    try:
        equation = remove_boxed(last_boxed_only_string(generated_text))
    except:
        # Try to extract from answer tags
        answer_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
        else:
            equation = generated_text
    
    if equation is None:
        return False, "No equation found"

    # Replace LaTeX operators with Python operators
    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")

    # Check for equation with equals sign and extract only the expression part
    # e.g. "3 + 4 = 7" -> "3 + 4"
    # But usually we want just the expression. If they put " = target", strip it.
    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()

    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            # Extract all numbers from the equation
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
            
            # Sort both lists to compare counts
            available_sorted = sorted(available_numbers)
            eq_sorted = sorted(numbers_in_eq)
            
            return available_sorted == eq_sorted
        except:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation."""
        try:
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                return float("Inf")
            # eval is safe-ish here due to regex check above, but still use restricted globals
            result = eval(equation_str.strip(), {"__builtins__": None}, {})
            return result
        except Exception:
            return float("Inf")

    is_valid_nums = validate_equation(equation, numbers)
    if not is_valid_nums:
        return False, f"Invalid numbers used. Expected {sorted(numbers)}, got tokens from {equation}"

    result = evaluate_equation(equation)
    if abs(result - target) < 1e-5:
        return True, f"Correct: {result} == {target}"
    
    return False, f"Wrong value: {result} != {target}"

# --- Metric Utilities ---

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

def save_summary_txt(args, agg_metrics, configurations):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"{args.name}_countdown_summary_{timestamp}.txt"
    with open(txt_filename, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"ASMS Countdown Benchmark Summary\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write("HYPERPARAMETER CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Experiment Name: {args.name}\n")
        f.write(f"Mode: {args.mode.upper()}\n")
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
    parser = argparse.ArgumentParser(description="Run ASMS on Countdown Benchmark")
    parser.add_argument("--mode", type=str, choices=["lcr", "rcr", "asms", "asms_elastic"], required=True)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--h_peak", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--beta_up", type=float, default=0.9)
    parser.add_argument("--lambda_down", type=float, default=1.5)
    parser.add_argument("--no_semantic", action="store_true")
    
    parser.add_argument("--name", type=str, default="countdown_exp")
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default="tests/datasets/countdown_cd3_test.jsonl")
    
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
    print(f"Loading Countdown Dataset from {args.dataset_path}...")
    dataset = []
    try:
        with open(args.dataset_path, "r") as f:
            for line in f:
                dataset.append(json.loads(line))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if args.num_examples:
        dataset = dataset[:args.num_examples]

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
        current_params["semantic"] = not args.no_semantic

    configurations = [{"name": args.name, "params": current_params}]
    
    results_log = []
    agg_metrics = {cfg["name"]: {"correct": 0, "total": 0, "flicker_sum": 0.0, "time_sum": 0.0} for cfg in configurations}

    mask_id = 126336
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id

    print("\nStarting Countdown Benchmark...")
    
    for idx, example in enumerate(tqdm(dataset, desc="Benchmarking")):
        # Fields: "input": "30,100,93", "output": "23"
        numbers_str = example["input"]
        target_str = example["output"]
        
        numbers = [int(n) for n in numbers_str.split(",")]
        target = int(target_str)
        
        question_text = f"Numbers: {numbers}\nTarget: {target}"
        
        # Prepare Prompt
        messages = [
            {"role": "system", "content": CTD_SYSTEM_PROMPT},
            {"role": "user", "content": question_text}
        ]
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
        
        sample_result = {"id": idx, "input": numbers, "target": target}

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
                
                # Check correctness
                is_correct, reason = validate_countdown(generated_text, numbers, target)
                
                avg_flicker = calculate_flicker(intermediate_results, check_last_n=16)
                
                sample_result[cfg_name] = {
                    "is_correct": is_correct,
                    "reason": reason,
                    "time": elapsed_time,
                    "flicker": avg_flicker,
                    "generated_text": generated_text
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
    json_filename = f"{args.name}_countdown_results_{timestamp}.json"
    
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

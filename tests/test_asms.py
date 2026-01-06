import torch
from transformers import AutoTokenizer
from asms import LLaDAModelLM, sample
import os

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    
    print("Loading model...")
    # Use cache_dir as seen in test_llada.py
    cache_dir = "./cache"
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, 
                                         cache_dir=cache_dir, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
    
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    m = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
    
    mask_id = 126336 # Standard mask ID for LLaDA
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id
        
    print(f"Running ASMS generation with mask_id={mask_id}...")
    try:
        out, _, _, _ = sample(model, input_ids, mask_id=mask_id, steps=64, gen_length=128, block_length=32, 
                                          asms=True, beta_base=0.8, h_peak=0.1, lambda_mom=0.5)
        
        generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print("-" * 50)
        print("Generated Output (ASMS):")
        print(generated_text)
        print("-" * 50)
        print("ASMS Generation Successful.")
    except Exception as e:
        print(f"Error during ASMS generation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ASMS Elastic Mode
    print(f"\nRunning ASMS Elastic Mode...")
    try:
        out, _, _, _ = sample(model, input_ids, mask_id=mask_id, steps=64, gen_length=128, block_length=32, 
                                          asms=True, elastic=True, beta_up=0.9, lambda_down=1.5,
                                          beta_base=0.8, h_peak=0.1, lambda_mom=0.5)
        
        generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print("-" * 50)
        print("Generated Output (ASMS Elastic):")
        print(generated_text)
        print("-" * 50)
        print("ASMS Elastic Generation Successful.")
    except Exception as e:
        print(f"Error during ASMS Elastic generation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ASMS Kinetic-Only (No Semantics)
    print(f"\nRunning ASMS Kinetic-Only (No Semantics)...")
    try:
        out, _, _, _ = sample(model, input_ids, mask_id=mask_id, steps=64, gen_length=128, block_length=32, 
                                          asms=True, semantic=False, beta_base=0.8, h_peak=0.1, lambda_mom=0.5)
        
        generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print("-" * 50)
        print("Generated Output (ASMS Kinetic-Only):")
        print(generated_text)
        print("-" * 50)
        print("ASMS Kinetic-Only Generation Successful.")
    except Exception as e:
        print(f"Error during ASMS Kinetic-Only generation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test ASMS Elastic + No Semantics
    print(f"\nRunning ASMS Elastic + Kinetic-Only...")
    try:
        out, _, _, _ = sample(model, input_ids, mask_id=mask_id, steps=64, gen_length=128, block_length=32, 
                                          asms=True, elastic=True, semantic=False,
                                          beta_up=0.9, lambda_down=1.5, beta_base=0.8, h_peak=0.1, lambda_mom=0.5)
        
        generated_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print("-" * 50)
        print("Generated Output (ASMS Elastic + Kinetic-Only):")
        print(generated_text)
        print("-" * 50)
        print("ASMS Elastic + Kinetic-Only Generation Successful.")
    except Exception as e:
        print(f"Error during ASMS Elastic + Kinetic-Only generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

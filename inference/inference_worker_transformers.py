import os
import json
import sqlite3
import torch
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
import os
from peft import PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REASONING = False
TEMPERATURE = 1.0

def get_args():
    parser = argparse.ArgumentParser(description="vLLM Slurm Worker")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for generation")
    parser.add_argument("--split_id", type=int, required=True, help="Current split ID")
    parser.add_argument("--num_splits", type=int, required=True, help="Total number of splits")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    
    #quantization
    parser.add_argument("--quantization", type=str, default=None, choices=["bitsandbytes"])
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32", "bfloat32"], help="Initial datatype for weights")
    return parser.parse_args()

def claim_batch(conn, args, batch_size):
    conn.execute("BEGIN IMMEDIATE")
    try:
        cursor = conn.cursor()
        job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
        worker_sig = f"proc_{job_id}_{task_id}"
        
        print(f"DEBUG: Searching for model_name: '{args.model_name}', seed: {args.seed}, split: {args.split_id}")
        #select the correct values from the db
        cursor.execute("""
            UPDATE tasks 
            SET status = ?
            WHERE rowid IN (
                SELECT rowid FROM tasks 
                WHERE model_name = ? 
                AND seed = ? 
                AND (global_index % ?) = ?
                AND status = 'pending' 
                LIMIT ?
            )
        """, (
            worker_sig,
            args.model_name,
            args.seed,
            args.num_splits,
            args.split_id,
            batch_size
        ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    
    query = """
        SELECT rowid, * FROM tasks 
        WHERE status = ?
        ORDER BY global_index
    """
    return pd.read_sql(query, conn, params=(worker_sig,))

def main():
    current_node = os.environ.get('SLURMD_NODENAME')
    print(f"Running on model: {current_node}")
    args = get_args()
    
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        
    torch.manual_seed(args.seed)
    set_seed(args.seed)

    db_root = cfg.get("db_root", "databases")
    
    safe_name = args.model_name.replace("/", "_").replace(".", "_")
    
    db_path = os.path.join(db_root, f"{safe_name}.db")
    db_path = os.path.expanduser(os.path.expanduser(db_path))
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} was not found! Run setup first.")
    
    print(f"[{args.split_id}] Target DB: {db_path}")
    print(f"[{args.split_id}] Loading model: {args.model_path}")
    
    quant_config = None
    if args.quantization == "bitsandbytes":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
        device_map="cuda",
        trust_remote_code=True
    )
    
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"Applying lora adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    else:
        print(f"No adapter found or provided. Using base model.")
        model = base_model
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # if tokenizer.padding_side == "left":
        # if tokenizer.pad_token is None:
            # tokenizer.pad_token = tokenizer.eos_token
    if "llama" in args.model_name:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    
    model = torch.compile(model)
        
    conn = sqlite3.connect(db_path, timeout=120)
    tokenizer.pad_token = tokenizer.eos_token
    processed_count = 0
    
    while True:
        df_batch = claim_batch(conn, args, cfg.get("batch_size", 50))
        if df_batch.empty:
            print(f"[{args.split_id}] No more pending tasks. Shutting down.")
            break
        
        print(f"[{args.split_id}] Processing batch of {len(df_batch)} prompts")
        
        if "qwen" in args.model_name: 
            prompts = [
                tokenizer.apply_chat_template([
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]}
                ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for _, row in df_batch.iterrows()
            ]
        elif "base" in args.model_name.split("_")[0] and "base" in args.model_name.split("_")[1]:
            prompts = [
                f"System: {row['system_prompt']}\nUser: {row['user_prompt']}\nAssistant:"
                for _, row in df_batch.iterrows()
            ]
            
        else: 
            prompts = [
                tokenizer.apply_chat_template([
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]}
                ], tokenize=False, add_generation_prompt=True)
                for _, row in df_batch.iterrows()
            ]

        with torch.no_grad():
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=cfg.get("max_token_count", 512),
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        results = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        update_data = []
        for i, output in enumerate(results):
            db_row_id = int(df_batch.iloc[i]["rowid"])
            update_data.append((output, db_row_id))
        
        # update_data = [(res, int(row['rowid'])) for res, (_, row) in zip(results, df_batch.iterrows())]
        conn.cursor().executemany("""
            UPDATE tasks SET output = ?,
            status = 'completed'
            WHERE rowid = ?
        """, update_data)
        conn.commit()
        
        processed_count += len(df_batch)
        print(f"[{args.split_id}] Total completed by this worker: {processed_count}")
        
        del inputs
        del output_tokens
        torch.cuda.empty_cache()

    conn.close()
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    print("Finished worker task!")
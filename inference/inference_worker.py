import os
import json
import sqlite3
import torch
import argparse
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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
    args = get_args()
    
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    db_root = cfg.get("db_root", "databases")
    
    safe_name = args.model_name.replace("/", "_").replace(".", "_")
    
    db_path = os.path.join(db_root, f"{safe_name}.db")
    db_path = os.path.expanduser(os.path.expanduser(db_path))
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} was not found! Run setup first.")
    
    print(f"[{args.split_id}] Target DB: {db_path}")
    print(f"[{args.split_id}] Loading model: {args.model_path}")
    
    tokenizer_path = args.lora_path if args.lora_path else args.model_path
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        quantization=args.quantization,
        dtype=args.dtype,
        tokenizer=tokenizer_path,
        enable_lora=(args.lora_path is not None),
        max_lora_rank=cfg.get("max_lora_rank", 64)
        )
    tokenizer = llm.get_tokenizer()
    
    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest(lora_name="lora", lora_int_id=1, lora_path=args.lora_path)
    
    sampling_params = SamplingParams(
        seed=args.seed,
        temperature=TEMPERATURE,
        max_tokens=cfg.get("max_token_count", 512)
    )

    conn = sqlite3.connect(db_path, timeout=120)
    processed_count = 0

    #keep pulling batches until the split is empty
    while True:
        df_batch = claim_batch(conn, args, cfg.get("batch_size", 50))
        
        if df_batch.empty:
            print(f"[{args.split_id}] No more pending tasks. Shutting down.")
            break
        
        print(f"[{args.split_id}] Processing batch of {len(df_batch)} prompts")
        
        #create chat-template
        formatted_prompts = []
        for _, row in df_batch.iterrows():
            messages = [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["user_prompt"]}
            ]

            # formatted_prompts.append(messages)     
            
            if "qwen" in args.model_name:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                formatted_prompts.append(prompt_text)
            else:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted_prompts.append(prompt_text)
        
        outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

        update_data = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            db_row_id = int(df_batch.iloc[i]['rowid'])
            # g_idx = int(df_batch.iloc[i]['global_index'])
            # update_data.append((generated_text, g_idx, args.model_path, args.seed))
            update_data.append((generated_text, db_row_id))

        cursor = conn.cursor()
        cursor.executemany("""
            UPDATE tasks 
            SET output = ?, status = 'completed' 
            WHERE rowid = ?
        """, update_data)
        conn.commit()
        
        processed_count += len(df_batch)
        print(f"[{args.split_id}] Total completed by this worker: {processed_count}")

    conn.close()
    del llm
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    print("Finished worker task!")
import sqlite3
import json
import os
import csv
import pandas as pd

def export_databases_to_csv(db_dir=".", output_dir="exports"):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(db_dir):
        if not filename.endswith(".db"):
            continue

        db_path = os.path.join(db_dir, filename)
        
        try:
            conn = sqlite3.connect(db_path)

            query = """
                SELECT global_index, input_text, system_prompt, user_prompt, 
                       ground_label, model_name, seed, task_lang, instance_lang, context, lora_path, quantization, output
                FROM tasks
            """
            df = pd.read_sql_query(query, conn)
            print(f"Processing {filename}: Found {len(df)} rows")
            conn.close()
            
            if df.empty:
                print(f"No data found in {filename}.db")
                conn.close()
                continue

            files_to_write = {}

            cols_to_clean = ["input_text", "system_prompt", "user_prompt", "output"]
            for col in cols_to_clean:
                df[col] = df[col].astype(str).str.replace(r"[\n\r]+", " ", regex=True)
                
            df['lora_path'] = df['lora_path'].fillna('')
            df['quantization'] = df['quantization'].fillna('')
            
            for (seed, t_lang, i_lang, context, m_name, lora, quantization), group_df in df.groupby(['seed', 'task_lang', 'instance_lang', 'context', 'model_name', 'lora_path', 'quantization']):
                            
                q_tag = "_quantized" if quantization and str(quantization).strip() else ""
                m_safe = str(m_name).replace("_quantized", "")
                m_safe = str(m_safe).replace("/", "_").replace(".", "_")
                if q_tag != "":
                    m_safe = str(m_safe).replace("_", "_quantized_", 1)
                csv_filename = f"{m_safe}_context_{context}_task_{t_lang}_instance_{i_lang}.csv"
                            
                seed_folder = f"seed{seed}"
                seed_path = os.path.join(output_dir, seed_folder)
                if not os.path.exists(seed_path):
                    os.makedirs(seed_path)

                output_cols = ["global_index", "input_text", "system_prompt", "user_prompt", "ground_label", "output", "model_name"]
                final_path = os.path.join(seed_path, csv_filename)
                            
                group_df[output_cols].to_csv(
                    final_path, 
                    index=False, 
                    sep=';', 
                    quoting=1,
                    encoding='utf-8-sig'
                )
                            
                print(f"Exported: (Seed: {seed}) -> {csv_filename}")

        except Exception as e:
            print(f"Error processing {db_path}: {e}")

if __name__ == "__main__":
    export_databases_to_csv(db_dir="./databases", output_dir="./csv_exports")
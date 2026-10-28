import sqlite3
import os
import json
import hashlib
import yaml
import pandas as pd
from pathlib import Path
from datasets import Dataset

def load_prompt_text(context_lang, task_lang, input_lang=None, prompt_id='bias_prompt_v1'):
    input_lang = task_lang
    task_context_pair = task_lang + "_" + context_lang
    
    with open("<ANONYMIZED-PATH>/prompts/prompt_components.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    prompt_group = next((p for p in data['prompts'] if p['id'] == prompt_id), None)
    if not prompt_group:
        return "Prompt ID not found."

    version = next((v for v in prompt_group['versions'] if v['config']['language'] == task_lang), None)
    if not version:
        return f"Task language version '{task_lang}' not found."

    try:
        content_key = "driving_stereotype"
        country_name = data['content_library'][content_key]["example_country_names"][task_context_pair]
        
        first_shot_input = data['content_library'][content_key]["girls_like_pink_stereotype"][input_lang]
        first_reason_trace = data['content_library'][content_key]["girls_like_pink_reason_trace"][input_lang].format(country_placeholder=country_name)
        
        second_shot_input = data['content_library'][content_key]["boys_like_pink_stereotype"][input_lang]
        second_reason_trace = data['content_library'][content_key]["boys_like_pink_reason_trace"][input_lang].format(country_placeholder=country_name)
        
        proper_context_pair = data['content_library']["context_types"][task_context_pair]
    except KeyError:
        return f"Input language '{input_lang}' not found in Content Library."

    template = version['template']
    
    part_0 = template['task'].format(context_language=proper_context_pair)
    part_1 = template['explanation_part_1'].format(first_example=first_shot_input, first_example_reasoning_trace=first_reason_trace)
    part_2 = template['explanation_part_2'].format(second_example=second_shot_input, second_example_reasoning_trace=second_reason_trace)
    
    
    full_prompt = (
        f"{part_0}\n"
        f"{part_1}\n"
        f"{part_2}"
    )
    
    return full_prompt.strip()

def load_biasshades_data(task_language: str, context: str, input_language=None):
    input_language = task_language
    
    df = pd.read_csv("<ANONYMIZED-PATH>", converters={"stereotype_valid_langs": eval}, encoding="utf-8")
        
    system_prompt = load_prompt_text(context, task_language, input_language)

    user_prefix = "Tekst:" if task_language == "nl" else "Text:"
    
    is_stereotype_mask = df["stereotype_valid_langs"].apply(lambda x: context in x)
    column_name = f"{input_language}_biased_sentences"
    
    def build_rows(sentences, label_str, label_int):
        processed_rows = []
        for t in sentences:
            if not isinstance(t, str):
                print(f"Skipping invalid data: {t} (Type: {type(t)})")
                continue
                
            processed_rows.append({
                "id": hashlib.md5(t.encode("utf-8")).hexdigest(),
                "text": t,
                "system": system_prompt,
                "user": f"{user_prefix} {t}",
                "label": label_str,
                "label_int": label_int
            })
        return processed_rows
        
    stereotypes = df[is_stereotype_mask][column_name].dropna().tolist()
    non_stereotypes = df[~is_stereotype_mask][column_name].dropna().tolist()
    
    all_data = (
        build_rows(stereotypes, "stereotype", 1) +
        build_rows(non_stereotypes, "nonstereotype", 0)
    )
    
    return pd.DataFrame(all_data)

def setup_database(config_path):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
        
    db_dir = "databases"
    os.makedirs(db_dir, exist_ok=True)
    
    task_langs = cfg.get("TASK_Language", ["en"])
    if isinstance(task_langs, str): task_langs = [task_langs]
    
    # instance_langs = cfg.get("Instance_language", ["en"])
    # if isinstance(instance_langs, str): instance_langs = [instance_langs]
    
    context_langs = cfg.get("context", ["en"])
    if isinstance(context_langs, str): context_langs = [context_langs]

    models = cfg['models']
    seeds = cfg['seeds']
    
    for model_name, model_info in models.items():
        safe_name = model_name.replace("/", "_").replace(".", "_")
        db_path = os.path.join(db_dir, f"{safe_name}.db")
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS tasks")
        cursor.execute("""
            CREATE TABLE tasks (
                global_index INTEGER,
                model_name TEXT,
                id TEXT,
                input_text TEXT,
                system_prompt TEXT,
                user_prompt TEXT,
                ground_label TEXT,
                path TEXT,
                quantization TEXT,
                lora_path TEXT,
                dtype TEXT,
                seed INTEGER,
                status TEXT,
                output TEXT,
                task_lang TEXT,
                instance_lang TEXT,
                context TEXT
            )
        """)

        print(f"Generating rows for {model_name}")
        model_data = []
        for context in context_langs:
            for t_lang in task_langs:
                try:
                    data_df = load_biasshades_data(t_lang, context)
                except Exception as e:
                    print(f"Skipping pair {t_lang}-{context} due to error {e}")
                    continue
                
                for seed in seeds:
                    for idx, row in data_df.iterrows():
                        model_data.append((
                            idx,  #chunk index
                            model_name,      #model name
                            row['id'],       #sample id
                            row['text'],     #sample text
                            row['system'],   #system example prompt
                            row['user'],     #user prompt
                            row['label'],    #text label (setereotype or nonstereotype)
                            model_info['path'], 
                            model_info['quantization'], 
                            model_info['lora_path'], 
                            model_info['dtype'], 
                            seed, 
                            'pending',       #status
                            None,             #inference output
                            t_lang,
                            t_lang,
                            context
                        ))


        print(f"Inserting {len(model_data)} task rows into {db_path}...")
        cursor.executemany("INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", model_data)
        cursor.execute("CREATE INDEX idx_worker_lookup ON tasks (model_name, seed, status, global_index)")
        
        conn.commit()
        conn.close()
        
    print("All databases initialized in ./databases/")

if __name__ == "__main__":
    setup_database(config_path="<ANONYMIZED-PATH>/inference/inference_config.json")
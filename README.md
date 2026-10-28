# Cultural (In)Competence: Investigating Culture-Specific Stereotype Detection Capabilities of LLMs

Code for the paper "[Cultural (In)Competence: Investigating Culture-Specific Stereotype Detection Capabilities of LLMs](https://aclanthology.org/2026.findings-emnlp.426/)".

For details on the approach, architecture and idea, please see the published paper.


```
@inproceedings{spliethover-etal-2026-cultural,
    title = "Cultural (In)Competence: Investigating Culture-Specific Stereotype Detection Capabilities of {LLM}s",
    author = {Splieth{\"o}ver, Maximilian  and L{\"o}tfering, Jannes and Bachmann, Dominik and Espig, Alexander and Zuidema, Willem and Schulz, Katrin and Wachsmuth, Henning},
    editor = "Lee, Hung-Yi and Dev, Sunipa and Li, Jessy and Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.426/",
    pages = "8964--8983",
    ISBN = "979-8-89176-462-0",
}
```


## Reproducing the Results

The pipeline has four steps: (1) prepare the instruction-tuning data, (2) instruction-tune the models with QLoRA, (3) run inference on SHADES, and (4) evaluate.

> [!NOTE]
> All experiments were conducted using Python 3.10.18


### 0. Setup

Install the dependencies for data preparation, inference, and evaluation:

```shell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Instruction-tuning uses a separate environment. Follow the installation instructions in [`instruction-tuning/alignment-handbook/README.md`](instruction-tuning/alignment-handbook/README.md#installation-instructions) (PyTorch, `pip install .`, and Flash Attention 2).

Download the base models into `instruction-tuning/alignment-handbook/models/`:

| Directory                 | Hugging Face model                                                                                    |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `Llama-3.1-8B-Instruct`   | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)           |
| `Llama-3.1-8B`            | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)                             |
| `Mistral-7B-Instruct-v0.3`| [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)       |
| `Mistral-7B-v0.3`         | [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)                         |
| `Qwen3-8B`                | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)                                                 |

Download SHADES ([LanguageShades/BiasShades](https://huggingface.co/datasets/LanguageShades/BiasShades)) and store it as `data/shades/all/all.csv`.

### 1. Data Preparation

The training configs expect the instruction-tuning datasets in `instruction-tuning/alignment-handbook/data/`:

| Directory                    | Language | Source                                                                                                                                  |
| ---------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `ultrachat_200k`             | en       | [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)                                            |
| `ultrachat_400k_nl`          | nl       | [ReBatch/ultrachat_400k_nl](https://huggingface.co/datasets/ReBatch/ultrachat_400k_nl)                                      |
| `no_robots`                  | en       | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)                                                      |
| `no_robots_dutch`            | nl       | [BramVanroy/no_robots_dutch](https://huggingface.co/datasets/BramVanroy/no_robots_dutch)                                                |
| `stackoverflow-chat-dutch`   | nl       | [BramVanroy/stackoverflow-chat-dutch](https://huggingface.co/datasets/BramVanroy/stackoverflow-chat-dutch)                              |
| `stackoverflow-en`           | en       | `stackoverflow_chat_data.json` from [Baize](https://github.com/project-baize/baize-chatbot), converted with `prepare-stackoverflow-en.py` |
| `belle-0.5m-zh`              | zh       | `Belle_open_source_0.5M.json` from [BelleGroup/train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN), converted with `prepare-belle-zh.marimo.py` |
| `subsets/ultrachat_*-subsample__*k` | en/nl | Created with `dataset-subsampling.py` (for H3)                                                                                    |

The preparation scripts are [marimo](https://marimo.io) notebooks. Open them with `marimo edit <file>`, or run them as scripts with `python <file>`. They read their raw input from `./data/` and write the prepared Parquet splits to `./output/`. Move the results to the alignment-handbook `data/` directory afterwards.

To create the UltraChat subsets for H3, set `dataset_name` in `dataset-subsampling.py` to `ultrachat_200k` (English) or `ultrachat_400k_nl` (Dutch) and run it. Add `200000` to `target_sizes` to also create the 200k Dutch subset.

### 2. Instruction-Tuning

All models are fine-tuned with QLoRA using the SFT script of the alignment-handbook. Each model family has a training script that takes the dataset name as its only argument and selects the matching YAML config:

```shell
cd instruction-tuning/alignment-handbook

# Llama 3.1 8B Instruct
bash train-llama.sh ultrachat-200k-en
bash train-llama.sh ultrachat-400k-nl
bash train-llama.sh norobots-en
bash train-llama.sh norobots-nl
bash train-llama.sh stackoverflow-en
bash train-llama.sh stackoverflow-nl
bash train-llama.sh belle-05m-zh

# Data-size experiments (H3, Llama 3.1 8B Instruct only)
bash train-llama.sh ultrachat-200k-subsample__10k-en   # also: __50k-en, __100k-en
bash train-llama.sh ultrachat-400k-subsample__10k-nl   # also: __50k-nl, __100k-nl, __200k-nl
```

The other model families use the same dataset names:

| Script                 | Model                     |
| ---------------------- | ------------------------- |
| `train-llama.sh`       | Llama 3.1 8B Instruct     |
| `train-llamabase.sh`   | Llama 3.1 8B (base)       |
| `train-mistral.sh`     | Mistral 7B Instruct v0.3  |
| `train-mistralbase.sh` | Mistral 7B v0.3 (base)    |
| `train-qwen.sh`        | Qwen3 8B                  |

Run a script without arguments to see the datasets it supports. The scripts launch training with `accelerate` on **2 GPUs** using DDP (`recipes/accelerate_configs/ddp.yaml`). To use a different number of GPUs, change `--num_processes` in the script. You may then also want to adjust `per_device_train_batch_size` and `gradient_accumulation_steps` in the YAML config to keep the same effective batch size. The LoRA adapters are written to `output/<model>-<dataset>/`.

### 3. Inference

Inference prompts every model with a few-shot stereotype detection prompt for each combination of stereotype context (nl/en/zh) and prompt language (nl/en/zh), using 5 seeds (1, 5, 15, 25, 35) and temperature 1.0. Tasks are tracked in one SQLite database per model, so that multiple workers can process them in parallel and interrupted runs can resume.

1. **Configure** the models in `inference/inference_config.json`: set `path` to the base model and `lora_path` to the adapter from step 2 (or `null` for the model without additional tuning). `inference/inference_config_all.json` lists all fine-tuned models from the paper.
2. **Create the task databases:**
   ```shell
   python inference/inference_database_setup.py
   ```
3. **Run inference**, either as a Slurm array job or directly:
   ```shell
   python inference/inference_worker.py \
       --config inference/inference_config.json \
       --model_name llama31_ultrachat_en \
       --model_path <path-to-base-model> \
       --lora_path <path-to-lora-adapter> \
       --quantization bitsandbytes \
       --seed 1 --split_id 0 --num_splits 1
   ```
   `inference_worker_transformers.py` takes the same arguments and uses HF transformers instead of vLLM.
4. **Monitor progress** with `python inference/inference_status.py`.
5. **Export the outputs** to CSV with `python inference/inference_export.py`. This writes one file per model and setting to `csv_exports/seed<N>/`.

### 4. Evaluation

The evaluation notebooks read the SHADES data from `data/shades/all/all.csv` and the model predictions from:

- `output/model-predictions/shades/seed<N>/` (H1, H2): the CSV exports from step 3
- `output/model-predictions/subsampling/seed<N>/` (H3): the CSV exports of the data-size models

Copy the exported CSV files to these directories. The notebooks extract the yes/no prediction from each model output, compute F1, precision, and recall per seed, run Wilcoxon signed-rank tests, and save the figures to `output/evaluation/`:

```shell
marimo edit evaluation_h01.marimo.py   # interactive
python evaluation_h01.marimo.py        # or run as a script
```

The models and settings to evaluate are set in the *Setup* cell of each notebook.

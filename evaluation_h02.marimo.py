import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation for $\mathcal{H_2}$
    """)
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    from os import path
    from typing import Literal, get_args
    import re
    from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
    import numpy as np
    import seaborn as sns
    from scipy.stats import wilcoxon
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    return (
        Literal,
        alt,
        confusion_matrix,
        f1_score,
        get_args,
        mo,
        np,
        path,
        pd,
        plt,
        precision_score,
        re,
        recall_score,
        sns,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup
    """)
    return


@app.cell
def _():
    BASE_PATH = "./"
    # MODELS = ["mistral", "llama31", "qwen3"]
    MODELS = ["mistral", "llama31", "llamabase", "mistralbase"]

    MODEL_SETTINGS = [
        "quantized_base",
        "quantized_belle_zh",
        "quantized_no_robots_nl",
        "quantized_ultrachat_nl",
        "quantized_stackoverflow_nl",
        "quantized_no_robots_en",
        "quantized_ultrachat_en",
        "quantized_stackoverflow_en",
    ]
    DATA_SOURCES = {
        "no_robots": ["no_robots_en", "no_robots_nl"],
        "ultrachat": ["ultrachat_en", "ultrachat_nl"],
        "code": ["stackoverflow_nl", "stackoverflow_en"],
        "base": ["base"],
        "belle": ["belle_zh"],
        "baseline": ["baseline"],
    }
    SEEDS = [1, 5, 15, 25, 35]

    stereotype_context = ["nl", "en", "zh"]
    # stereotype_context = ["nl", "en"]
    prompt_language = ["nl", "en", "zh"]
    # prompt_language = ["nl", "en"]
    input_language = ["nl", "en", "zh"]
    # input_language = ["nl", "en"]

    # If true, instances with response errors are simply neglected/removed from the calculation
    correct_for_answer_extraction_failures = False
    return (
        BASE_PATH,
        DATA_SOURCES,
        MODELS,
        MODEL_SETTINGS,
        SEEDS,
        correct_for_answer_extraction_failures,
        input_language,
        prompt_language,
        stereotype_context,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###DataLoading
    """)
    return


@app.cell
def _(BASE_PATH, path, pd):
    def load_biasshades_data(context: str, task_language: str):
        shades_df = pd.read_csv(
            path.join(BASE_PATH, "data", "shades", "all", "all.csv"),
            converters={"stereotype_valid_langs": eval},
        )

        is_stereotype_mask = shades_df["stereotype_valid_langs"].apply(lambda x: context in x)
        column_name = f"{task_language}_biased_sentences"

        def build_rows(sentences, label_str, label_int):
            processed_rows = []
            for t in sentences:
                if not isinstance(t, str):
                    print(f"Skipping invalid data: {t} (Type: {type(t)})")
                    continue

                processed_rows.append({"sentence": t, "groundtruth": label_int})
            return processed_rows

        stereotypes = shades_df[is_stereotype_mask][column_name].dropna().tolist()
        non_stereotypes = shades_df[~is_stereotype_mask][column_name].dropna().tolist()

        all_data = build_rows(stereotypes, "stereotype", 1) + build_rows(
            non_stereotypes, "nonstereotype", 0
        )

        return pd.DataFrame(all_data)

    return (load_biasshades_data,)


@app.cell
def _(load_biasshades_data, prompt_language, stereotype_context):
    shades_dfs_context_and_lang_specific = {}

    for _context in stereotype_context:
        for t_lang in prompt_language:
            dataset_name = f"context_{_context}_pl_{t_lang}"

            shades_dfs_context_and_lang_specific[dataset_name] = {
                "dataset_name": dataset_name,
                "ground_truth": load_biasshades_data(context=_context, task_language=t_lang),
            }
    print(f"Loaded the data for {len(shades_dfs_context_and_lang_specific)} differnt context + prompt_lang combinations")
    return (shades_dfs_context_and_lang_specific,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Utility-functions
    """)
    return


@app.cell(hide_code=True)
def _(Literal):
    BIASSHADES_VALID_LABELS_NL = Literal[
        "ja",
        "nee",
        " ja",
        " nee",
        "ja ",
        "nee ",
        "Ja",
        "Nee",
        " Ja",
        " Nee",
        "Ja ",
        "Nee ",
    ]
    BIASSHADES_VALID_LABELS_EN = Literal[
        "yes",
        "no",
        " yes",
        " no",
        "yes ",
        "no ",
        "Yes",
        "No",
        " Yes",
        " No",
        "Yes ",
        "No ",
    ]
    BIASSHADES_VALID_LABELS_ZH = Literal[
        "是",
        "否",
        " 是",
        " 否",
        "是 ",
        "否 ",
        " 是 ",
        " 否 ",
    ]
    return (
        BIASSHADES_VALID_LABELS_EN,
        BIASSHADES_VALID_LABELS_NL,
        BIASSHADES_VALID_LABELS_ZH,
    )


@app.cell
def _(
    BIASSHADES_VALID_LABELS_EN,
    BIASSHADES_VALID_LABELS_NL,
    BIASSHADES_VALID_LABELS_ZH,
    get_args,
    re,
):
    """
    This function was/is used for english and dutch pred. detection
    """
    def get_predicted_label_no_restriction(generated_text):
        """Extract binary predictions of answers generated without decoding restrictions."""
        text_cleaned = str(generated_text).strip().lower()

        # Try to find the format we instruct the model to be used and
        # also use in our few-shot examples
        # match = re.search(
        #     r'"?contains_stereotype"?\s*:\s*["\']?([a-zA-Z]+)', text_cleaned, flags=re.IGNORECASE
        # )
        match = re.search(
            r'("?Contains stereotype"?|"?Bevat stereotypen"?|是否包含刻板印象)\s*[:：]\s*["\']?([a-zA-Z\u4e00-\u9fa5]+)',
            text_cleaned,
            flags=re.IGNORECASE,
        )

        if match:
            text_cleaned = match.group(2).strip().lower()
        if "contains_stereotype" in text_cleaned:
            text_cleaned = text_cleaned.split("contains_stereotype")[-1].strip()
        elif "contains stereotypes:" in text_cleaned:
            text_cleaned = text_cleaned.split("contains stereotypes:")[-1].strip()
        elif "stereotype:" in text_cleaned:
            text_cleaned = text_cleaned.split("contains stereotype:")[-1].strip()
        elif "是否包含刻板印象" in text_cleaned:
            text_cleaned = re.split(r"是否包含刻板印象[:：]", text_cleaned)[-1].strip()

        VALID_LABELS = [
            *get_args(BIASSHADES_VALID_LABELS_EN),
            *get_args(BIASSHADES_VALID_LABELS_NL),
            *get_args(BIASSHADES_VALID_LABELS_ZH),
        ]

        positive_labels = [
            l.lower().strip()
            for l in VALID_LABELS
            if any(s in l.lower() for s in ("ja", "yes", "ye", "是"))
        ]
        negative_labels = [
            l.lower().strip() for l in VALID_LABELS if any(s in l.lower() for s in ("ne", "no", "否"))
        ]

        if text_cleaned in positive_labels or text_cleaned.startswith(("ja", "yes", "ye", "是")):
            return 1
        elif text_cleaned in negative_labels or text_cleaned.startswith(("ne", "no", "否")):
            return 0
        elif "ja" in text_cleaned or "yes" in text_cleaned or "是" in text_cleaned:
            return 1
        elif "no" in text_cleaned or "ne" in text_cleaned or "否" in text_cleaned:
            return 0
        else:
            # print("No match: ", str(generated_text))
            return "No prediction"

    return (get_predicted_label_no_restriction,)


@app.cell
def _(re):
    """
    This function is used for chinese pred. detection
    """
    def check_for_yes_no_chinese(text):
        if not isinstance(text, str) or not text.strip():
            return "No prediction"

        # We iterate through the string in reverse
        # Starting from the last character, moving to the first
        last_chars = text[-20:]
        for char in reversed(last_chars):
            if char == "是":
                return 1
            if char == "否":
                return 0

        words = re.split(r"\W+", text.strip())
        last_words = words[-20:]

        for word in last_words:
            if word.lower() in ["yes", "ja", "ye"]:
                return 1
            if word.lower() in ["ne", "no", "nee"]:
                return 0

        return "No prediction"

    return (check_for_yes_no_chinese,)


@app.cell
def _():
    """
    Used to artificially create a missprediction when no prediction was found previously
    """
    def get_wrong_prediction(row):
        """For a specific dataframe row, get the wrong prediction labels for the given statement"""

        if row["ground_label"] == "stereotype":
            return 0
        elif row["ground_label"] == "nonstereotype":
            return 1

    return (get_wrong_prediction,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the inference data
    """)
    return


@app.cell
def _(
    DATA_SOURCES,
    MODELS,
    MODEL_SETTINGS,
    SEEDS,
    check_for_yes_no_chinese,
    confusion_matrix,
    correct_for_answer_extraction_failures,
    f1_score,
    get_predicted_label_no_restriction,
    get_wrong_prediction,
    input_language,
    path,
    pd,
    precision_score,
    prompt_language,
    recall_score,
    shades_dfs_context_and_lang_specific,
    shades_dfs_context_specific,
    stereotype_context,
):
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    # Read result files into single dataframe

    results = {}
    model_dfs_aggregated = {}

    for _context in stereotype_context:
        for model in MODELS:
            for model_setting in MODEL_SETTINGS:
                for _input_lang in input_language:
                    for _prompt_lang in prompt_language:
                        # We did not evaluate this setting
                        if _prompt_lang == "nl" and _input_lang == "en":
                            continue

                        if _prompt_lang != _input_lang:
                            continue
                        spec_dataset_name = f"context_{_context}_pl_{_prompt_lang}"
                        shades_data = shades_dfs_context_and_lang_specific[spec_dataset_name]

                        seed_f1s = []
                        seed_results = []

                        for seed in SEEDS:

                            prediction_path = path.join(
                                BASE_PATH,
                                "output",
                                "model-predictions",
                                "shades",
                                f"seed{seed}",
                                f"{model}_{model_setting}_context_{_context}_task_{_prompt_lang}_instance_{_input_lang}.csv",
                            )

                            if not path.isfile(prediction_path):
                                raise FileExistsError(prediction_path)
                            df = pd.read_csv(prediction_path, sep=";")

                            if _prompt_lang == "zh":
                                df["predicted_score"] = df["output"].apply(check_for_yes_no_chinese)
                            else:
                                df["predicted_score"] = df["output"].apply(
                                    get_predicted_label_no_restriction
                                )

                            if correct_for_answer_extraction_failures:
                                print(
                                    len(df[df["predicted_score"] == "No prediction"]),
                                    "of",
                                    len(df),
                                    "for",
                                    model,
                                    "with",
                                    model_setting,
                                )
                                df = df[df["predicted_score"] != "No prediction"]

                            df["original_predictions"] = df["predicted_score"]

                            # If we cannot extract a prediction, we measure it as wrong prediction
                            df["predicted_score"] = df.apply(
                                lambda x: (
                                    get_wrong_prediction(x)
                                    if x["predicted_score"] == "No prediction"
                                    else x["predicted_score"]
                                ),
                                axis=1,
                            )

                            if len(df) != len(shades_data["ground_truth"]):
                                print(
                                    f"Error parsing labels for prompt {_prompt_lang} and seed {seed}, input {_input_lang}, {model} has a different length as groundtruth"
                                )
                                print(
                                    f"{len(df)} and expected {len(shades_dfs_context_specific[_context]['ground_truth'])}"
                                )
                            # if df["predicted_score"].isna().sum() > 100:
                            #     print(f"model: {model}, {model_setting}, {_input_lang}, {_prompt_lang}")
                            # print(df["predicted_score"].isna().sum())
                            y_pred = df["predicted_score"]
                            f1_macro = f1_score(
                                df["ground_label"].apply(lambda x: 1 if x == "stereotype" else 0),
                                y_pred,
                                average="macro",
                            )
                            with warnings.catch_warnings():
                                warnings.filterwarnings("error", category=UndefinedMetricWarning)
                                try:
                                    precision = precision_score(
                                        df["ground_label"].apply(
                                            lambda x: 1 if x == "stereotype" else 0
                                        ),
                                        y_pred,
                                        pos_label=0,
                                    )
                                except UndefinedMetricWarning:
                                    print(f"Warning triggered for file: {prediction_path}")
                                    # Handle the failure (e.g., set precision to 0 or skip)
                                    precision = 0.0

                            recall = recall_score(
                                df["ground_label"].apply(lambda x: 1 if x == "stereotype" else 0),
                                y_pred,
                                pos_label=0,
                            )

                            seed_f1s.append(f1_macro)
                            tn, fp, fn, tp = (
                                confusion_matrix(
                                    df["ground_label"].apply(lambda x: 1 if x == "stereotype" else 0),
                                    y_pred,
                                )
                                .ravel()
                                .tolist()
                            )

                            data_source = ""
                            for source in DATA_SOURCES.keys():
                                if model_setting == "base" or model_setting == "base_quantized":
                                    data_source = "base"
                                elif model_setting in DATA_SOURCES[source]:
                                    data_source = source
                                elif (
                                    "quantized" in model_setting and model_setting[: len("_quantized")]
                                ):
                                    data_source = source
                                else:
                                    raise AttributeError("Couldn't fine data source", source)
                            seed_results.append(
                                {
                                    "model": model,
                                    "seed": seed,
                                    "input_language": _input_lang,
                                    "prompt_language": _prompt_lang,
                                    "f1_macro": f1_macro,
                                    "true_positive": tp,
                                    "true_negative": tn,
                                    "false_positive": fp,
                                    "false_negative": fn,
                                    "precision": precision,
                                    "recall": recall,
                                    "data_source": data_source,
                                    "model_setting": model_setting,
                                    "stereotype_context": _context,
                                    "instruction_tuning_language": "nl"
                                    if model_setting.endswith("-nl")
                                    else "en",
                                }
                            )
                            model_dfs_aggregated[model] = df

                        if seed_f1s:
                            avg_f1 = sum(seed_f1s) / len(seed_f1s)
                        else:
                            avg_f1 = None

                        results[f"{_context}-{model}-{model_setting}-{_prompt_lang}-{_input_lang}"] = {
                            "per_seed": seed_results,
                            "macro_f1_avg": avg_f1,
                        }


    # Convert results to DataFrame for easier access
    all_model_results = [
        {
            "model": result["model"],
            "data_source": result["data_source"],
            "model_setting": result["model_setting"],
            "input_language": result["input_language"],
            "prompt_language": result["prompt_language"],
            "stereotype_context": result["stereotype_context"],
            "instruction_tuning_language": result["instruction_tuning_language"],
            "seed": result["seed"],
            "f1_macro": result["f1_macro"],
            "true_positive": result["true_positive"],
            "true_negative": result["true_negative"],
            "false_positive": result["false_positive"],
            "false_negative": result["false_negative"],
            "false_negative_rate": result["false_negative"]
            / (result["false_negative"] + result["true_positive"]),
            "false_positive_rate": result["false_positive"]
            / (result["false_positive"] + result["true_negative"]),
            "true_negative_rate": result["true_negative"]
            / (result["true_negative"] + result["false_positive"]),
            "true_positive_rate": result["true_positive"]
            / (result["true_positive"] + result["false_negative"]),
            "recall": result["recall"],
            "precision": result["precision"],
            "average": results[model]["macro_f1_avg"],
        }
        for model in results.keys()
        for result in results[model]["per_seed"]
    ]
    results_df = pd.DataFrame(data=all_model_results)
    return (results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting for $\mathcal{H}_{2}$
    """)
    return


@app.cell
def _():
    model_ordering = ["qwen3", "mistral", "mistralbase", "llama31", "llamabase", "gpt-5.2"]
    return


@app.cell
def _():
    model_setting_ordering = [
        "quantized_base",
        "quantized_ultrachat_en",
        "quantized_ultrachat_nl",
        "quantized_no_robots_en",
        "quantized_no_robots_nl",
        "quantized_stackoverflow_en",
        "quantized_stackoverflow_nl",
        "quantized_belle_zh",
    ]
    return (model_setting_ordering,)


@app.cell
def _(results_df):
    _filtered_df = results_df[
        (results_df["prompt_language"] == results_df["input_language"])
        # & (results_df["prompt_language"] == results_df["stereotype_context"])
        & (results_df["stereotype_context"] == "nl")
        # & (results_df["model_setting"] != "baseline")
    ].copy()
    _filtered_df["model+context_combined"] = (
        _filtered_df["model"] + " - " + _filtered_df["stereotype_context"]
    )
    _filtered_df
    return


@app.cell
def _(model_setting_ordering, np, pd, plt, results_df, sns, stats):
    def plot_final_significance_heatmap_with_stubs(df, ordering, baseline="quantized_base"):
        counter_test = 0
        # 1. Processing Logic
        _local_df = df[
            (df["prompt_language"] == df["input_language"]) & 
            (df["prompt_language"] == df["stereotype_context"])
        ].copy()

        processed_list = []
        for (m_name, ctx), group in _local_df.groupby(['model', 'stereotype_context'], sort=True):
            base_group = group[group['model_setting'] == baseline].sort_values('seed')
            base_scores = base_group['f1_macro'].values

            for setting in group['model_setting'].unique():
                setting_group = group[group['model_setting'] == setting].sort_values('seed')
                test_scores = setting_group['f1_macro'].values

                if len(test_scores) == 0 or len(base_scores) == 0: continue

                mean_f1 = np.mean(test_scores)
                is_sig = False
                is_norm = False

                if setting != baseline and len(test_scores) == len(base_scores):
                    # _, p = stats.ttest_rel(test_scores, base_scores, alternative='greater')
                    _, p = stats.wilcoxon(test_scores, base_scores, alternative="greater")
                    differences = test_scores - base_scores
                    shapiro_stat12, shapiro_p12 = stats.shapiro(differences)
                    is_sig = p < 0.05


                    # is_norm = shapiro_p12 > 0.05
                    # if not is_norm:
                    #     counter_test += 1
                processed_list.append({
                    'Group': f"{m_name} - {ctx}",
                    'Setting': setting,
                    'F1': mean_f1,
                    'Label': f"{mean_f1:.3f}{'*' if is_sig else ''}",
                    # 'Label': f"{mean_f1:.3f}{'*' if is_sig else ''}{'@' if is_norm else ''}",
                    'ModelName': m_name
                })

        res_df = pd.DataFrame(processed_list)

        heatmap_data = res_df.pivot(index='Group', columns='Setting', values='F1')
        heatmap_data = heatmap_data.reindex(columns=[s for s in ordering if s in heatmap_data.columns])

        real_annots = res_df.pivot(index='Group', columns='Setting', values='Label')
        real_annots = real_annots.reindex(columns=heatmap_data.columns)

        plt.figure(figsize=(12, 10))

        ax = sns.heatmap(
            heatmap_data, 
            annot=np.array(real_annots), 
            fmt="", 
            cmap="YlGnBu", 
            linewidths=.8, 
            linecolor='white',
            cbar_kws={'label': 'F1 Score'}
        )


        #tick marks
        ax.tick_params(axis='both', which='major', left=True, bottom=True, length=8, width=1, color='gray')
        ax.set_yticklabels(
            ax.get_yticklabels(), 
            rotation=30, 
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=12
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            fontsize=12,
            horizontalalignment="right"
        )

        #Line separators
        for idx in range(1, len(heatmap_data)):
            curr_model = heatmap_data.index[idx].split(" - ")[0]
            prev_model = heatmap_data.index[idx-1].split(" - ")[0]

            if curr_model != prev_model:
                ax.axhline(idx, color='white', lw=15)

        plt.title("Hypothesis 3: Localized Fine-tuning Performance", fontsize=16, pad=30)
        plt.xlabel("Model Setting", fontsize=12, labelpad=15)
        plt.ylabel("Model - Context", fontsize=12, labelpad=20)

        # Adjust layout to ensure angled labels aren't cut off
        plt.tight_layout()
        plt.savefig("output/evaluation/h02-real_sig.png")
        plt.show()
        print(counter_test)

    plot_final_significance_heatmap_with_stubs(results_df, model_setting_ordering)


    ###########################

    def plot_dual_heatmaps(df, ordering, baseline="quantized_base"):

        _local_df = df[
            (df["prompt_language"] == df["input_language"]) & 
            (df["prompt_language"] == df["stereotype_context"])
        ].copy()

        processed_list = []
        for (m_name, ctx), group in _local_df.groupby(['model', 'stereotype_context'], sort=True):
            base_group = group[group['model_setting'] == baseline].sort_values('seed')
            base_scores = base_group['f1_macro'].values
            base_mean = np.mean(base_scores)

            for setting in group['model_setting'].unique():
                setting_group = group[group['model_setting'] == setting].sort_values('seed')
                test_scores = setting_group['f1_macro'].values

                if len(test_scores) == 0 or len(base_scores) == 0: continue

                mean_f1 = np.mean(test_scores)
                delta_f1 = mean_f1 - base_mean
                is_sig = False

                if setting != baseline and len(test_scores) == len(base_scores):
                    _, p = stats.ttest_rel(test_scores, base_scores, alternative='greater')
                    is_sig = p < 0.05

                processed_list.append({
                    'Group': f"{m_name} - {ctx}",
                    'Setting': setting,
                    'F1': mean_f1,
                    'Delta': delta_f1,
                    'Label': f"{delta_f1:+.3f}{'*' if is_sig else ''}",
                    'ModelName': m_name
                })

        res_df = pd.DataFrame(processed_list)

        delta_data = res_df.pivot(index='Group', columns='Setting', values='Delta')
        delta_data = delta_data.reindex(columns=[s for s in ordering if s in delta_data.columns])

        delta_annots = res_df.pivot(index='Group', columns='Setting', values='Label')
        delta_annots = delta_annots.reindex(columns=delta_data.columns)

        plt.figure(figsize=(12, 10))

        ax = sns.heatmap(
            delta_data, 
            annot=np.array(delta_annots), 
            fmt="", 
            cmap="RdBu_r", 
            center=0,
            linewidths=.8, 
            linecolor='white',
            cbar_kws={'label': 'Δ F1 Score (vs Base)'}
        )

        ax.tick_params(axis='both', which='major', left=True, bottom=True, length=8, width=1, color='gray')
        ax.set_yticklabels(
            ax.get_yticklabels(), 
            rotation=30, 
            horizontalalignment='right',
            verticalalignment='top',
            fontsize=12
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            fontsize=12,
            horizontalalignment="right"
        )


        #Line separators
        for idx in range(1, len(delta_data)):
            curr_model = delta_data.index[idx].split(" - ")[0]
            prev_model = delta_data.index[idx-1].split(" - ")[0]

            if curr_model != prev_model:
                ax.axhline(idx, color='white', lw=15)

        plt.title("Hypothesis 3: Delta Performance relative to Quantized Base", fontsize=16, pad=30)
        plt.xlabel("Model Setting", fontsize=12, labelpad=15)
        plt.ylabel("Model - Context", fontsize=12, labelpad=20)
        plt.tight_layout()
        plt.savefig("output/evaluation/h02-delta_sig.png")
        plt.show()

    plot_dual_heatmaps(results_df, model_setting_ordering)
    return


@app.cell
def _(alt, model_setting_ordering, results_df):
    _filtered_df = results_df[
        (results_df["prompt_language"] == results_df["input_language"])
        & (results_df["prompt_language"] == results_df["stereotype_context"])
        # & (results_df["stereotype_context"] == "zh")
        # & (results_df["model_setting"] != "baseline")
    ].copy()
    _filtered_df["model+context_combined"] = (
        _filtered_df["model"] + " - " + _filtered_df["stereotype_context"]
    )


    _chart = (
        alt.Chart(_filtered_df)
        .encode(
            alt.X("model_setting", sort=model_setting_ordering, title="Model Setting"),
            alt.Y(
                "average(f1_macro)",
                title="Macro F1 Score (5 seeds)",
                scale=alt.Scale(domain=[0, 0.45]),
            ),
            alt.XOffset(
                "model:N",
            ).scale(
                paddingInner=1,  # Try 0.1 to 0.5. Lower value = marks are further apart.
                paddingOuter=1,  # Space at the edges of the categorical tick.
            ),
            alt.Color("model", legend=None),
            alt.Shape("stereotype_context", legend=None),
            # alt.StrokeDash("stereotype_context", legend=None),
        )
        .properties(
            width=700,
            # height=150
        )
    )

    # Add primary line plot
    _lines = _chart.mark_line(point=alt.OverlayMarkDef(size=50, filled=True))

    # Add point values
    _text = _chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        dx=-20,
    ).encode(text=alt.Text("average(f1_macro)", format=".3f"))

    # Add Error Bars (Standard Deviation)
    _errors = _chart.mark_errorbar(extent="ci").encode(
        y=alt.Y("f1_macro:Q", title="Macro F1 Score (5 seeds)"),
        # Ensure they are colored by model like the lines
        color=alt.Color("model"),
    )

    # Add model label to last data points
    _last_setting = model_setting_ordering[-1]
    _labels = (
        _chart.mark_text(
            align="left",
            baseline="middle",
            dx=10,  # Offset to the right
        )
        .encode(text="model+context_combined", detail="stereotype_context")
        .transform_filter(
            # Only show this label on the very last x-axis step
            alt.datum.model_setting == _last_setting
        )
    )

    # Plot all the things
    # print("Note: in this plot, prompt language == input language (always), for consistency")

    # (_lines + _text + _labels).save("output/evaluation/h3.png", scale_factor=2.0)
    # _lines + _text + _labels
    final_chart = (
        alt.layer(_lines, _errors, _text, _labels)
        .properties(
            width=700,  # Width per facet
            height=150,
        )
        .facet(
            row=alt.Row(
                "stereotype_context", header=alt.Header(title="Stereotype Context", titleFontSize=14)
            )
        )
        .properties(title=f"Hypothesis 3: Localized Fine-tuning")
    )

    print("Note: in this plot, prompt language == input language (always), for consistency")

    # Compute average false prediction numbers

    # Save and Display
    final_chart.save("output/evaluation/h3.pdf", scale_factor=2.0)
    final_chart
    return


if __name__ == "__main__":
    app.run()

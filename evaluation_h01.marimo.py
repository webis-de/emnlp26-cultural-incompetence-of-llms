import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation for $\mathcal{H_1}$
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
        wilcoxon,
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
    MODELS = ["mistral", "llama31", "qwen3", "gpt-5.2"]

    MODEL_SETTINGS = [
        "quantized_base",
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
    ### DataLoading
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
                                r"./output/model-predictions/shades",
                                f"seed{seed}",
                                f"{model}_{model_setting}_context_{_context}_task_{_prompt_lang}_instance_{_input_lang}.csv",
                            )
                            # prediction_path = path.join(
                            #     BASE_PATH,
                            #     "output",
                            #     "model-predictions",
                            #     "shades",
                            #     f"seed{seed}",
                            #     f"{model}_{model_setting}_context_{_context}_task_{_prompt_lang}_instance_{_input_lang}.csv",
                            # )

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
    ## Dataframe - Analysis
    """)
    return


@app.cell(hide_code=True)
def _(results_df):
    def analyze_matching_macro_stats(input_results_df):
        # 1. Apply your exact filtering logic
        # Filter for 'quantized_base' or 'baseline' where prompt language matches input language
        mask = (input_results_df["prompt_language"] == input_results_df["input_language"]) & (
            input_results_df["model_setting"].isin(["quantized_base", "baseline"])
        )
        analysis_subset = input_results_df[mask].copy()

        # 2. Replicate your combined key
        analysis_subset["context+language_combined"] = (
            analysis_subset["stereotype_context"] + " - " + analysis_subset["prompt_language"]
        )

        # 3. Group and Aggregate
        # We aggregate the F1 scores you already calculated (f1_macro)
        # and the raw counts to check for overprediction
        grouped_results = (
            analysis_subset.groupby(["model", "context+language_combined"])
            .agg(
                {
                    "f1_macro": "mean",  # Matches your "average macro f1" per group
                    "true_positive": "sum",
                    "false_positive": "sum",
                    "true_negative": "sum",
                    "false_negative": "sum",
                }
            )
            .reset_index()
        )

        # 4. Calculate the Overprediction Metrics for the group
        # Predicted Negatives = TN + FN
        # Actual Negatives = TN + FP
        total = (
            grouped_results["true_positive"]
            + grouped_results["false_positive"]
            + grouped_results["true_negative"]
            + grouped_results["false_negative"]
        )

        # Negative Overprediction Bias: (Predicted Negs - Actual Negs) / Total
        # Simplified: (FN - FP) / Total
        grouped_results["neg_overprediction_bias"] = (
            grouped_results["false_negative"] - grouped_results["false_positive"]
        ) / total

        # 5. Calculate Prediction Ratios (The "Dumb Classifier" Check)
        grouped_results["pred_stereotype_share"] = (
            (grouped_results["true_positive"] + grouped_results["false_positive"]) / total
        ) * 100

        return grouped_results.sort_values(by=["model", "f1_macro"], ascending=[True, False])


    final_stats = analyze_matching_macro_stats(results_df)
    # final_stats = final_stats[final_stats["model"] == "mistral"]
    final_stats
    return (final_stats,)


@app.cell
def _(final_stats, mo):
    MODEL_COLORS = {
        "mistral": "#FF4B4B",  # Red-ish
        "llama31": "#4D96FF",  # Blue-ish
        "qwen3": "#FFD93D",  # Yellow-ish
        "baseline": "#6BCB77",  # Green-ish
    }

    model_list = sorted(final_stats["model"].unique().tolist())
    model_selector = mo.ui.multiselect(
        options=model_list, value=model_list, label="Select Models to Display:"
    )

    model_selector
    return MODEL_COLORS, model_selector


@app.cell
def _(MODEL_COLORS, alt, final_stats, mo, model_selector):
    def create_highlighted_chart(df, selected_models):
        filtered_df = df[df["model"].isin(selected_models)].copy()

        if filtered_df.empty:
            return mo.md("### No models selected.")

        filtered_df["is_zh_prompt"] = filtered_df["context+language_combined"].str.endswith("zh")

        base = alt.Chart(filtered_df).encode(
            x=alt.X(
                "pred_stereotype_share:Q",
                title="Stereotype Trigger Rate (%)",
                scale=alt.Scale(domain=[45, 90]),
            ),
            y=alt.Y("f1_macro:Q", title="F1 Macro Score", scale=alt.Scale(domain=[0.25, 0.55])),
            color=alt.Color(
                "model:N",
                scale=alt.Scale(domain=list(MODEL_COLORS.keys()), range=list(MODEL_COLORS.values())),
            ),
            tooltip=[
                "model",
                "context+language_combined",
                "f1_macro",
                "pred_stereotype_share",
                "neg_overprediction_bias",
            ],
        )

        # POINTS: Highlight 'zh' with full opacity and larger size
        points = base.mark_circle().encode(
            opacity=alt.condition(alt.datum.is_zh_prompt, alt.value(1.0), alt.value(0.2)),
            size=alt.condition(alt.datum.is_zh_prompt, alt.value(200), alt.value(60)),
            stroke=alt.condition(alt.datum.is_zh_prompt, alt.value("black"), alt.value(None)),
            strokeWidth=alt.value(1.5),
        )

        # LABELS: Only show labels for 'zh' prompts to avoid clutter
        labels = base.mark_text(
            align="left", baseline="middle", dx=10, fontSize=11, fontWeight="bold"
        ).encode(
            text=alt.condition(alt.datum.is_zh_prompt, "context+language_combined:N", alt.value("")),
            opacity=alt.value(1.0),
        )

        # TREND LINE: Global trend to show the overall inverse relationship
        trend = base.transform_regression("pred_stereotype_share", "f1_macro").mark_line(
            color="#333", strokeDash=[4, 4], opacity=0.4
        )

        return (
            (trend + points + labels)
            .properties(
                width=750, height=500, title="Highlighting Chinese Prompts: F1 Macro vs. Trigger Rate"
            )
            .interactive()
        )

    mo.vstack([model_selector, create_highlighted_chart(final_stats, model_selector.value)])
    return


@app.cell
def _(alt, final_stats, mo, model_selector):
    plot_df = final_stats.copy()
    plot_df[["context_lang", "prompt_lang"]] = plot_df["context+language_combined"].str.split(
        " - ", expand=True
    )

    #Calculate the Actual vs Predicted distribution
    #Actual Stereotypes = TP + FN
    #Predicted Stereotypes = TP + FP
    #Total = TP + FP + TN + FN
    plot_df["total_samples"] = plot_df[
        ["true_positive", "false_positive", "true_negative", "false_negative"]
    ].sum(axis=1)
    plot_df["Actual %"] = (plot_df["true_positive"] + plot_df["false_negative"]) / plot_df[
        "total_samples"
    ]
    plot_df["Predicted %"] = (plot_df["true_positive"] + plot_df["false_positive"]) / plot_df[
        "total_samples"
    ]

    melted_df = plot_df.melt(
        id_vars=["model", "context_lang", "prompt_lang"],
        value_vars=["Actual %", "Predicted %"],
        var_name="Type",
        value_name="Ratio",
    )


    def create_distribution_chart(df, selected_model):
        chart_data = df[df["model"] == selected_model]

        if chart_data.empty:
            return mo.md("No data for selected model.")

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("Type:N", title=None, axis=alt.Axis(labels=False)),
                y=alt.Y("Ratio:Q", axis=alt.Axis(format="%"), title="Share of Samples"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(range=["#6BCB77", "#FF4B4B"]),
                    legend=alt.Legend(orient="bottom"),
                ),
                column=alt.Column("context_lang:N", title="Context (Ground Truth Source)"),
                tooltip=["context_lang", "prompt_lang", "Type", alt.Tooltip("Ratio:Q", format=".1%")],
            )
            .properties(
                width=120,
                height=250,
                title=f"Distribution Gap: Why {selected_model} 'prefers' Chinese Context",
            )
            .configure_view(stroke=None)
        )

        return chart

    create_distribution_chart(melted_df, model_selector.value[0] if model_selector.value else "mistral")
    return


@app.cell
def _(plt, results_df, sns):
    results_df["total_samples"] = (
        results_df["true_positive"]
        + results_df["true_negative"]
        + results_df["false_positive"]
        + results_df["false_negative"]
    )
    results_df1 = results_df[results_df["model_setting"] == "quantized_base"].copy()
    # Net Bias = (Predicted Negatives - Actual Negatives) / Total
    # This simplifies to: (False Negatives - False Positives) / Total
    results_df1["neg_bias_score"] = (
        results_df1["false_negative"] - results_df1["false_positive"]
    ) / results_df1["total_samples"]

    lang_comparison = (
        results_df1.groupby(["model", "prompt_language"])["neg_bias_score"].mean().reset_index()
    )

    pivot_df = lang_comparison.pivot(index="model", columns="prompt_language", values="neg_bias_score")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, cmap="RdBu_r", center=0)
    plt.title("Negative Class Overprediction Bias (FN - FP)")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting for $\mathcal{H}_{1}$
    """)
    return


@app.cell
def _():
    model_ordering = ["qwen3", "mistral", "llama31", "gpt-5.2"]
    return (model_ordering,)


@app.cell
def _(alt, model_ordering, np, results_df):
    _filtered_df = results_df[
        (results_df["prompt_language"] == results_df["input_language"])
        & (
            (results_df["model_setting"] == "quantized_base")
            | (results_df["model_setting"] == "baseline")
        )
    ].copy()
    _filtered_df["context+language_combined"] = (
        _filtered_df["stereotype_context"] + " - " + _filtered_df["prompt_language"]
    )

    _chart = (
        alt.Chart(_filtered_df, title="Hypothesis 1: Regional LLMs")
        .encode(
            alt.X(
                "model",
                title="Model family",
                sort=model_ordering,
                scale=alt.Scale(paddingInner=0.2, paddingOuter=0.1),
            ),
            alt.Y(
                "average(f1_macro)", title="Macro F1 Score (5 seeds)", scale=alt.Scale(domain=[0, 0.45])
            ),
            # Use the combined field for the offset
            alt.XOffset(
                "context+language_combined",
                title="Context & Prompt Language",
                # sort=["nl - nl", "nl - en", "en - nl", "en - en"],
                sort=[
                    "nl - nl",
                    "nl - en",
                    "nl - zh",
                    "zh - nl",
                    "zh - en",
                    "zh - zh",
                    "en - nl",
                    "en - en",
                    "en - zh",
                ],
            ),
            alt.Color(
                "context+language_combined",
                title="Context & Prompt Language",
                # sort=["nl - nl", "nl - en", "en - nl", "en - en"],
                sort=[
                    "nl - nl",
                    "nl - en",
                    "nl - zh",
                    "zh - nl",
                    "zh - en",
                    "zh - zh",
                    "en - nl",
                    "en - en",
                    "en - zh",
                ],
            ),
        )
        .properties(
            width=400,
            # height=150
        )
    )

    _bar = _chart.mark_bar(
        # size=12,
    )

    _text = _chart.mark_text(align="center", baseline="bottom", fontSize=13, dx=0, dy=-15).encode(
        text=alt.Text("average(f1_macro)", format=".3f")
    )

    _error_bars = _chart.mark_errorbar(extent="ci").encode(
        x=alt.X("model", sort=model_ordering),
        y="average(f1_macro)",
        color=alt.value("black"),
    )

    print("Note: in this plot, prompt language == input language (always), for consistency")

    # Compute average false prediction numbers
    for _model in _filtered_df["model"].unique():
        for _context_lang in _filtered_df[_filtered_df["model"] == _model][
            "context+language_combined"
        ].unique():
            _fps = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["false_positive"].values
            _fns = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["false_negative"].values
            _tns = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["true_negative"].values
            _tnr = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["true_negative_rate"].values
            _fnr = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["false_negative_rate"].values
            _fpr = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["false_positive_rate"].values
            _tpr = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["true_positive_rate"].values
            _recall = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["recall"].values
            _precision = _filtered_df[
                (_filtered_df["context+language_combined"] == _context_lang)
                & (_filtered_df["model"] == _model)
            ]["precision"].values

            # print(f"{_model} on context+language {_context_lang}, average FP: {np.mean(_fps)}")
            # print(f"{_model} on context+language {_context_lang}, average FN: {np.mean(_fns)}")
            # print(f"{_model} on context+language {_context_lang}, average TN: {np.mean(_tns)}")
            # print(f"{_model} on context+language {_context_lang}, average FPR: {np.mean(_fpr)}")
            # print(f"{_model} on context+language {_context_lang}, average FNR: {np.mean(_fnr)}")
            # print(f"{_model} on context+language {_context_lang}, average TPR: {np.mean(_tpr)}")
            # print(f"{_model} on context+language {_context_lang}, average TNR: {np.mean(_tnr)}")
            print(
                f"{_model} on context+language {_context_lang}, average Precision (for negative class): {np.mean(_precision)}"
            )
            print(
                f"{_model} on context+language {_context_lang}, average Recall (for negative class): {np.mean(_recall)}"
            )
            print("\n")

    (_bar + _text + _error_bars).save("output/evaluation/h1.pdf", ppi=600)
    _bar + _text + _error_bars
    return


@app.cell
def _(alt, model_ordering, results_df):
    _filtered_df = results_df[
        (results_df["prompt_language"] == results_df["input_language"])
        & (
            (results_df["model_setting"] == "quantized_base")
            | (results_df["model_setting"] == "baseline")
        )
    ].copy()
    _filtered_df["context+language_combined"] = (
        _filtered_df["stereotype_context"] + " - " + _filtered_df["prompt_language"]
    )
    base = alt.Chart(_filtered_df).encode(
        alt.X("prompt_language:N", title="Prompt Language", axis=alt.Axis(labelAngle=0)),
        alt.Y("average(f1_macro):Q", title="Macro F1 Score", scale=alt.Scale(domain=[0, 0.50])),
        alt.XOffset("model:N", sort=model_ordering),
        alt.Color("model:N", title="Model Family", sort=model_ordering)
    )

    # 2. Create the individual layers
    _bars = base.mark_bar()

    _text = base.mark_text(align="center", baseline="bottom", dy=-15).encode(
        text=alt.Text("average(f1_macro):Q", format=".3f"),
        color=alt.value("black") # Keep text readable regardless of bar color
    )

    _error_bars = base.mark_errorbar(extent="ci").encode(
        color=alt.value("black")
    )

    # 3. Layer them together FIRST, then Facet LAST
    _combined_chart = (
        alt.layer(_bars, _text, _error_bars)
        .facet(
            column=alt.Column("stereotype_context:N", title="Stereotype Context")
        )
        .properties(
            title="Hypothesis 1: Regional LLMs"
        )
        .configure_facet(
            spacing=20 # This controls the "clean gap" you wanted between contexts
        )
    )

    (_combined_chart).save("output/evaluation/h1.pdf", ppi=600)
    # Combine them
    _combined_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Significance for $\mathcal{H}_{1}$
    """)
    return


@app.cell
def _(results_df):
    _filtered_df = results_df[
        (results_df["prompt_language"] == results_df["input_language"])
        & (
            (results_df["model_setting"] == "quantized_base")
            | (results_df["model_setting"] == "baseline")
        )
    ].copy()
    _filtered_df["context+language_combined"] = (
        _filtered_df["stereotype_context"] + " - " + _filtered_df["prompt_language"]
    )
    _filtered_df
    return


@app.cell
def _(stats):
    #Tests if row_f1 is significantly larger than base_f1. This function should recieve the averaged f1-scores
    def calculate_significance(row_f1, base_f1):
        t_stat, p_val = stats.ttest_rel(base_f1, base_f1, alternative="greater")
        return p_val
    return


@app.cell
def _(results_df, stats):
    significance_df = results_df[
        (results_df["prompt_language"] == results_df["input_language"])
        & (
            (results_df["model_setting"] == "quantized_base")
            | (results_df["model_setting"] == "baseline")
        )
    ].copy()
    significance_df["context+language_combined"] = (
        significance_df["stereotype_context"] + " - " + significance_df["prompt_language"]
    )


    for model_name, model_group in significance_df.groupby("model"):
        baseline_maks = (model_group["prompt_language"] == "en")
        baseline_scores = model_group[baseline_maks]
        # print(baseline_scores)
        if baseline_scores.empty:
            print(f"Error! There is no data for model {model_name}")
            continue

        for stereo_context in ["nl", "en", "zh"]:
            baseline_f1_scores = baseline_scores[
                baseline_scores["stereotype_context"] == stereo_context
            ]["f1_macro"].to_numpy()
            if len(baseline_f1_scores) <= 1:
                continue

            for other_lang in ["nl", "zh"]:

                comparison_mask = (significance_df["model"] == model_name) & \
                                  (significance_df["stereotype_context"] == stereo_context) & \
                                  (significance_df["prompt_language"] == other_lang)
                comparison_array = significance_df.loc[comparison_mask, "f1_macro"].to_numpy()
                # print(len(comparison_array))

                if len(baseline_f1_scores) > 1 and len(comparison_array) > 1:
                    tmp_t_stat, tmp_p_val = stats.ttest_rel(comparison_array, baseline_f1_scores, alternative="greater")
                    # print(comparison_array)
                    shap_diff = comparison_array - baseline_f1_scores
                    shapiro_stat1, shapiro_p1 = stats.shapiro(shap_diff)
                    if shapiro_p1 < 0.05:
                        print(f"{model_name} - {stereo_context} - {other_lang}")
                    significance_df.loc[comparison_mask, "p_value"] = tmp_p_val

    significance_df.loc[significance_df["prompt_language"] == "en", "p_value"] = 1.0

    significance_df
    return (significance_df,)


@app.cell
def _(np, pd, plt, significance_df, sns):
    # Assuming imports (pd, sns, plt, np) are already handled in your top cells

    # 1. Filter for Seed 1 
    plot_df_sig = significance_df[significance_df["seed"] == 1].copy()
    plot_df_sig = plot_df_sig.sort_values(["stereotype_context", "prompt_language"])

    # 2. Pivot the Data for scores and p-values
    heatmap_data = plot_df_sig.pivot_table(
        index="context+language_combined", 
        columns="model", 
        values="average",
        sort=False
    )

    p_values_pivot = plot_df_sig.pivot_table(
        index="context+language_combined", 
        columns="model", 
        values="p_value",
        sort=False
    )

    # 3. Calculate Deltas relative to English baseline within each context
    delta_data = heatmap_data.copy()

    for ctx_name in plot_df_sig["stereotype_context"].unique():
        base_idx = f"{ctx_name} - en"

        if base_idx in delta_data.index:
            # Get the English baseline row for this specific context
            ctx_baseline = heatmap_data.loc[base_idx]

            # Identify all rows for this context group
            matching_rows = [idx for idx in delta_data.index if idx.startswith(f"{ctx_name} - ")]

            # Subtract baseline
            delta_data.loc[matching_rows] = delta_data.loc[matching_rows] - ctx_baseline

    # 4. Create the Annotation Matrix with unique variable names
    delta_annots = []
    for row_idx in range(len(delta_data)):
        row_values = []
        for col_idx in range(len(delta_data.columns)):
            val = delta_data.iloc[row_idx, col_idx]
            p_val1 = p_values_pivot.iloc[row_idx, col_idx]

            # Format: +0.05* or -0.10
            label = f"{val:+.2f}*" if (pd.notna(p_val1) and p_val1 < 0.05) else f"{val:+.2f}"
            row_values.append(label)
        delta_annots.append(row_values)


    real_annots = []
    for row_idx in range(len(heatmap_data)):
        row_values = []
        for col_idx in range(len(heatmap_data.columns)):
            val = heatmap_data.iloc[row_idx, col_idx]
            p_val_real = p_values_pivot.iloc[row_idx, col_idx]

            # Using simple .2f formatting; keeping the * if significant
            label = f"{val:.2f}*" if (pd.notna(p_val_real) and p_val_real < 0.05) else f"{val:.2f}"
            row_values.append(label)
        real_annots.append(row_values)

    # 5. Plotting
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        delta_data, 
        annot=np.array(delta_annots), 
        fmt="", 
        cmap="RdBu_r", 
        center=0, 
        linewidths=.8, 
        linecolor='white',
        cbar_kws={'label': 'Δ F1 Score'}
    )

    # Add separator lines
    for idx in range(1, len(delta_data)):
        curr_ctx = delta_data.index[idx].split(" - ")[0]
        prev_ctx = delta_data.index[idx-1].split(" - ")[0]
        if curr_ctx != prev_ctx:
            plt.gca().axhline(idx, color='white', lw=15)

    plt.title("Δ F1 Scores vs. en prompting baseline\n(* = p < 0.05)", fontsize=14, pad=20)
    plt.ylabel("Context - Prompt Language")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("output/evaluation/h01-significance.png")
    plt.show()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_data, 
        annot=np.array(real_annots), 
        fmt="", 
        cmap="YlGnBu",   # Different colormap for absolute values
        linewidths=.8, 
        linecolor='white',
        cbar_kws={'label': 'F1 Score'}
    )

    # Add separator lines
    for idx in range(1, len(heatmap_data)):
        curr_ctx = heatmap_data.index[idx].split(" - ")[0]
        prev_ctx = heatmap_data.index[idx-1].split(" - ")[0]
        if curr_ctx != prev_ctx:
            plt.gca().axhline(idx, color='white', lw=15)

    plt.title("Real F1 Score Values\n(* = p < 0.05)", fontsize=14, pad=20)
    plt.ylabel("Context - Prompt Language")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("output/evaluation/h01-real_values.png")
    plt.show()
    return


@app.cell
def _(mo):
    names = ["mistral", "qwen3", "llama31"]
    language = [
        "nl - nl",
        "nl - en",
        "nl - zh",
        "zh - zh",
        "zh - en",
        "zh - nl",
        "en - en",
        "en - zh",
        "en - nl",
    ]
    # language = ["nl - nl", "nl - en", "en - nl", "en - en"]

    selector1 = mo.ui.multiselect(
        options=names,
        label="Select the model:",
    )

    selector2 = mo.ui.multiselect(
        options=language,
        label="Select exactly first language setting:",
    )

    selector3 = mo.ui.multiselect(
        options=language,
        label="Select exactly second language setting:",
    )

    # Display them together using a vertical stack
    mo.vstack([selector1, selector2, selector3])
    return selector1, selector2, selector3


@app.cell
def _(np, plt, results_df, selector1, selector2, selector3, stats, wilcoxon):
    selected = selector1.value
    if len(selected) != 1:
        print(f"WARNING: Please select exactly 1 name. Currently selected: **{len(selected)}**")

    else:
        print(f"You have selected: {', '.join(selected)}")

    _filtered_df = results_df[
        (results_df["prompt_language"] == results_df["input_language"])
        & (
            (results_df["model_setting"] == "quantized_base")
            | (results_df["model_setting"] == "baseline")
        )
    ].copy()
    _filtered_df["context+language_combined"] = (
        _filtered_df["stereotype_context"] + " - " + _filtered_df["prompt_language"]
    )


    if len(selector2.value) == 1 and len(selector3.value) == 1:
        # filter by language
        df_lang1 = _filtered_df[_filtered_df["context+language_combined"] == str(selector2.value[0])]
        df_lang2 = _filtered_df[_filtered_df["context+language_combined"] == str(selector3.value[0])]

        # filter by model name
        df_lang1 = df_lang1[df_lang1["model"] == str(selector1.value[0])]
        df_lang2 = df_lang2[df_lang2["model"] == str(selector1.value[0])]

    else:
        print("Only select one language setting!")

    first_f1_values = df_lang1["f1_macro"].to_numpy()
    second_f1_values = df_lang2["f1_macro"].to_numpy()

    differences = first_f1_values - second_f1_values
    non_zero_diffs = np.count_nonzero(differences)

    if non_zero_diffs < 5:
        print(f"WARNING: Only {non_zero_diffs} non-zero differences found.")
        print("With n < 5, it is mathematically impossible to reach p < 0.05.")

    print(f"{selector1.value[0]} - ({selector2.value[0]}): {first_f1_values}")
    print(f"{selector1.value[0]} - ({selector3.value[0]}): {second_f1_values}")

    shapiro_stat, shapiro_p = stats.shapiro(differences)
    print(f"Shapiro-Wilk Statistics: {shapiro_stat:.4f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    plt.figure(figsize=(6, 4))
    stats.probplot(differences, dist="norm", plot=plt)
    plt.title("Q-Q Plot of F1 Score Differences")
    plt.grid(True)
    plt.show()

    if shapiro_p > 0.05:
        print("Result: Differences look Normal (p > 0.05). Use Paired T-Test.")
        t_stat, p_val = stats.ttest_rel(first_f1_values, second_f1_values, alternative="greater")
        print(f"Paired T-test p-value: {p_val:.4f}")
    else:
        print("Result: Differences are NOT Normal (p <= 0.05). Use Wilcoxon Signed-Rank Test.")
        w_stat, p_val = stats.wilcoxon(first_f1_values, second_f1_values, alternative="greater")
        print(f"Wilcoxon p-value: {p_val:.4f}")

    try:
        stat, p_value = wilcoxon(
            first_f1_values, second_f1_values, alternative="greater", zero_method="wilcox"
        )

        print(f"Results for {selector1.value[0]}:")
        print(f"Median Diff: {np.median(differences):.4f}")
        print(f"Wilcoxon Statistics: {stat}")
        print(f"P-value (one-sided): {p_value:.4f}")

    except ValueError as e:
        print(f"Error calculating Wilcoxon: {e}")


    def statistic(x, y):
        return np.mean(x - y)


    print("--------------- Perm test:")
    res = stats.permutation_test(
        (first_f1_values, second_f1_values),
        statistic,
        permutation_type="samples",
        alternative="greater",
    )

    print(f"One-sided P-value: {res.pvalue:.5f}")
    return


if __name__ == "__main__":
    app.run()

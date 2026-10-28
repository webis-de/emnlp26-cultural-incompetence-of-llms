import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation for $\mathcal{H_3}$
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
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning

    return (
        Literal,
        UndefinedMetricWarning,
        alt,
        confusion_matrix,
        f1_score,
        get_args,
        mo,
        path,
        pd,
        precision_score,
        re,
        recall_score,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup
    """)
    return


@app.cell
def _():

    stereotype_context = ["nl", "en", "zh"]
    BASE_PATH = "./"

    # stereotype_context = ["nl", "en"]
    prompt_language = ["nl", "en", "zh"]
    # prompt_language = ["nl", "en"]
    input_language = ["nl", "en", "zh"]
    # input_language = ["nl", "en"]

    sizes = [10, 50, 100]
    SEEDS = [1, 5, 15, 25, 35]

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
    return (
        BASE_PATH,
        SEEDS,
        model_setting_ordering,
        prompt_language,
        sizes,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the inference data
    """)
    return


@app.cell
def _(
    SEEDS,
    UndefinedMetricWarning,
    check_for_yes_no_chinese,
    confusion_matrix,
    f1_score,
    get_predicted_label_no_restriction,
    get_wrong_prediction,
    path,
    pd,
    precision_score,
    recall_score,
    shades_dfs_context_and_lang_specific,
    shades_dfs_context_specific,
    sizes,
    warnings,
):
    instruction_tuning_lang = ["en", "nl"]
    context_languages = ["en", "nl", "zh"]
    prompting_languages = ["en", "nl", "zh"]

    sub_results = {}

    for language in instruction_tuning_lang:
        for size in sizes:
            for _context_sub in context_languages:
                for _prompt_lang_sub in prompting_languages:
                    spec_dataset_name_subsample = f"context_{_context_sub}_pl_{_prompt_lang_sub}"
                    shades_gt = shades_dfs_context_and_lang_specific[spec_dataset_name_subsample]

                    subsampling_seed_f1s = []
                    subsampling_seed_results = []

                    for seed_sub in SEEDS:
                        file_path = path.join(
                            BASE_PATH,
                            "output",
                            "model-predictions",
                            "subsampling",
                            f"seed{seed_sub}",
                            f"llama31-ultrachat-subsample-{size}k-{language}_context_{_context_sub}_task_{_prompt_lang_sub}_instance_{_prompt_lang_sub}.csv",
                        )

                        if not path.isfile(file_path):
                            raise FileExistsError(file_path)
                        df_read = pd.read_csv(file_path, sep=";")

                        if _prompt_lang_sub == "zh":
                            df_read["predicted_score"] = df_read["output"].apply(
                                check_for_yes_no_chinese
                            )
                        else:
                            df_read["predicted_score"] = df_read["output"].apply(
                                get_predicted_label_no_restriction
                            )

                        df_read["original_predictions"] = df_read["predicted_score"]

                        # If we cannot extract a prediction, we measure it as wrong prediction
                        df_read["predicted_score"] = df_read.apply(
                            lambda x: (
                                get_wrong_prediction(x)
                                if x["predicted_score"] == "No prediction"
                                else x["predicted_score"]
                            ),
                            axis=1,
                        )

                        if len(df_read) != len(shades_gt["ground_truth"]):
                            print(
                                f"Error parsing labels for prompt {_prompt_lang_sub} and seed {seed_sub}, input {_prompt_lang_sub}, {size} has a different length as groundtruth"
                            )
                            print(
                                f"{len(df_read)} and expected {len(shades_dfs_context_specific[_context_sub]['ground_truth'])}"
                            )
                        y_pred_sub = df_read["predicted_score"]
                        f1_macro_sub = f1_score(
                            df_read["ground_label"].apply(lambda x: 1 if x == "stereotype" else 0),
                            y_pred_sub,
                            average="macro",
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("error", category=UndefinedMetricWarning)
                            try:
                                precision_sub = precision_score(
                                    df_read["ground_label"].apply(
                                        lambda x: 1 if x == "stereotype" else 0
                                    ),
                                    y_pred_sub,
                                    pos_label=0,
                                )
                            except UndefinedMetricWarning:
                                print(f"Warning triggered for file: {file_path}")
                                # Handle the failure (e.g., set precision to 0 or skip)
                                precision_sub = 0.0

                        recall_sub = recall_score(
                            df_read["ground_label"].apply(lambda x: 1 if x == "stereotype" else 0),
                            y_pred_sub,
                            pos_label=0,
                        )

                        subsampling_seed_f1s.append(f1_macro_sub)
                        tn_sub, fp_sub, fn_sub, tp_sub = (
                            confusion_matrix(
                                df_read["ground_label"].apply(lambda x: 1 if x == "stereotype" else 0),
                                y_pred_sub,
                            )
                            .ravel()
                            .tolist()
                        )

                        subsampling_seed_results.append(
                            {
                                "model": f"llama31-ultrachat-subsample-{language}",
                                "seed": seed_sub,
                                "input_language": _prompt_lang_sub,
                                "prompt_language": _prompt_lang_sub,
                                "f1_macro": f1_macro_sub,
                                "true_positive": tp_sub,
                                "true_negative": tn_sub,
                                "false_positive": fp_sub,
                                "false_negative": fn_sub,
                                "precision": precision_sub,
                                "recall": recall_sub,
                                "size": size,
                                "stereotype_context": _context_sub,
                                "instruction_tuning_language": language,
                            }
                        )

                    if subsampling_seed_f1s:
                        avg_f1_sub = sum(subsampling_seed_f1s) / len(subsampling_seed_f1s)
                    else:
                        avg_f1_sub = None

                    sub_results[
                        f"ultrachat-{language}-{size}-c-{_context_sub}-p-{_prompt_lang_sub}"
                    ] = {
                        "per_seed": subsampling_seed_results,
                        "macro_f1_avg": avg_f1_sub,
                    }


    # Convert results to DataFrame for easier access
    subsampling_all_model_results = [
        {
            "model": s_result["model"],
            "size": s_result["size"],
            "input_language": s_result["input_language"],
            "prompt_language": s_result["prompt_language"],
            "stereotype_context": s_result["stereotype_context"],
            "instruction_tuning_language": s_result["instruction_tuning_language"],
            "seed": s_result["seed"],
            "f1_macro": s_result["f1_macro"],
            "true_positive": s_result["true_positive"],
            "true_negative": s_result["true_negative"],
            "false_positive": s_result["false_positive"],
            "false_negative": s_result["false_negative"],
            "false_negative_rate": s_result["false_negative"]
            / (s_result["false_negative"] + s_result["true_positive"]),
            "false_positive_rate": s_result["false_positive"]
            / (s_result["false_positive"] + s_result["true_negative"]),
            "true_negative_rate": s_result["true_negative"]
            / (s_result["true_negative"] + s_result["false_positive"]),
            "true_positive_rate": s_result["true_positive"]
            / (s_result["true_positive"] + s_result["false_negative"]),
            "recall": s_result["recall"],
            "precision": s_result["precision"],
            "average": sub_results[model]["macro_f1_avg"],
        }
        for model in sub_results.keys()
        for s_result in sub_results[model]["per_seed"]
    ]
    subsampling_results = pd.DataFrame(data=subsampling_all_model_results)
    return (subsampling_results,)


@app.cell
def _(subsampling_results):
    subsampling_results
    return


@app.cell
def _(subsampling_results):
    _filtered_df = subsampling_results[
        (subsampling_results["prompt_language"] == subsampling_results["input_language"])
        # & (subsampling_results["prompt_language"] == subsampling_results["stereotype_context"])
        & (subsampling_results["stereotype_context"] == "nl")
        #& (subsampling_results["model"].str.contains("ultrachat-nl"))
    ].copy()
    _filtered_df["model+size_combined"] = _filtered_df["model"] + " - " + _filtered_df["size"].astype(str)
    _filtered_df.groupby("model+size_combined")["f1_macro"].mean()
    #_filtered_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotting for $\mathcal{H}_{3}$
    """)
    return


@app.cell
def _(alt, model_setting_ordering, subsampling_results):
    display_df = subsampling_results[
        (subsampling_results["prompt_language"] == subsampling_results["prompt_language"])
        & (subsampling_results["prompt_language"] == subsampling_results["stereotype_context"])
    ].copy()
    display_df["identifier"] = "Ultrachat-" + str(display_df["instruction_tuning_language"])


    _chart = (
        alt.Chart(display_df)
        .encode(
            alt.X("size", sort=model_setting_ordering, title="Dataset size"),
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
            alt.Color("instruction_tuning_language", legend=None),
            alt.Shape("instruction_tuning_language", legend=None),
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
    last_setting = model_setting_ordering[-1]
    _labels = (
        _chart.mark_text(
            align="left",
            baseline="middle",
            dx=10,  # Offset to the right
        )
        .encode(text="identifier", detail="stereotype_context")
        .transform_filter(
            # Only show this label on the very last x-axis step
            alt.datum.model_setting == last_setting
        )
    )

    # Plot all the things
    # print("Note: in this plot, prompt language == input language (always), for consistency")

    # (_lines + _text + _labels).save("output/evaluation/h3.png", scale_factor=2.0)
    # _lines + _text + _labels
    sub_final_chart = (
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
        .properties(title=f"Test-plot")
    )

    print("Note: in this plot, prompt language == input language (always), for consistency")


    sub_final_chart
    return


if __name__ == "__main__":
    app.run()

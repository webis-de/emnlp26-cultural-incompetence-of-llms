import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare English Stackoverflow Data

    _Aim is to follow the same format as https://huggingface.co/datasets/BramVanroy/stackoverflow-chat-dutch in order to have this dataset as direct English comparison._
    """)
    return


@app.cell
def _():
    import json
    import re

    import datasets
    import marimo as mo
    import pandas as pd

    return datasets, json, mo, re


@app.cell
def _(json):
    with open("./data/stackoverflow_chat_data.json", "r") as f:
        stackoverflow_data = json.load(f)

    stackoverflow_data[5]
    return (stackoverflow_data,)


@app.cell
def _(mo, re, stackoverflow_data):
    parsed_data = []
    for i, conversation in mo.status.progress_bar(
        enumerate(stackoverflow_data), total=len(stackoverflow_data)
    ):
        split_conversation = re.split(
            r"(?=(?:\[\|Human\|\]|\[\|AI\|\]))", stackoverflow_data[1713]["input"]
        )
        # Remove trailing whitespaces
        split_conversation_postproc = [
            {
                "content": t.replace("[|Human|]", "").replace("[|AI|]", "").strip(),
                "role": "user" if t.startswith("[|Human|]") else "assistant",
            }
            for t in split_conversation
            if t != "[|Human|] "
        ]

        parsed_data.append(
            {
                "prompt": split_conversation[0].strip(),
                "prompt_id": i,
                "messages": split_conversation_postproc,
            }
        )
    return (parsed_data,)


@app.cell
def _(datasets, parsed_data):
    ds = datasets.Dataset.from_list(parsed_data).train_test_split(test_size=0.1)

    ds["train"].to_parquet("./output/stackoverflow-en/data/train.parquet")
    ds["test"].to_parquet("./output/stackoverflow-en/data/test.parquet")
    return


if __name__ == "__main__":
    app.run()

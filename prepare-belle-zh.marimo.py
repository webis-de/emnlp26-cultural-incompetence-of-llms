import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import re
    import datasets
    return datasets, json, mo


@app.cell
def _(json):
    belle_data = []

    with open("./data/Belle_open_source_0.5M.json", "r", encoding="utf-8") as f:
        for line in f:
            # Parse each line as a JSON object
            belle_data.append(json.loads(line))

    belle_data[5]
    return (belle_data,)


@app.cell
def _(belle_data, mo):
    parsed_data = []
    for i, conversation in mo.status.progress_bar(enumerate(belle_data), total=len(belle_data)):
        parsed_data.append(
            {
                "prompt": conversation["instruction"],
                "prompt_id": i,
                "messages": [
                    {"role": "user", "content": conversation["instruction"]},
                    {"role": "assistant", "content": conversation["output"]},
                ],
            }
        )
    return (parsed_data,)


@app.cell
def _(parsed_data):
    parsed_data[5]
    return


@app.cell
def _(datasets, parsed_data):
    ds = datasets.Dataset.from_list(parsed_data).train_test_split(test_size=0.1)

    ds["train"].to_parquet("./output/belle-0.5m-zh/data/train.parquet")
    ds["test"].to_parquet("./output/belle-0.5m-zh/data/test.parquet")
    return


if __name__ == "__main__":
    app.run()

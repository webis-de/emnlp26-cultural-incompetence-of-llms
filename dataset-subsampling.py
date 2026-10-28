import os

from datasets import load_dataset


def subsample_ultrachat():
    dataset_name = "ultrachat_200k"
    # dataset_name = "ultrachat_400k_nl"
    dataset_basepath = "./data"

    ds_train = load_dataset(
        os.path.join(dataset_basepath, dataset_name), split="train_sft"
    )
    ds_test = load_dataset(
        os.path.join(dataset_basepath, dataset_name), split="test_sft"
    )

    target_sizes = [100000, 50000, 10000]
    total_rows = len(ds_train)

    base_output_dir = os.path.join("output", "subsets")
    os.makedirs(base_output_dir, exist_ok=True)

    for i, size in enumerate(target_sizes):
        if size > total_rows:
            print(f"Skipping {size}: larger than source ({total_rows})")
            continue

        print(f"Generating independent {size} sample")

        current_subset = ds_train.shuffle(seed=42 + i).select(range(size))

        output_dir = os.path.join(
            base_output_dir,
            f"{dataset_name}-subsample__{size // 1000}k",
            "data",
        )
        os.makedirs(output_dir, exist_ok=True)

        current_subset.to_parquet(os.path.join(output_dir, "train.parquet"))
        # Test set remains the same, we only subsample training set
        ds_test.to_parquet(os.path.join(output_dir, "test.parquet"))

        print(f"Saved to {output_dir}")


if __name__ == "__main__":
    subsample_ultrachat()
    print("Done.")

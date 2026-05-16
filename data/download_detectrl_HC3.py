from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
HC3_DATASET_ID = "Hello-SimpleAI/HC3"
DETECTRL_REPO = "NLP2CT/DetectRL"
DETECTRL_TASKS = {
    "Task1": [
        "data_mixing_attacks_test.json",
        "multi_domains_arxiv_test.json",
        "multi_domains_arxiv_train.json",
        "multi_domains_writing_prompt_test.json",
        "multi_domains_writing_prompt_train.json",
        "multi_domains_xsum_test.json",
        "multi_domains_xsum_train.json",
        "multi_domains_yelp_review_test.json",
        "multi_domains_yelp_review_train.json",
        "multi_llms_ChatGPT_test.json",
        "multi_llms_ChatGPT_train.json",
        "multi_llms_Claude-instant_test.json",
        "multi_llms_Claude-instant_train.json",
        "multi_llms_Google-PaLM_test.json",
        "multi_llms_Google-PaLM_train.json",
        "multi_llms_Llama-2-70b_test.json",
        "multi_llms_Llama-2-70b_train.json",
        "paraphrase_attacks_llm_test.json",
        "paraphrase_attacks_llm_train.json",
        "perturbation_attacks_llm_test.json",
        "perturbation_attacks_llm_train.json",
        "prompt_attacks_llm_test.json",
        "prompt_attacks_llm_train.json",
    ],
    "Task2": [
        "data_mixing_attacks_test.json",
        "multi_domains_arxiv_test.json",
        "multi_domains_arxiv_train.json",
        "multi_domains_writing_prompt_test.json",
        "multi_domains_writing_prompt_train.json",
        "multi_domains_xsum_test.json",
        "multi_domains_xsum_train.json",
        "multi_domains_yelp_review_test.json",
        "multi_domains_yelp_review_train.json",
        "multi_llms_ChatGPT_test.json",
        "multi_llms_ChatGPT_train.json",
        "multi_llms_Claude-instant_test.json",
        "multi_llms_Claude-instant_train.json",
        "multi_llms_Google-PaLM_test.json",
        "multi_llms_Google-PaLM_train.json",
        "multi_llms_Llama-2-70b_test.json",
        "multi_llms_Llama-2-70b_train.json",
        "paraphrase_attacks_llm_test.json",
        "paraphrase_attacks_llm_train.json",
        "perturbation_attacks_llm_test.json",
        "perturbation_attacks_llm_train.json",
        "prompt_attacks_llm_test.json",
        "prompt_attacks_llm_train.json",
    ],
    "Task3": [
        "cross_length_20_test.json",
        "cross_length_40_test.json",
        "cross_length_60_test.json",
        "cross_length_80_test.json",
        "cross_length_100_test.json",
        "cross_length_120_test.json",
        "cross_length_140_test.json",
        "cross_length_160_test.json",
        "cross_length_180_test.json",
        "cross_length_200_test.json",
        "cross_length_220_test.json",
        "cross_length_240_test.json",
        "cross_length_260_test.json",
        "cross_length_280_test.json",
        "cross_length_300_test.json",
        "cross_length_320_test.json",
        "cross_length_340_test.json",
        "cross_length_360_test.json",
    ],
    "Task4": [
        "data_mixing_attacks_test.json",
        "data_mixing_attacks_train.json",
        "direct_prompt_test.json",
        "direct_prompt_train.json",
        "paraphrase_attacks_human_test.json",
        "paraphrase_attacks_human_train.json",
        "perturbation_attacks_human_test.json",
        "perturbation_attacks_human_train.json",
    ],
}


def load_detectrl_file(task: str, filename: str) -> Dataset:
    url = f"https://raw.githubusercontent.com/{DETECTRL_REPO}/main/Benchmark/Tasks/{task}/{filename}"
    dataset = load_dataset("json", data_files=url, split="train")
    return dataset


def summarize_split_counts(name: str, dataset: DatasetDict | Dataset) -> None:
    if isinstance(dataset, DatasetDict):
        print(f"\n{name} split counts:")
        for split_name, split_dataset in dataset.items():
            print(f"  {split_name}: {len(split_dataset):,}")
        return

    print(f"\n{name} split counts:")
    print(f"  train: {len(dataset):,}")


def summarize_attack_types(name: str, dataset: DatasetDict | Dataset) -> None:
    combined_frames: list[pd.DataFrame] = []
    if isinstance(dataset, DatasetDict):
        items = dataset.items()
    else:
        items = [("train", dataset)]

    for split_name, split_dataset in items:
        frame = split_dataset.to_pandas()
        frame["split"] = split_name
        combined_frames.append(frame)

    combined = pd.concat(combined_frames, ignore_index=True)
    print(f"\n{name} attack_type counts:")
    for candidate_column in ["attack_type", "data_type", "subset", "category"]:
        if candidate_column in combined.columns:
            print(combined[candidate_column].fillna("<missing>").value_counts(dropna=False).to_string())
            return
    print("  attack_type column not found")


def save_split_parquet(dataset: DatasetDict | Dataset, dataset_name: str) -> None:
    output_dir = RAW_DIR / dataset_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, DatasetDict):
        items = dataset.items()
    else:
        items = [("train", dataset)]

    for split_name, split_dataset in items:
        output_path = output_dir / f"{split_name}.parquet"
        split_dataset.to_parquet(str(output_path))
        print(f"Saved {dataset_name} {split_name} to {output_path.relative_to(ROOT_DIR)}")


def download_hc3() -> tuple[DatasetDict | Dataset, str]:
    return load_dataset(HC3_DATASET_ID), HC3_DATASET_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DetectRL and HC3 from Hugging Face and save them as parquet files.")
    parser.add_argument(
        "--hc3-id",
        default=HC3_DATASET_ID,
        help="Hugging Face dataset repo name for HC3.",
    )
    args = parser.parse_args()

    print(f"Loaded DetectRL from GitHub repo: {DETECTRL_REPO}")
    for task_name, file_names in DETECTRL_TASKS.items():
        task_dir = RAW_DIR / "detectrl" / task_name.lower()
        task_dir.mkdir(parents=True, exist_ok=True)

        task_frames: list[pd.DataFrame] = []
        print(f"\nProcessing {task_name}:")
        for file_name in file_names:
            split_name = Path(file_name).stem
            dataset = load_detectrl_file(task_name, file_name)
            frame = dataset.to_pandas()
            frame["split"] = split_name
            frame["task"] = task_name
            if "attack_type" not in frame.columns and "data_type" in frame.columns:
                frame["attack_type"] = frame["data_type"]
            task_frames.append(frame)

            output_path = task_dir / f"{split_name}.parquet"
            dataset.to_parquet(str(output_path))
            print(f"  {split_name}: {len(dataset):,} rows -> {output_path.relative_to(ROOT_DIR)}")

        task_dataset = Dataset.from_pandas(pd.concat(task_frames, ignore_index=True), preserve_index=False)
        summarize_attack_types(task_name, task_dataset)

    # HC3 has multiple configs; load them all and combine into a DatasetDict
    hc3_configs = ["wikipedia", "reddit", "open_domain", "finance"]
    hc3_dict = {}
    print(f"\nLoading HC3 configs from: {args.hc3_id}")
    for config in hc3_configs:
        try:
            dataset = load_dataset(args.hc3_id, name=config, trust_remote_code=True)
            # Flatten the dataset if it has splits, otherwise use as-is
            if isinstance(dataset, DatasetDict):
                for split_name, split_data in dataset.items():
                    hc3_dict[f"{config}_{split_name}"] = split_data
            else:
                hc3_dict[config] = dataset
            print(f"  Loaded config: {config}")
        except Exception as e:
            print(f"  Warning: Failed to load config {config}: {e}")
    
    hc3 = DatasetDict(hc3_dict) if hc3_dict else None
    if hc3 is None:
        print("Warning: Could not load any HC3 configs")
    else:
        summarize_split_counts("HC3", hc3)
        summarize_attack_types("HC3", hc3)
        save_split_parquet(hc3, "hc3")


if __name__ == "__main__":
    main()
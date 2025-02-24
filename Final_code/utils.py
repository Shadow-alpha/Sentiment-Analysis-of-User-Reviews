import os
import math
import inspect
from typing import Optional, Text

from accelerate import Accelerator
from transformers.utils import is_peft_available
from datasets import load_dataset, Dataset, DatasetDict


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_prompt_dataset(dataset_name: Text) -> Dataset:
    # make sure the returned dataset has "train" and "test" split
    seed = 1
    if dataset_name == "GAIR/lima":
        dataset = load_dataset(dataset_name)
        dataset = dataset.filter(lambda row: row["source"] != "multi_turn")
        dataset = dataset.map(lambda row: {"query": row["conversations"][0]})
    elif dataset_name == "HuggingFaceH4/ultrachat_200k":
        dataset = load_dataset(dataset_name)
        train_split = dataset["train_sft"].shuffle(seed=seed).select(range(50000))
        test_split = dataset["test_sft"].shuffle(seed=seed).select(range(10000))
        dataset = DatasetDict({"train": train_split, "test": test_split})
        dataset = dataset.rename_columns({"prompt": "query"}).remove_columns(["prompt_id", "messages"])
    elif dataset_name == "HuggingFaceH4/helpful-anthropic-raw":
        dataset = load_dataset("HuggingFaceH4/helpful-anthropic-raw")["train"]
        dataset = dataset.shuffle(seed=seed).select(range(60000))
        dataset = dataset.train_test_split(test_size=10000)
        dataset = dataset.rename_columns({"instruction": "query"}).remove_columns(["demonstration"])
    else:
        raise NotImplementedError
    return dataset


def split_dataset_per_rank(dataset: Dataset, rank: int, world_size: int) -> Dataset:
    assert 1 <= rank <= world_size
    split_size = math.ceil(len(dataset) / world_size)
    dataset = dataset.select(range(
        (rank-1)*split_size, 
        min((rank)*split_size, len(dataset))
    ))
    return dataset


def get_output_path(output_dir: str, rank: int, world_size: int, suffix: str = "jsonl") -> Text:
    assert 1 <= rank <= world_size
    return os.path.join(output_dir, f"{str(rank).zfill(5)}-of-{str(world_size).zfill(5)}.{suffix}")


@Accelerator().on_local_main_process
def print_local_main(text):
    print(text)

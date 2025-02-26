import argparse
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict

from llm_knowledge_injection_survey.datasets.tasks import (
    NthLetterTask,
    SimpleWordTask,
    WordLengthTask,
)


def _create_dataset_split(samples: list[Any]) -> Dataset:
    """Create a dataset split from the samples."""
    return Dataset.from_dict(
        {
            "prompt": [sample.prompt for sample in samples],
            "completion": [sample.completion for sample in samples],
        }
    )


def create_simple_task_dataset(
    tasks: list[SimpleWordTask],
    num_train_samples: int,
    num_validation_samples: int,
    num_test_samples: int,
    seed: int = 128,
) -> DatasetDict:
    """Create a dataset for simple word tasks."""
    rng = np.random.default_rng(seed=seed)
    num_total_samples = num_train_samples + num_validation_samples + num_test_samples
    samples = []
    num_samples_per_task = 1 + num_total_samples // len(
        tasks
    )  # Add one to ensure we have enough samples
    for task in tasks:
        samples.extend(task.sft_sample(num_samples_per_task))
    rng.shuffle(samples)
    samples = samples[:num_total_samples]  # Drop excess samples
    train_samples = samples[:num_train_samples]
    validation_samples = samples[
        num_train_samples : num_train_samples + num_validation_samples
    ]
    test_samples = samples[-num_test_samples:]
    return DatasetDict(
        {
            "train": _create_dataset_split(train_samples),
            "validation": _create_dataset_split(validation_samples),
            "test": _create_dataset_split(test_samples),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_samples", type=int, default=100000)
    parser.add_argument("--num_validation_samples", type=int, default=10000)
    parser.add_argument("--num_test_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--save_path", type=str, default="simple-word-tasks")
    args = parser.parse_args()

    save_path = Path(args.save_path)
    if save_path.suffix != ".hf":
        save_path = save_path.parent / (save_path.name + ".hf")
    save_path.mkdir(parents=True, exist_ok=True)

    tasks = [WordLengthTask(seed=args.seed), NthLetterTask(seed=args.seed)]
    dataset: DatasetDict = create_simple_task_dataset(
        tasks,
        args.num_train_samples,
        args.num_validation_samples,
        args.num_test_samples,
        seed=args.seed,
    )
    dataset.save_to_disk(save_path)

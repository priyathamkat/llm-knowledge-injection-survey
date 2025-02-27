import argparse
from pathlib import Path
from typing import Literal

from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def train(
    training_type: Literal["sft", "peft"],
    dataset_path: str,
    model: str,
    batch_size: int,
    num_epochs: int,
    output_dir: Path,
    logging_steps: int,
    save_steps: int,
):
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    if training_type == "sft":
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        config = AutoConfig.from_pretrained(
            model, attn_implementation="flash_attention_2"
        )
        model = AutoModelForCausalLM.from_config(config)
    elif training_type == "peft":
        max_seq_length = 1024
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model,
            max_seq_length=max_seq_length,
            dtype="bf16",
            load_in_4bit=False,
            gpu_memory_utilization=0.6,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(f"Unsupported training type: {training_type}")

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataset_num_proc=32,
        dataloader_num_workers=32,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        num_train_epochs=num_epochs,
        bf16=True,
        logging_first_step=True,
        logging_steps=logging_steps,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=logging_steps,
        save_steps=save_steps,
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_type", choices=["sft", "peft"], default="sft")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--logging_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=1000)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir = output_dir / f"{args.model}-word-task-{args.training_type}"

    train(
        args.training_type,
        args.dataset_path,
        args.model,
        args.batch_size,
        args.num_epochs,
        output_dir,
        args.logging_steps,
        args.save_steps,
    )

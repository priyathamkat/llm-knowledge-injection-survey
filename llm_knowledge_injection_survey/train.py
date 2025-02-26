import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def train(
    dataset_path: str,
    model: str,
    batch_size: int,
    num_epochs: int,
    output_dir: str,
    logging_steps: int,
    save_steps: int,
):
    dataset = load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    config = AutoConfig.from_pretrained(model, attn_implementation="flash_attention_2")
    model = AutoModelForCausalLM.from_config(config)

    output_dir = Path(output_dir)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataset_num_proc=32,
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

    trainer.save_model(output_dir / "final_model.safetensors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--logging_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=3000)

    args = parser.parse_args()

    train(
        args.dataset_path,
        args.model,
        args.batch_size,
        args.num_epochs,
        args.output_dir,
        args.logging_steps,
        args.save_steps,
    )

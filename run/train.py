import sys
import logging

import pathlib
import click
from unsloth import FastLanguageModel

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import torch

from ethexco.util import validate, model_setup, prepare_dataset

logger = logging.getLogger(__name__)


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option(
    "--validation-ds-path", type=click.Path(path_type=pathlib.Path), required=True
)
@click.option(
    "--model-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
)
@click.option(
    "--model-name",
    type=click.STRING,
    default="unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
)
@click.option(
    "--max-steps",
    type=click.INT,
    default=30,
)
@click.option(
    "--n-repeat",
    type=click.INT,
    default=3,
)
@click.option("--report-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--fresh", type=click.BOOL, is_flag=True, default=False)
def main(
    dataset_path,
    model_name,
    model_path,
    report_path,
    max_steps,
    n_repeat,
    validation_ds_path,
    fresh,
):
    model_path = model_path.expanduser()

    max_seq_length = 2048
    if fresh:
        model, tokenizer = model_setup(model_name, max_seq_length=max_seq_length)
    else:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path.as_posix(),
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        except Exception:
            logger.info("no valid model found; init fresh model")
            model, tokenizer = model_setup(model_name, max_seq_length=max_seq_length)

    dataset = prepare_dataset(dataset_path, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            # num_train_epochs = 1, # For longer training runs!
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=13,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    # used_percentage = round(used_memory / max_memory * 100, 3)
    # lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    logger.info(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    logger.info(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    logger.info(f"Peak reserved memory = {used_memory} GB.")

    validate(validation_ds_path, model, tokenizer, n_repeat, report_path)

    # model.save_pretrained_merged(f"{model_path}.merged", tokenizer, save_method="lora")
    # model.save_pretrained_merged(
    #     f"{model_path}.16",
    #     tokenizer,
    #     save_method="merged_16bit",
    # )
    # model.save_pretrained_merged(
    #     f"{model_path}.4",
    #     tokenizer,
    #     save_method="`merged_4bit_forced",
    # )
    #
    # model.save_pretrained_gguf(
    #     f"{model_path}.q8",
    #     tokenizer,
    # )
    # model.save_pretrained_gguf(
    #     f"{model_path}.q4_k_m", tokenizer, quantization_method="q4_k_m"
    # )
    # tokenizer.save_pretrained_merged(f"{model_path}.merged")
    # model.save_pretrained_gguf("dir", tokenizer, quantization_method="q4_k_m")
    # model.save_pretrained_gguf("dir", tokenizer, quantization_method="q8_0")
    # model.save_pretrained_gguf(
    #     f"{model_path}.f16", tokenizer, quantization_method="f16"
    # )
    # model.save_pretrained_gguf(
    #     f"{model_path}.q4", tokenizer, quantization_method="q4_k_m"
    # )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

import logging

import sys

import pathlib
import click
import suthing
from datasets import Dataset
import numpy as np
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from itertools import product
from unsloth import FastLanguageModel

from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextStreamer
from unsloth import is_bfloat16_supported
import torch

logger = logging.getLogger(__name__)


def validate(
    validation_ds_path: pathlib.Path,
    model,
    tokenizer,
    n_repeat,
    report_path: pathlib.Path,
):
    vds = suthing.FileHandle.load(validation_ds_path.expanduser())

    # system_messages = [ ... {"person": "William James"} ..]
    system_messages = vds["system"]["characters"]
    # instructions = [ ... {"simple": "Respond concisely." } , ..]
    instructions = vds["system"]["instructions"]
    # questions = [
    #         {
    #             "body": "What is the ultimate criterion for determining the rightness of an action?",
    #             "type": "simple",
    #         },
    # ... ]

    questions = vds["questions"]

    FastLanguageModel.for_inference(model)

    convos = []
    answers = []

    for system, question in product(system_messages, questions):
        person = system.get("name")
        qtype = question["type"]
        q = question["body"]
        instruction = instructions[qtype]
        s = f"You are {person}, a philosopher. {instruction}"
        convos += [
            [
                {
                    "role": "user",
                    "content": f"{s}\n\n{q}",
                    "system": s,
                    "question": q,
                    "person": person,
                }
            ]
        ]

    ixs = list(range(len(convos))) * n_repeat
    np.random.shuffle(ixs)
    report = []

    for ix in ixs:
        convo = convos[ix]
        input_ids = tokenizer.apply_chat_template(
            convo,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        logger.info("----------------------\n\n")
        logger.info(f"{convo[0]['content']}\n\n")
        tt = model.generate(
            input_ids,
            streamer=text_streamer,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # use_cache = True
        )
        answers += [tt]

        generated_tokens = tt[:, input_ids.shape[1] :]

        generated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        report += [
            {
                "person": convo[0]["person"],
                "question": convo[0]["question"],
                "system": convo[0]["system"],
                "answer": generated_text,
            }
        ]

    suthing.FileHandle.dump(report, report_path / "report.json")


def model_setup(model_name, max_seq_length):
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer


def prepare_dataset(dataset_path, tokenizer=None):
    dataset0 = Dataset.load_from_disk(dataset_path.expanduser())

    n_samples = len(dataset0)
    indices = np.random.choice(len(dataset0), size=n_samples, replace=False)
    dataset0_rs = dataset0.select(indices)

    dataset_shr = to_sharegpt(
        dataset0_rs,
        merged_prompt="{system}[[\nYour input is:\n{input}]]",
        output_column_name="output",
        conversation_extension=1,
    )

    dataset = standardize_sharegpt(dataset_shr)

    chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

    ### Instruction:
    {INPUT}

    ### Response:
    {OUTPUT}"""

    if tokenizer is not None:
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            chat_template=chat_template,
            # default_system_message = "You are a helpful assistant", << [OPTIONAL]
        )

    return dataset


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

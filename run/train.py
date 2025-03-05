import logging

import sys

import pathlib
import click
from datasets import Dataset
import numpy as np
from unsloth import to_sharegpt
from unsloth import standardize_sharegpt
from unsloth import apply_chat_template
from unsloth import FastLanguageModel

from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextStreamer
from unsloth import is_bfloat16_supported
import torch

logger = logging.getLogger(__name__)


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


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=pathlib.Path), required=True)
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
def main(dataset_path, model_name, model_path, max_steps):
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!

    model, tokenizer = model_setup(model_name, max_seq_length=max_seq_length)
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

    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        # default_system_message = "You are a helpful assistant", << [OPTIONAL]
    )

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

    model.save_pretrained(model_path)  # Local saving
    tokenizer.save_pretrained(model_path)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    # used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    # used_percentage = round(used_memory / max_memory * 100, 3)
    # lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    system_message = "You are William James, a philosopher. Respond concisely.\n"
    system_message = "You are John Stuart Mill, a philosopher. Respond concisely.\n"

    messages = [
        # {"role": "system", "content": f"{system_message}"},
        # {"role": "user", "content": f"Do moral truths hold universally?"},
        {
            "role": "user",
            "content": f"{system_message} Do moral truths hold universally?",
        },
    ]

    # prompt = f"<|system|>\n{system_message}\n<|user|>\n{user_message}\n<|assistant|> "
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        input_ids,
        streamer=text_streamer,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # use_cache = True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

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

logger = logging.getLogger(__name__)


def model_setup(model_name):
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
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
    required=True,
    default="unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
)
def main(
    dataset_path,
    model_name,
    model_path,
):
    model, tokenizer = model_setup(model_name)
    dataset0 = Dataset.load_from_disk(dataset_path.expanduser())

    n_samples = len(dataset0)
    indices = np.random.choice(len(dataset0), size=n_samples, replace=False)
    dataset0_rs = dataset0.select(indices)

    dataset_shr = to_sharegpt(
        dataset0_rs,
        merged_prompt="{system}[[\nYour input is:\n{input}]]",
        output_column_name="output",
        conversation_extension=3,
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

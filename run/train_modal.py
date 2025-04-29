import modal
import logging
import pathlib
from pathlib import Path
import os

remote_working_dir = "/wd"
remote_model_dir = "/models"

app = modal.App("test-mount")

data_path = Path("~/data/ethexco/wps/wp.a").expanduser()
model_path = Path("~/data/ethexco/models").expanduser()


image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("suthing")
    .pip_install("unsloth", gpu="T4")
    .run_commands("pip uninstall unsloth -y")
    .run_commands(
        "pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git@nightly git+https://github.com/unslothai/unsloth-zoo.git"
        )
    .add_local_dir(data_path, remote_path="/wd")
    .add_local_dir(model_path, remote_path="/models")
    .add_local_python_source("ethexco")
)



@app.function(image=image)
def test_mounts(wd_path):
    print(f"Contents of {wd_path}:")
    print(os.listdir(wd_path))


@app.function(image=image)
def train():
    from pathlib import Path
    from ethexco.util import validate, model_setup, prepare_dataset
    import unsloth
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported
    import torch

    model_name = "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"

    # model_path = Path(remote_working_dir) / "ethexco/models/" / model_name.split("/")[-1]
    dataset_path = Path("/content/wd/ethexco/datasets")
    dataset_name = "ethexco.dataset.ab"
    frames_path = Path("/content/wd/ethexco/frames")
    validation_ds_path = Path("/content/wd/ethexco/validation/validation.dataset.b.json")
    report_path = Path("/content/wd/ethexco/reports")

    # model_name = "unsloth/llama-3-8b-bnb-4bit"
    fresh = True
    max_seq_length = 2048
    max_steps = 20

    n_repeat = 1

@app.local_entrypoint()
def main():
    test_mounts.remote(remote_working_dir)
    test_mounts.remote(remote_model_dir)

    train.remote()


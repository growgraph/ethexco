import re
import logging
from itertools import product

import numpy as np
import suthing
from datasets import Dataset

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


import pathlib

from enum import StrEnum

from transformers import TextStreamer

logger = logging.getLogger(__name__)


class Frame(StrEnum):
    ASSUMPTIONS = "Assumptions"
    CONTEXT = "Context"
    METHOD = "Method"
    QUESTION = "Question"
    THESIS = "Thesis"
    TITLE = "Title"


frame_content = {
    Frame.QUESTION: "What questions or issues are being addressed in the text?",
    Frame.CONTEXT: "What is the context of the passage? Are there any historical references, references to particular places, group of people, time periods, or professions?",
    Frame.ASSUMPTIONS: "What assumptions are made, either explicitly or implicitly?",
    Frame.THESIS: "What is the main statement or conclusion of the text?",
    Frame.METHOD: "What logical or philosophical methods are used to develop their argument and reach their thesis?",
    Frame.TITLE: "What title would you give to the passage if it were an essay?",
}


def render_response(text: str, llm, onto_str: str | None = None):
    frames_str = "\n".join({f" - **{k}**:{v}" for k, v in frame_content.items()})

    prompt = f"""
Please process a text - a fragment from a philosophical book.
There are two independent tasks: task A and task B.
Task A. Analyze the provided text from the perspective of logical reasoning and provide answers to  the following elements:
        
{frames_str}
        
For Task A follow the instructions:
    
 - each answer must be placed in a block marked correspondingly, eg ```{Frame.CONTEXT} ...``` or ```{Frame.METHOD} ...```.  Do not use any markup other than that. 
 - {Frame.QUESTION} and {Frame.TITLE} are the most important elements.

Task B. Generate semantic triples in turtle (ttl) format from the text below.
         
For Task B follow the instructions:
 
 - mark extracted semantic triples as ```ttl ```.
 - use commonly known ontologies (RDFS, OWL, schema etc) to place encountered abstract entities/properties and facts within a broader ontology.
 - entities representing facts must use the namespace `@prefix cd: <https://growgraph.dev/current#> .` 
 - all entities from `cd:` namespace must IMPERATIVELY linked to entities from basic ontologies (RDFS, OWL etc), e.g. rdfs:Class, rdfs:subClassOf, rdf:Property, rdfs:domain, owl:Restriction, schema:Person schema:Organization, etc
 - all facts must form a connected graph with respect to namespace `cd`.
 - make semantic representation as atomic as possible.
 - keep in mind Task B is independent from Task A.

Below is the input text:
        
```input_text

{{input_text}}
```
    """

    parser = StrOutputParser()

    prompt = PromptTemplate(template=prompt, input_variables=["input_text"])

    chain = prompt | llm | parser

    response = chain.invoke({"input_text": text})
    return response


def extract_struct(text, key):
    # Pattern to match text between ```key and ```
    pattern = rf"```{key}(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def crawl_directories(input_path: pathlib.Path, suffixes=(".pdf", ".json")):
    file_paths: list[pathlib.Path] = []

    if not input_path.is_dir():
        print(f"The path {input_path} is not a valid directory.")
        return file_paths

    for file in input_path.rglob("*"):
        if file.is_file() and file.suffix in suffixes:
            file_paths.append(file)
    return file_paths


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

    try:
        from unsloth import (
            FastLanguageModel,
        )

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
    except ImportError as e:
        logger.error("Could not import unsloth")
        raise ImportError(f"ImportError: {e}")


def model_setup(model_name, max_seq_length):
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    try:
        from unsloth import (
            FastLanguageModel,
        )

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
    except ImportError as e:
        logger.error("Could not import unsloth")
        raise ImportError(f"ImportError: {e}")

    return model, tokenizer


def prepare_dataset(dataset_path, tokenizer=None):
    dataset0 = Dataset.load_from_disk(dataset_path.expanduser())

    n_samples = len(dataset0)
    indices = np.random.choice(len(dataset0), size=n_samples, replace=False)
    dataset0_rs = dataset0.select(indices)

    try:
        from unsloth import (
            to_sharegpt,
            standardize_sharegpt,
            apply_chat_template,
        )

        dataset_shr = to_sharegpt(
            dataset0_rs,
            merged_prompt="{system}[[\nYour input is:\n{input}]]",
            output_column_name="output",
            conversation_extension=1,
        )

        dataset = standardize_sharegpt(dataset_shr)

        chat_template = """Below are instructions that describe the tasks. Write responses that appropriately complete each request.
    
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
    except ImportError as e:
        logger.error("Could not import unsloth")
        raise ImportError(f"ImportError: {e}")

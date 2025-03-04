import logging
import sys

import pathlib
import click
import suthing
from datasets import Dataset
from ethexco.util import Frame
from ethexco.util import crawl_directories

logger = logging.getLogger(__name__)


# Prepare a function to convert each doc into a training example
def format_as_chat_example(doc):
    """
    Convert a doc into OpenAI-like chat format for unsloth apply_chat_template.
    You could swap to LLaMA or Mistral format if needed.
    """
    frames = doc.pop("response.structured")
    context = doc.pop("source.context")

    if set(frames) != set(Frame) or any(v is None for v in frames.values()):
        print(frames)
        return {}

    system_message = ""
    user_message = ""
    assistant_message = ""

    if "author" in context and context["author"] is not None:
        system_message += f"You are {context['author']}.\n"

    if "author.context" in context and context["author.context"] is not None:
        system_message += f"Character description : {context['author.context']}\n"

    if "title" in context and context["title"] is not None:
        system_message += (
            f"""The section is from a book called "{context['title']}".\n"""
        )

    if Frame.METHOD in frames and frames[Frame.METHOD] is not None:
        system_message += f"Your argumentation method : {frames[Frame.METHOD]}\n"

    if Frame.CONTEXT in frames and frames[Frame.CONTEXT] is not None:
        system_message += (
            f"The context of the question/problem : {frames[Frame.CONTEXT]}\n"
        )

    if Frame.ASSUMPTIONS in frames and frames[Frame.ASSUMPTIONS] is not None:
        system_message += f"The assumptions for this : {frames[Frame.ASSUMPTIONS]}\n"

    if Frame.QUESTION in frames and frames[Frame.QUESTION] is not None:
        user_message += f"{frames[Frame.QUESTION]}\n"

    if Frame.THESIS in frames and frames[Frame.THESIS] is not None:
        assistant_message += f"{frames[Frame.THESIS]}\n"

    return {
        "system": system_message,
        "input": user_message,
        "output": assistant_message,
    }


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--output-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option(
    "--dataset-name", type=click.STRING, required=True, default="ethexco.dataset.a"
)
def main(input_path, output_path, dataset_name):
    input_path = input_path.expanduser()

    input_path = input_path.expanduser()

    files = sorted(crawl_directories(input_path.expanduser()))

    agg = []
    for fname in files:
        data = suthing.FileHandle.load(fname)
        r = [format_as_chat_example(doc) for doc in data]
        r2 = [item for item in r if item.values() and all(v for v in item.values())]
        agg += r2

    dataset = Dataset.from_list(agg)
    print(dataset)
    dataset.save_to_disk(output_path / dataset_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

import logging
import sys

import pathlib
import click
import suthing
from datasets import Dataset
from ethexco.util import Frame, crawl_directories, parse_qa_pairs

logger = logging.getLogger(__name__)


def format_as_chat_example(doc):
    """
    Convert a doc into OpenAI-like chat format for unsloth apply_chat_template.
    You could swap to LLaMA or Mistral format if needed.
    """

    frame = doc.get("response_structured")
    response = doc.get("response")
    context = doc.get("source_context")
    text = frame.get("text")

    important_fields = {Frame.PROBLEM, Frame.THESIS}

    absent_fields = set(Frame) - set(frame)

    if absent_fields or any(frame[k] is None for k in important_fields):
        logger.warning(f"Absent fields: {absent_fields}")
        logger.warning(
            f"Among them Nones: fields: {sorted(k for k, v in frame.items() if v is None)}"
        )
        logger.warning(f"text : {text[:500]}")
        logger.warning(f"response : {response}")
        return {}

    system_message = ""

    if "author" in context and context["author"] is not None:
        system_message += f"You are {context['author']}.\n\n"

    if "author_context" in context and context["author_context"] is not None:
        system_message += f"Character description : {context['author_context']}\n\n"

    if "title" in context and context["title"] is not None:
        system_message += (
            f"""The section is from a book called "{context['title']}".\n\n"""
        )

    if Frame.METHOD in frame and frame[Frame.METHOD] is not None:
        system_message += f"Your argumentation method : {frame[Frame.METHOD]}\n\n"

    if Frame.CONTEXT in frame and frame[Frame.CONTEXT] is not None:
        system_message += (
            f"The context of the question/problem : {frame[Frame.CONTEXT]}\n\n"
        )

    if Frame.ASSUMPTIONS in frame and frame[Frame.ASSUMPTIONS] is not None:
        system_message += (
            f"The assumptions for this passage are: {frame[Frame.ASSUMPTIONS]}\n\n"
        )

    if Frame.PROBLEM in frame and frame[Frame.PROBLEM] is not None:
        system_message += f"The main problem raised: {frame[Frame.PROBLEM]}\n\n"

    if Frame.THESIS in frame and frame[Frame.THESIS] is not None:
        system_message += f"The thesis/statement about the problem: {frame[Frame.THESIS]}\n\n"

    qas_str = frame[Frame.QUESTION_ANSWER_PAIRS]
    if qas_str is not None and qas_str:
        qas = parse_qa_pairs(qas_str)
        return [{
            "system": system_message,
            "input": q,
            "output": a} for q, a in qas]
    else:
        return []


def format_as_essay_example(doc):
    """
    Convert a doc into OpenAI-like chat format for unsloth apply_chat_template.
    You could swap to LLaMA or Mistral format if needed.
    """
    frame = doc.get("response_structured")
    context = doc.get("source_context")
    text = frame.get("text")

    important_fields = {Frame.TITLE}

    absent_fields = set(Frame) - set(frame)

    if absent_fields or any(frame[k] is None for k in important_fields):
        logger.warning(f"Absent fields: {absent_fields}")
        logger.warning(
            f"Among them Nones: fields: {sorted(k for k, v in frame.items() if v is None)}"
        )
        return {}

    system_message = ""
    user_message = ""
    assistant_message = text

    if "author" in context and context["author"] is not None:
        system_message += f"You are {context['author']}.\n\n"

    if "author_context" in context and context["author_context"] is not None:
        system_message += f"Character description : {context['author_context']}\n\n"

    if Frame.TITLE in frame and frame[Frame.TITLE] is not None:
        user_message += f"Please write an essay titled: {frame[Frame.TITLE]}\n\n"

    if Frame.PROBLEM in frame and frame[Frame.PROBLEM] is not None:
        user_message += f"The essay addresses the following problem: {frame[Frame.PROBLEM]}\n\n"

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

    files = sorted(crawl_directories(input_path.expanduser()))

    agg = []
    for fname in files:
        data = suthing.FileHandle.load(fname)
        r = []
        for doc in data:
            r += format_as_chat_example(doc)
            r += [format_as_essay_example(doc)]
        r2 = [item for item in r if item.values() and all(v for v in item.values())]
        agg += r2

    dataset = Dataset.from_list(agg)
    dataset.save_to_disk(output_path / dataset_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

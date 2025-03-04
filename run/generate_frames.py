"""
1. read pdfs from `input_path`
2. process them using openai api (OPENAI_API_KEY loaded from .env found at `env_path`) using ontology found at `onto_path`
3. store triple as `.ttl` in `output_path`

"""

import logging

import sys

import pathlib
import click
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import suthing

from ethexco.util import render_response, extract_struct, Frame
from ethexco.util import crawl_directories

logger = logging.getLogger(__name__)


def process_unit(text) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini")

    response_raw = render_response(text=text, llm=llm)

    r = {}
    for u in Frame:
        uextracted = extract_struct(response_raw, u)
        r[u] = uextracted

    unit = {"response.raw": response_raw, "response.structured": r}
    return unit


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option(
    "--output-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
)
def main(
    input_path,
    env_path,
    output_path,
):
    _ = load_dotenv(dotenv_path=env_path.expanduser())
    input_path = input_path.expanduser()

    files = sorted(crawl_directories(input_path.expanduser()))
    for fname in files:
        data = suthing.FileHandle.load(fname)
        acc = []
        for chunk in data["chunks"]:
            ch = process_unit(chunk)
            ch["source.context"] = {
                k: data[k] for k in ["author", "title", "author.context"] if k in data
            }
            acc += [ch]

        suthing.FileHandle.dump(acc, output_path / (fname.stem + ".frame.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

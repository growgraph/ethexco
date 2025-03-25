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

    r: dict[Frame | str, str] = {}
    for u in Frame:
        uextracted = extract_struct(response_raw, u)
        r[u] = uextracted
    ttl = extract_struct(response_raw, "ttl")
    r["ttl"] = ttl

    unit = {"response_raw": response_raw, "response_structured": r}
    return unit


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--tail", type=click.INT)
@click.option(
    "--output-path",
    type=click.Path(path_type=pathlib.Path),
    required=True,
)
def main(input_path, env_path, output_path, tail):
    _ = load_dotenv(dotenv_path=env_path.expanduser())
    input_path = input_path.expanduser()

    files = sorted(crawl_directories(input_path.expanduser()))
    if tail is not None:
        files = files[-tail:]

    for fname in files:
        data = suthing.FileHandle.load(fname)
        acc = []
        if tail is not None:
            data = data[-tail:]
        for doc in data:
            chunks = doc.pop("chunks", [])
            context = {k: v for k, v in doc.items() if k != "text"}
            if tail is not None:
                chunks = chunks[-tail:]
            for chunk in chunks:
                ch = process_unit(chunk)
                ch["source_context"] = context
                acc += [ch]
        if tail is not None:
            fname_pre = f"{fname.stem}.tail.{tail}"
        else:
            fname_pre = fname.stem

        suthing.FileHandle.dump(acc, output_path / (fname_pre + ".frame.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

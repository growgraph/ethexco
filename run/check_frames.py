import logging
import sys

import pathlib
import click
import suthing
from datasets import Dataset
from ethexco.util import Frame, crawl_directories, parse_qa_pairs
import pandas as pd

logger = logging.getLogger(__name__)


def check(doc):
    """
    Convert a doc into OpenAI-like chat format for unsloth apply_chat_template.
    You could swap to LLaMA or Mistral format if needed.
    """

    frame = doc.get("response_structured")
    flags = {k: (True if (v is not None) and v else False) for k, v in frame.items()}
    return flags


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
def main(input_path):

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)


    input_path = input_path.expanduser()

    files = sorted(crawl_directories(input_path.expanduser()))

    agg = []
    for fname in files:
        data = suthing.FileHandle.load(fname)
        for doc in data:
            f = check(doc)
            f["source"] = fname.stem
            agg += [f]
    stats = pd.DataFrame(agg)
    print(stats.groupby("source").mean())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

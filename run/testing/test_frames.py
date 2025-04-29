import logging

import sys

import pathlib
import click
import suthing
from ethexco.util import parse_qa_pairs, Frame

logger = logging.getLogger(__name__)




@click.command()
@click.option("--frame-path", type=click.Path(path_type=pathlib.Path), required=True)
def main(frame_path):
    data = suthing.FileHandle.load(frame_path)
    s = data[0]["response_structured"][Frame.QUESTION_ANSWER_PAIRS]
    qas = parse_qa_pairs(s)
    print(len(qas))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

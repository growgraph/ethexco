import logging

import sys

import pathlib
import click
from dotenv import load_dotenv
import suthing

from ethexco.util import crawl_directories

logger = logging.getLogger(__name__)


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
        if tail is not None:
            fname_pre = f"{fname.stem}.tail.{tail}"
        else:
            fname_pre = fname.stem

        frames = suthing.FileHandle.load(output_path / (fname_pre + ".frame.json"))
        if tail is not None:
            data = data[-tail:]
        cnt = 0
        print(len(frames))
        print(sum([len(x["chunks"]) for x in data]))
        for doc in data:
            chunks = doc.pop("chunks", [])
            if tail is not None:
                chunks = chunks[-tail:]
            for chunk in chunks:
                item = frames[cnt]
                item["text"] = chunk
                cnt += 1
                print(cnt)

        suthing.FileHandle.dump(frames, output_path / (fname_pre + ".frame.json"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    main()

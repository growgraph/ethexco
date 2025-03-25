import numpy as np
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
import logging
import click
import sys
import pathlib
from dotenv import load_dotenv
from ethexco.util import crawl_directories
from llama_index.core import Document
import suthing


@click.command()
@click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
def run(env_path, input_path):
    _ = load_dotenv(dotenv_path=env_path.expanduser())

    input_path = input_path.expanduser()

    files = sorted(crawl_directories(input_path.expanduser()))
    for fname in files:
        data = suthing.FileHandle.load(fname)
        for item in data:
            if "chunks" in item:
                continue
            doc = Document(text=item["text"])
            documents = [doc]

            embed_model = OpenAIEmbedding()
            base_splitter = SentenceSplitter()
            base_nodes = base_splitter.get_nodes_from_documents(documents)

            splitter = SemanticSplitterNodeParser(
                buffer_size=5,
                breakpoint_percentile_threshold=90,
                embed_model=embed_model,
            )

            nodes = splitter.get_nodes_from_documents(documents)
            sizes = [len(x.get_content()) for x in base_nodes]
            print(fname)
            print(len(nodes), len(base_nodes))
            print(sizes)
            print(np.histogram(sizes))

            item.update({"chunks": [x.get_content() for x in base_nodes]})
        suthing.FileHandle.dump(data, fname)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()

from llama_index.core import SimpleDirectoryReader
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


@click.command()
# @click.option("--input-path", type=click.Path(path_type=pathlib.Path), required=True)
# @click.option("--output-path", type=click.Path(path_type=pathlib.Path), required=True)
@click.option("--env-path", type=click.Path(path_type=pathlib.Path), required=True)
def run(env_path):
    _ = load_dotenv(dotenv_path=env_path.expanduser())

    reader = SimpleDirectoryReader(
        input_dir="~/data/ethexco",
    )
    documents = reader.load_data()

    chunk_size = 1024
    embed_model = OpenAIEmbedding()
    base_splitter = SentenceSplitter(chunk_size=chunk_size)
    base_nodes = base_splitter.get_nodes_from_documents(documents)

    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(len(nodes), len(base_nodes))
    print(nodes[1].get_content())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()

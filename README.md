# Ethics Experts Council [EthExCo]

## Dev notes

1. `uv venv`
2. `source .venv/bin/activate`
3. `uv sync`

## Flow
1. Chunk raw data:
    ```shell
    python -m run.chunk_raw_data    
    ```

2. Generate Frames
    ```shell
    python -m run.generate_frames    
    ```
3. Generate Dataset
    ```shell
    python -m run.generate_dataset   
    ```

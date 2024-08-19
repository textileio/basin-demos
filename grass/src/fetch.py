"""
This file is a one-time dataset loader that fetches the Grass Reddit dataset
from HuggingFace, chunks the rows into smaller groups, and then writes them to
Basin as parquet files. Namely, it emulates Basin-native storage, but in
reality, the dataset would already exist on Basin.
"""

from os import path

from datasets import IterableDataset
from polars import DataFrame

from .util import flatten_row_lists


# Save groups of chunked rows to a parquet file
def save_chunk_to_parquet(chunk: list, index: int, output_dir: str) -> None:
    df = DataFrame(chunk)
    index_str = f"{index:03}"
    output_file = path.join(output_dir, f"chunk_{index_str}.parquet")
    df.write_parquet(output_file)


# Chunk the dataset into smaller groups and save them to parquet files
def chunk_dataset_to_parquet(
    dataset: IterableDataset, chunk_size: int, max_size: int, output_dir: str
) -> None:
    chunk = []
    current_chunk_size = 0
    total_data_processed = 0
    index = 0

    for _, row in enumerate(dataset):
        # Flatten the row to handle `media_urls` appropriately
        row = flatten_row_lists(row, "media_urls")

        # Simple approximation on row size for chunking
        # TODO: make this more accurate
        row_size = len(str(row))
        if total_data_processed + row_size > max_size:
            break

        # If adding this row exceeds the chunk size, save the current chunk
        if current_chunk_size + row_size > chunk_size:
            save_chunk_to_parquet(chunk, index, output_dir)
            chunk = []
            index += 1
            current_chunk_size = 0

        # Add the row to the current chunk
        chunk.append(row)
        current_chunk_size += row_size
        total_data_processed += row_size

    # Save the last chunk
    if chunk:
        save_chunk_to_parquet(chunk, index, output_dir)

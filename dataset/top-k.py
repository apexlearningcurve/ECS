import json
import math
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


def extract_embeddings(jsonl_file: Path, id_key: str = "id", chunk_size: int = 10000):
    """
    Extract embeddings from responses jsonl and chunk them
    """
    failed_ids = []

    def save_chunk(chunk_embeddings, chunk_ids, chunk_idx):
        df = pd.DataFrame({"embedding": chunk_embeddings, "item_id": chunk_ids})

        # Define the file path for the Parquet file
        parquet_file = f"dataset/embeddings/chunk_{chunk_idx}.parquet"

        # Save the DataFrame to a Parquet file
        df.to_parquet(parquet_file, index=False)

    chunk_idx = 0
    chunk_embeddings = []
    chunk_ids = []

    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            item_id = None  # Initialize id before the try block
            try:
                data = json.loads(line)
                item_id = data[2][id_key]
                embedding = data[1]["data"][0]["embedding"]
                chunk_embeddings.append(embedding)
                chunk_ids.append(item_id)

                if len(chunk_embeddings) >= chunk_size:
                    save_chunk(chunk_embeddings, chunk_ids, chunk_idx)
                    chunk_idx += 1
                    chunk_embeddings.clear()
                    chunk_ids.clear()

            except Exception as e:
                if item_id is not None:
                    failed_ids.append(item_id)
                logger.warning(
                    f"JSON loads failed for ID: {item_id}, with exception: {e}"
                )

        # Save any remaining data
        if chunk_embeddings:
            save_chunk(chunk_embeddings, chunk_ids, chunk_idx)

    return failed_ids


def load_chunk(index: int) -> pd.DataFrame:
    # Load the array from the .npy file
    return pd.read_parquet(f"dataset/embeddings/chunk_{index}.parquet")


def get_ids_by_indexes(indexes: list[int], df: pd.DataFrame) -> list[str]:
    return df.iloc[indexes]["item_id"].tolist()


def get_embeddings_by_indexes(indexes: ArrayLike) -> NDArray:
    results = []
    for index in indexes:
        chunk_index = math.floor(index / 100_000)
        relative_index = index - chunk_index * 100_000
        print(chunk_index, relative_index)
        chunk = pd.read_parquet(f"dataset/embeddings/chunk_{chunk_index}.parquet")
        embedding = chunk.iloc[relative_index]["embedding"]
        results.append(embedding)
    return np.stack(results)


def test_random_sample(num_samples: int, df_item_ids: pd.DataFrame) -> None:
    """
    Samples num_samples embeddings and creates a "sample_top_k.csv" with
    item_id -> top_k(item_ids) mappings
    """
    random_indices = np.random.choice(1_000_000, num_samples, replace=False)
    random_indices.sort()
    random_embeddings = get_embeddings_by_indexes(random_indices)
    random_ids = get_ids_by_indexes(random_indices.tolist(), df_item_ids)

    print("Searching...")

    print(random_embeddings.shape)
    distances, indices_list = index.search(random_embeddings, k)
    print(random_indices, distances, indices_list)
    top_ks = [get_ids_by_indexes(indices, df_item_ids) for indices in indices_list]
    df_item_ids = pd.DataFrame()
    df_item_ids["item_id"] = random_ids
    df_item_ids["top_k"] = top_ks
    df_item_ids.to_csv("sample_top_k.csv")


def batch_search_and_save(
    index: faiss.Index,
    df: pd.DataFrame,
    k: int = 101,
    batch_size: int = 1000,
    output_file: str = "nearest_neighbors.csv",
):
    """
    Performs a nearest neighbors search in batches, stores the results in a CSV file,
    and saves progress after each batch.

    Args:
    - index: The FAISS index or similar nearest neighbor index.
    - df: The DataFrame containing the embeddings and item_ids.
    - k: The number of nearest neighbors to search for each embedding.
    - batch_size: The number of embeddings to process in each batch.
    - output_file: The path to the CSV file where results will be stored.

    Returns:
    - None
    """
    num_batches = (
        len(df) + batch_size - 1
    ) // batch_size  # Calculate number of batches

    for i in tqdm(num_batches):
        # Determine the start and end of the current batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))

        # Extract the embeddings and item_ids for the current batch
        batch_embeddings = np.stack(df["embedding"].values[start_idx:end_idx])
        batch_item_ids = df["item_id"].iloc[start_idx:end_idx].tolist()

        # Perform the search for the current batch
        _, indices = index.search(batch_embeddings, k)

        # Collect the results as a list of dictionaries
        batch_results = []
        for j, item_id in enumerate(batch_item_ids):
            # Get the similar item_ids, excluding the first one (the queried vector itself)
            similar_item_ids = df["item_id"].iloc[indices[j][1:]].tolist()
            batch_results.append(
                {"item_id": item_id, "similar_item_ids": similar_item_ids}
            )

        # Convert the batch results to a DataFrame
        batch_df = pd.DataFrame(batch_results)

        # Append to the output CSV file
        if i == 0:
            batch_df.to_csv(
                output_file, index=False, mode="w"
            )  # Write header in the first batch
        else:
            batch_df.to_csv(
                output_file, index=False, mode="a", header=False
            )  # Append without header

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    """
    Embeddings responses from OpenAI are written to "embeddings_out.jsonl"
    This file is huge, we want to extract the embeddings and corresponding item_id s only
    we will chunk these and save them as parquet files in the folder embeddings
    """

    # jsonl_file = Path("embeddings_out.jsonl")
    # failed_ids = extract_embeddings(jsonl_file, id_key="item_id", chunk_size=100_000)
    # exit(1)

    """
    Once the embeddings are extracted and chunked, we'll use a faiss index to perform 
    similarity search for each vector, and save results to top_k.csv
    """
    # Initialize the FAISS index
    vector_size = 1536
    index = faiss.IndexFlatL2(vector_size)  # Use L2 distance (Euclidean)
    chunks = []
    print("Loading data and adding the embeddings to the index...")
    for i in tqdm(range(11)):
        chunk = load_chunk(i)
        chunks.append(chunk)
        index.add(np.stack(chunk["embedding"].values))
    df = pd.DataFrame()
    df = pd.concat(chunks)
    chunks.clear()
    del chunk

    k = 101
    print("Starting similarity search...")
    batch_search_and_save(
        index, df, k=101, batch_size=5000, output_file="nearest_neighbors.csv"
    )

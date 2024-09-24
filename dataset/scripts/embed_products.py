import os
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from utils import (
    OpenAIConfig,
    create_embedding_jobs,
    run_api_request_processor,
)

"""
We are embedding product Title + Description using OpenAi text-embedding-3-small
For getting top k similar products 
"""


def get_product_embeddings(
    df: pd.DataFrame, jobs_path: Path, out_path: Path, config: OpenAIConfig
):

    create_embedding_jobs(
        df,
        model="text-embedding-3-small",
        file_path=jobs_path,
        product_keys=["title", "description"],
        id_key="item_id",
    )
    scale_factor = 0.8
    run_api_request_processor(
        requests_filepath=jobs_path,
        save_filepath=out_path,
        request_url=config.url.embedding,
        max_requests_per_minute=int(
            scale_factor * config.limits.requests_per_minute.text_embedding_3_small
        ),
        max_tokens_per_minute=int(
            scale_factor * config.limits.tokens_per_minute.text_embedding_3_small
        ),
        token_encoding_name=config.token_encoding.text_embedding_3_small,
        max_attempts=config.max_attempts,
        logging_level=config.logging_level,
    )


def main():
    pass


if __name__ == "__main__":
    assert load_dotenv(find_dotenv()) and len(os.getenv("OPENAI_API_KEY")), ValueError(
        "env file not found"
    )

    # Load the config
    config_path = Path("dataset/config.yaml")
    config = OpenAIConfig.load_config_yaml(config_path.__str__())

    # Load the dataset
    dataset_path = Path(
        "dataset/c4-raw-meta-filtered_2024-Aug-20_20-44-50/sampled_item_metadata_1M_filtered.jsonl"
    )
    jobs_path = Path("requests.jsonl")
    out_path = Path("embeddings_out.jsonl")
    df = pd.read_json(dataset_path, lines=True)
    get_product_embeddings(df, jobs_path, out_path, config)

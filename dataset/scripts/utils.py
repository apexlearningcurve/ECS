import tiktoken
import asyncio
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
from scripts.api_request_parallel_processor import process_api_requests_from_file
from loguru import logger

# Log to a file with rotation
logger.add("data_processing.log", rotation="10 MB")


def extract_gz(path: Path) -> Path:
    """Extracts a .gz file to the same location without the .gz extension.

    Args:
        path (Path): Path to the .gz file to be extracted.

    Returns:
        Path: Path to the extracted file.

    Raises:
        ValueError: If the file extension is not .gz.
    """
    if path.suffix != ".gz":
        logger.info(f"Expected a .gz file, got {path.suffix} instead")
        raise ValueError(f"Expected a .gz file, got {path.suffix} instead")

    extracted_path = path.with_suffix("")  # Remove .gz suffix for the output file
    try:
        with gzip.open(path, "rb") as f_in, open(extracted_path, "wb") as f_out:
            f_out.write(f_in.read())
        # logger.info(f"File '{path}' has been unzipped to '{extracted_path}'")
    except Exception as e:
        logger.error(f"Failed to extract '{path}': {e}")
        raise RuntimeError(f"Extraction failed for {path}") from e

    return extracted_path


def create_category_permutations(full_category: str) -> List[str]:
    """Generate all hierarchical permutations of a category path."""
    parts = full_category.split(" > ")
    return [" > ".join(parts[: i + 1]) for i in range(len(parts))]


def load_and_process_data(file_path: Path, lines: int = None) -> List[dict]:
    """Load and process data from a given file path."""
    if file_path.suffix != ".jsonl":
        logger.info(f"Expected a .jsonl file, got {file_path.suffix} instead")
        raise ValueError(f"Expected a .jsonl file, got {file_path.suffix} instead")
    products = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                try:
                    products.append(json.loads(line.strip()))
                except Exception as e:
                    logger.warning(f"Exception occurred while loading data: {e}")

                if i == lines - 1:
                    break
            logger.info(
                f"Processed file {file_path} successfully, collected: {len(products)} products."
            )
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data=products)


def save_data(data: List[dict], output_file_path: Path) -> None:
    """Save the DataFrame to a JSON file."""
    try:
        with output_file_path.open(mode="w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
        # logger.info(f'Data saved to "{output_file_path}"')
    except Exception as e:
        logger.error(f'Failed to save data to "{output_file_path}": {e}')


def remove_processed_files(input_dir: Path, output_dir: Path) -> List[Path]:
    """
    This function removes processed files from the input directory and returns the names of unprocessed files as a list of file.

    Args:
        input_dir (Path): The path to the input directory.
        output_dir (Path): The path to the output directory.

    Returns:
        A list of file paths that are not found in the output directory.
    """
    if not input_dir.exists():
        logger.error(f"{input_dir} direcory dos not exist.")
        return []
    if not output_dir.exists():
        logger.warning(f"{output_dir} direcory dos not exist.")
        return list(input_dir.iterdir())

    input_file_names = set([file.with_suffix("").stem for file in input_dir.iterdir()])
    output_file_names = set(
        [
            file.with_suffix("").stem.removesuffix("_categories")
            for file in output_dir.iterdir()
        ]
    )

    difference = input_file_names.difference(output_file_names)
    return [input_dir / (file + ".jsonl.gz") for file in difference]


@dataclass
class URLConfig:
    embedding: str
    chat: str


@dataclass
class ModelLimit:
    gpt_4o: int
    gpt_4o_mini: int
    gpt_3_5_turbo: int
    text_embedding_3_large: int
    text_embedding_3_small: int
    text_embedding_ada_002: int


@dataclass
class LimitsConfig:
    requests_per_minute: ModelLimit
    tokens_per_minute: ModelLimit


@dataclass
class TokenEncodingConfig:
    gpt_4o: str
    gpt_4o_mini: str
    gpt_3_5_turbo: str
    text_embedding_3_large: str
    text_embedding_3_small: str
    text_embedding_ada_002: str


@dataclass
class OpenAIConfig:
    url: URLConfig
    max_attempts: int
    logging_level: int
    limits: LimitsConfig
    token_encoding: TokenEncodingConfig

    @staticmethod
    def load_config_yaml(yaml_file: Path) -> "OpenAIConfig":
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
            return OpenAIConfig(
                url=URLConfig(**data["openai"]["url"]),
                max_attempts=data["openai"]["max_attempts"],
                logging_level=data["openai"]["logging_level"],
                limits=LimitsConfig(**data["openai"]["limits"]),
                token_encoding=TokenEncodingConfig(
                    **data["openai"]["token_encoding_name"]
                ),
            )


def run_api_request_processor(
    requests_filepath: Path,
    save_filepath: Path,
    request_url: str,
    max_requests_per_minute: int,
    max_tokens_per_minute: int,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
) -> None:
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url=request_url,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            token_encoding_name=token_encoding_name,
            max_attempts=int(max_attempts),
            logging_level=int(logging_level),
        )
    )


def save_jsonl(entries: List[Dict], file_path: Path) -> None:
    with open(file_path, "w") as f:
        for entry in entries:
            json_string = json.dumps(entry)
            f.write(json_string + "\n")


def truncate_input(input: str):
    EMBEDDING_CTX_LENGTH = 8191
    EMBEDDING_ENCODING = "cl100k_base"
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    tokens = encoding.encode(input)
    if len(tokens) > EMBEDDING_CTX_LENGTH:
        return encoding.decode(
            tokens[:EMBEDDING_CTX_LENGTH]
        )  # not sure if i can pass tokens or text only
    return input


def create_embedding_jobs(
    df: pd.DataFrame,
    model: str,
    file_path: Path,
    product_keys: list[str] = ["product_text"],
    id_key: str = "id",
) -> None:

    assert file_path.suffix == ".jsonl", ValueError("File path must be a JSONL file!")

    jobs = [
        {
            "model": model,
            "input": truncate_input(
                "\n\n".join([getattr(row, product_key) for product_key in product_keys])
            ),
            "metadata": {id_key: getattr(row, id_key)},
        }
        for row in df.itertuples()
    ]
    save_jsonl(entries=jobs, file_path=file_path)


def load_results(
    results_path: Path, id_key: str = "id"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load results from a JSONL file and return a DataFrame and a List of faild IDs.
    """
    assert results_path.exists(), FileNotFoundError("There is no results file!")
    assert results_path.suffix == ".jsonl", ValueError(
        "File path must be a JSONL file!"
    )

    embeddings = []
    fail_ids = []
    with open(results_path, "r", encoding="utf-8") as file:
        for line in file:
            id = None  # Initialize id before the try block
            try:
                data = json.loads(line)
                id = data[2][id_key]
                embedding = data[1]["data"][0]["embedding"]
                embeddings.append({id_key: id, "embeddings": embedding})
            except Exception as e:
                if id is not None:
                    fail_ids.append(id)
                logger.warning(f"JSON loads failed for ID: {id}, with exception: {e}")
    df = pd.DataFrame(embeddings)
    return df, fail_ids

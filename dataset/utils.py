import gzip
import json
from pathlib import Path
from typing import List

import pandas as pd
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

from pathlib import Path

from datasets import load_dataset
from utils import OpenAIConfig, run_api_request_processor


def main():
    pass


if __name__ == "__main__":
    # Load the config
    config_path = Path("config.yaml")
    config = OpenAIConfig.load_config_yaml(config_path.__str__())

    # Load the dataset
    # dataset_id =
    # dataset = load_dataset()

import sys
from math import exp
from typing import Dict, Literal

import tiktoken
from loguru import logger
from openai import OpenAI

# Constants
MODEL_NAME = "gpt-4o-mini"
LABELS = ["Yes", "No"]


def product_relevance(
    client: OpenAI,
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    logprobs: bool = True,
    logit_bias: Dict[int, int] = None,
    max_tokens: int = 1,
):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=temperature,
        logprobs=logprobs,
        logit_bias=logit_bias,
        max_tokens=max_tokens,
    )

    return response


def get_tokenizer(model: str) -> tiktoken.Encoding:
    tokenizer = tiktoken.encoding_for_model(model)
    logger.info(f"Tokenizer for {model}: {tokenizer.name}")
    return tokenizer


def run_reranker(
    client: OpenAI,
    query: str,
    product_text: str,
    prompt: str,
    model: str = MODEL_NAME,
    labels: list = LABELS,
    logit_bias_value: float = 1.0,
    logger_level: Literal[
        "TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"
    ] = None,
) -> str:

    if logger_level:
        logger.remove()
        logger.add(sink=sys.stderr, level=logger_level)

    tokenizer = get_tokenizer(model)

    label_ids = [tokenizer.encode(label) for label in labels]
    logger.debug(f"Label IDs: {label_ids}")
    logit_bias = {id[0]: logit_bias_value for id in label_ids if len(id) == 1}
    logger.debug(f"Logit bias: {logit_bias}")

    responses = product_relevance(
        client,
        prompt=prompt.format(query=query, product_text=product_text),
        logit_bias=logit_bias,
    )

    label = responses.choices[0].message.content
    logprob = responses.choices[0].logprobs.content[0].logprob
    probability = exp(logprob)

    logger.success(f"Label: {label} with Probability: {probability}")
    return label, probability


if __name__ == "__main__":
    import os
    from openai import OpenAI
    from prompts import AMAZON_RANKING_PROMPT

    product_text = """
    YMIX Macbook Pro 13" Case Non-Retina,Folio Embroidered Shell Plastic Hard Protective Cover for Old MacBook Pro 13 Inch with CD-ROM Drive,Model A1278(A_Embroidered Floral)

    product category: Electronics
    """
    query = "case for apple laptop"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    label, probab = run_reranker(
        client=client,
        prompt=AMAZON_RANKING_PROMPT,
        query=query,
        product_text=product_text,
        logger_level="DEBUG",
    )

QUERY_GENERATION_PROMPT = """
### Task: E-commerce Product Query Generation ###

Generate realistic user search queries for a given e-commerce product. These queries will be used to train an embedding model for search, retrieval, and ranking.

### Input ###
Product Text: {product_text}

### Instructions ###
1. Analyze the product description thoroughly.
2. Generate 3 types of queries as if you were a real user searching for this product in this order:

   a. Short query (3-5 words):
      - Create a concise, focused search
      - Capture the essence of the product
      - You MUST follow word number restrictions: min 3 words, max 5 words

   b. Long query (5-8 words):
      - Create a long query that reflects natural user search behavior
      - Use the short query as a base and expand with relevant details
      - Focus on key features, common use cases, and the product's overall purpose
      - Include standout features or intended use that distinguish this product
      - Avoid overly specific measurements, technical details, or exact product specs unless they are essential to the product’s identity
      - Combine general product categories with key features and brand names
      - Reflect how an average user would naturally phrase their search
      - Ensure the query is more detailed than the short query but still user-friendly
      - Make the query specific enough to identify the product without being overly technical
      - Prioritize information that a typical shopper would consider important
      - Exclude details that a user wouldn't typically include in a search
      - You MUST follow word number restrictions: min 5 words, max 8 words

   c. Keywords (3-10 words):
      - List the most important and distinctive terms related to the product
      - Focus on specific features that set this product apart from similar items
      - Include brand name, product category, and key identifying characteristics
      - Prioritize unique materials, colors, sizes, or functionalities
      - Incorporate terms that users are likely to search for
      - Avoid generic descriptors or filler words
      - Ensure each keyword contributes significantly to identifying the product
      - Consider including one or two common use cases or target audiences
      - Exclude overly technical terms unless they're commonly used by shoppers
      - Aim for a mix of broad category terms and specific product features
      - You MUST follow word number restrictions: min 3 words, max 10 words

3. Ensure queries are:
   - Realistic and conversational
   - Free of excessive punctuation or symbols
   - Relevant to the product's main attributes

4. Avoid:
   - Using exact phrases from the product description
   - Query or parts of query can not be the same as if extracted from product text
   - Including uncommon abbreviations or technical jargon
   - Generating overly generic queries

### Output Format ###
Provide the result in JSON format:

{{
    "short_query": "your short query here",
    "long_query": "your generated long query here",
    "keywords": ["keyword1", "keyword2", "keyword3", ...]
}}
"""

EXAMPLE = """
### Example ###
For a "Red Leather Office Chair", a possible output could be:

{{
    "short_query": "red office chair",
    "long_query": "comfortable red leather chair for home office",
    "keywords": ["leather", "office", "chair", "red", "ergonomic", "adjustable"]
}}"""

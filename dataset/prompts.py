QUERY_GENERATION_PROMPT = """
# Task: E-commerce Product Query Generation
Generate realistic search queries for an e-commerce platform's like Amazon, Target, or Walmart. 
These queries should reflect how users search when looking to buy products, not general inquiries.

## Instructions
1. Analyze the product description briefly.
   a. Write a short reasoning about the product's purpose and features to guide query generation.

2. Generate two types of search queries:
   a. Short query (1-3 words): A concise search that captures the product's essence.
   Should contain the minimum, essential words needed to successfully search for this unique product.
   The query must consist of words that are sufficient to differentiate it from other similar products.
   Make sure to use the mose relevant keywords from the provided product text.
   b. Long query (4-6 words): A  longer search query expanding on the short query with relevant details like key features, 
   intended use, and product category. Avoid overly specific measurements or technical specs unless essential.
   The base should be the short query, expand it with only a few keywords that further differentiate this unique product. 
   Use no more than 6 words, do NOT include features that would exceed these limits.

## Queries should:
- Be realistic, product-centric, and shopper-focused.
- Exclude irrelevant details like "buy", "price", or "for sale."

## Output JSON Format
{{
   "reasoning": "short text of your reasoning here",
   "short_query": "your short query here",
   "long_query": "your generated query here"
}}

## Summary
- Short query: 1-3 words.
- Long query: 4-6 words, based on short query.
- Be realistic, avoid irrelevant purchase terms.
"""


BOOKS_PROMPT = """
When writing queries for books, focus on the book title, as most users search using the book title or part of it and author name.
"""

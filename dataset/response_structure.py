from pydantic import BaseModel


class ResponseStructure(BaseModel):
    reasoning: str
    short_query: str
    long_query: str

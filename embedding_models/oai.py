from typing import List, Literal, Iterator, Optional
from .base import EmbeddingBase
from openai import OpenAI
import tiktoken


class EmbeddingOpenai(EmbeddingBase):
    def __init__(self, keys: List[str]):
        super().__init__()
        self.name = "OpenAI Embedding Model"
        self._keys = keys
        self.key_index = 0
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, query: str) -> int:
        return len(self._encoding.encode(query))

    def embed(self, query: str, model: Literal["text-embedding-ada-002"] = "text-embedding-ada-002") -> List[float]:
        n = len(self._keys)

        query_token_nums = self._count_tokens(query)

        if query_token_nums >= 8192:
            raise ValueError(f"[Embed]-Openai: The total number of tokens in a query cannot exceed 8192, get {query_token_nums}.")

        data = {
            "model": model,
            "input": query,
            "encoding_format": "float"
        }

        errors = []

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.embeddings.create(**data)
                ans = completion.data[0].embedding
                return ans
            except Exception as e:
                errors.append(f"Key {self.key_index}: {e}")
                self.key_index = (self.key_index + 1) % n

        errors_info = "\n".join(errors)
        raise ValueError(f"[Embed]-Openai: All keys are invalid. The error message for the key is as follows: \n{errors_info}")

    def embed_batch(self, query_batch: List[str], model: Literal["text-embedding-ada-002"] = "text-embedding-ada-002") -> List[List[float]]:
        n = len(self._keys)

        query_token_nums = 0
        for query in query_batch:
            query_token_nums += self._count_tokens(query)

        if query_token_nums >= 8192:
            raise ValueError(f"[Embed Batch]-Openai: The total number of tokens in a batch cannot exceed 8192, get {query_token_nums}.")

        data = {
            "model": model,
            "input": query_batch,
            "encoding_format": "float"
        }

        errors = []

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.embeddings.create(**data)
                ans = [i.embedding for i in completion.data]
                return ans
            except Exception as e:
                errors.append(f"Key {self.key_index}: {e}")
                self.key_index = (self.key_index + 1) % n

        errors_info = "\n".join(errors)
        raise ValueError(f"[Embed Batch]-Openai: All keys are invalid. The error message for the key is as follows: \n{errors_info}")

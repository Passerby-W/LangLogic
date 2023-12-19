from abc import ABC, abstractmethod
from typing import List


class EmbeddingBase(ABC):
    def __init__(self):
        self.name = "Base Embedding Model"

    @abstractmethod
    def embed(self, query: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, query_batch: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def _count_tokens(self, query: str) -> int:
        pass

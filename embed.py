from typing import List
from key_management import KeyManager


class OpenAIEmbeddings:
    def __init__(self):
        self.km = KeyManager()

    def embed_query(self, text: str) -> List[float]:
        data = {
            "model": "text-embedding-ada-002",
            "input": text,
            "encoding_format": "float"
        }
        embed = self.km.embedding(**data).data[0].embedding
        return embed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = {
            "model": "text-embedding-ada-002",
            "input": texts,
            "encoding_format": "float"
        }
        embed = [data.embedding for data in self.km.embedding(**data).data]
        return embed


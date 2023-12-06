from typing import List
from manager import OpenAIManager


class OpenAIEmbedding:
    def __init__(self):
        self.km = OpenAIManager()

    def embed_query(self, text: str) -> List[float]:
        data = {
            "model": "text-embedding-ada-002",
            "input": text,
            "encoding_format": "float"
        }
        embed = self.km.embedding(**data).data[0].embedding
        return embed

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeds = []
        for text in texts:
            embeds.append(self.embed_query(text))
        return embeds


if __name__ == "__main__":
    e = OpenAIEmbedding()
    d1 = "我想吃火锅"
    d2 = ["我想吃火锅", "我想吃鱼"]
    print(e.embed_query(d1))
    print(e.embed_documents(d2))

from abc import ABC, abstractmethod
from typing import TypedDict, List, Literal, Iterator


class Massage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLM(ABC):
    def __init__(self):
        self.name = "base"

    @abstractmethod
    def chat(self, prompt: str, history: List[Massage], max_memory_token: int, **kwargs) -> str:
        pass

    @abstractmethod
    def streaming_chat(self, prompt: str, history: List[Massage], max_memory_token: int, **kwargs) -> Iterator[str]:
        pass

    @staticmethod
    def response_to_string(response: "Streaming Response") -> Iterator[str]:
        pass

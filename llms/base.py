from abc import ABC, abstractmethod
from typing import TypedDict, List, Tuple, Literal


class Massage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLM(ABC):
    def __init__(self):
        self.name = "base"

    @abstractmethod
    def chat(self, prompt: str, history: List[Massage], max_memory_token: int, **kwargs) -> Tuple[str, List[Massage]]:
        pass

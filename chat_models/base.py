from abc import ABC, abstractmethod
from typing import TypedDict, List, Literal, Iterator, Optional


class Massage(TypedDict):
    role: Literal["system", "user", "assistant"]
    name: Optional[str] = str
    content: str


class ChatBase(ABC):
    def __init__(self):
        self.name = "base llm"

    @abstractmethod
    def chat(self, prompt: str, history: List[Massage], max_memory_token: int, **kwargs) -> str:
        pass

    @abstractmethod
    def chat_stream(self, prompt: str, history: List[Massage], max_memory_token: int, **kwargs) -> Iterator[str]:
        pass

    @abstractmethod
    def _count_tokens(self, message: Massage) -> int:
        pass

    def _get_valid_messages(self, prompt: str, history: List[Massage], max_tokens: int) -> List[Massage]:
        counted_token_nums = 0
        valid_messages = []
        valid_messages_nums = 0

        prompt_message = {"role": "user", "content": prompt}
        prompt_token_nums = self._count_tokens(prompt_message)
        if counted_token_nums + prompt_token_nums > max_tokens:
            raise ValueError(f"Prompt is too long, the number of tokens in prompt should be less than or equal to {max_tokens}, get {prompt_token_nums}")
        else:
            counted_token_nums += prompt_token_nums
            if history[0]["role"] == "system":
                system_token_nums = self._count_tokens(history[0])
                if counted_token_nums + system_token_nums <= max_tokens:
                    valid_messages.append(history[0])
                    counted_token_nums += system_token_nums
                    history = history[1:]

            for message in reversed(history):
                message_token_nums = self._count_tokens(message)
                if counted_token_nums + message_token_nums <= max_tokens:
                    valid_messages_nums += 1
                    counted_token_nums += message_token_nums
                else:
                    break

            if valid_messages_nums > 0:
                valid_messages.extend(history[-valid_messages_nums:])

            valid_messages.append(prompt_message)

        return valid_messages

from abc import ABC, abstractmethod
from typing import List, Iterator, Optional
from .formats import ChatMassage


class ChatBase(ABC):
    def __init__(self):
        self.name = "Base Chat Model"

    @abstractmethod
    def chat(self, name: Optional[str], prompt: str, history: List[ChatMassage], max_memory_token: int, **kwargs) -> str:
        pass

    @abstractmethod
    def chat_stream(self, name: Optional[str], prompt: str, history: List[ChatMassage], max_memory_token: int, **kwargs) -> Iterator[str]:
        pass

    @abstractmethod
    def _count_tokens(self, message: ChatMassage) -> int:
        pass

    def _get_valid_messages(self, prompt: str, history: List[ChatMassage], max_memory_tokens: int, name: Optional[str] = None) -> List[ChatMassage]:
        counted_token_nums = 0
        valid_messages = []
        valid_messages_nums = 0

        if name:
            prompt_message = ChatMassage(role="user", name=name, content=prompt)
        else:
            prompt_message = ChatMassage(role="user", content=prompt)

        # count prompt message tokens
        prompt_token_nums = self._count_tokens(prompt_message)
        if counted_token_nums + prompt_token_nums > max_memory_tokens:
            raise ValueError(f"Prompt is too long, the number of tokens in prompt should be less than or equal to {max_memory_tokens}, get {prompt_token_nums}")
        else:
            counted_token_nums += prompt_token_nums
            # count system message tokens if it is in the history
            if history[0]["role"] == "system":
                system_token_nums = self._count_tokens(history[0])
                if counted_token_nums + system_token_nums <= max_memory_tokens:
                    valid_messages.append(history[0])
                    counted_token_nums += system_token_nums
                    history = history[1:]

            # count valid history message numbers
            for message in reversed(history):
                message_token_nums = self._count_tokens(message)
                if counted_token_nums + message_token_nums <= max_memory_tokens:
                    valid_messages_nums += 1
                    counted_token_nums += message_token_nums
                else:
                    break

            if valid_messages_nums > 0:
                valid_messages.extend(history[-valid_messages_nums:])

            valid_messages.append(prompt_message)

        return valid_messages

from typing import List, Literal, Iterator, Optional
from .base import ChatBase
from .formats import ChatMassage
from openai import OpenAI
import tiktoken


class ChatOpenai(ChatBase):
    def __init__(self, keys: List[str]):
        super().__init__()
        self.name = "OpenAI Chat Model"
        self._keys = keys
        self.key_index = 0
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, message: ChatMassage) -> int:
        if message.get("name"):
            name_tokens = 1
        else:
            name_tokens = 0
        return 3 + name_tokens + len(self._encoding.encode(message["content"]))

    def chat(self,
             prompt: str,
             history: List[ChatMassage],
             name: Optional[str] = None,
             max_memory_tokens: int = 4096,
             model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
             ) -> str:

        n = len(self._keys)

        messages = self._get_valid_messages(prompt=prompt, history=history, max_memory_tokens=max_memory_tokens)

        data = {
            "model": model,
            "messages": messages
        }

        errors = []

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.chat.completions.create(**data)
                ans = completion.choices[0].message.content
                return ans
            except Exception as e:
                errors.append(f"Key {self.key_index}: {e}")
                self.key_index = (self.key_index + 1) % n

        errors_info = "\n".join(errors)
        raise ValueError(f"[Chat]-OpenAI: All keys are invalid. The error message for the key is as follows: \n{errors_info}")

    def chat_stream(self,
                    prompt: str,
                    history: List[ChatMassage],
                    name: Optional[str] = None,
                    max_memory_tokens: int = 4096,
                    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
                    ) -> Iterator[str]:
        def response_to_string(openai_stream_completions) -> Iterator[str]:
            for openai_stream_completion in openai_stream_completions:
                character = openai_stream_completion.choices[0].delta.content
                stop = openai_stream_completion.choices[0].finish_reason
                if stop:
                    break
                yield character

        n = len(self._keys)

        messages = self._get_valid_messages(prompt=prompt, history=history, max_memory_tokens=max_memory_tokens)

        data = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        errors = []

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.chat.completions.create(**data)
                return response_to_string(completion)
            except Exception as e:
                errors.append(f"Key {self.key_index}: {e}")
                self.key_index = (self.key_index + 1) % n

        errors_info = "\n".join(errors)
        raise ValueError(f"[Chat Stream]-Openai: All keys are invalid. The error message for the key is as follows: \n{errors_info}")

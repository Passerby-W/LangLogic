from typing import List, Literal, Iterator, Optional
from .base import ChatBase
from .formats import ChatMassage
from openai import OpenAI
import tiktoken


class ChatOpenai(ChatBase):
    def __init__(self, keys: List[str]):
        super().__init__()
        self.name = "openai"
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

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.chat.completions.create(**data)
                ans = completion.choices[0].message.content
                return ans
            except (Exception,):
                self.key_index = (self.key_index + 1) % n

        raise ValueError(f"[Chat]-OAI_LLM: All keys are invalid or all keys not support {model}")

    def chat_stream(self,
                    prompt: str,
                    history: List[ChatMassage],
                    name: Optional[str] = None,
                    max_memory_tokens: int = 4096,
                    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
                    ) -> Iterator[str]:
        def response_to_string(openai_stream_completions) -> Iterator[str]:
            for openai_stream_completion in openai_stream_completions:
                yield openai_stream_completion.choices[0].delta.content

        n = len(self._keys)

        messages = self._get_valid_messages(prompt=prompt, history=history, max_memory_tokens=max_memory_tokens)

        data = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.chat.completions.create(**data)
                return response_to_string(completion)

            except (Exception,):
                self.key_index = (self.key_index + 1) % n

        raise ValueError(f"[Streaming Chat]-OAI_LLM: All keys are invalid or all keys not support {model}")

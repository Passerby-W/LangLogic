from typing import List, Literal, Iterator
from llms.base import LLM, Massage
from openai import OpenAI

from llms.utils import get_messages_by_length


class OaiLLM(LLM):
    def __init__(self, keys: List[str]):
        super().__init__()
        self.name = "openai"
        self._keys = keys
        self.key_index = 0

    def chat(self,
             prompt: str,
             history: List[Massage],
             max_memory_tokens: int = 4096,
             model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
             ) -> str:

        n = len(self._keys)

        messages = get_messages_by_length(history=history, max_memory_tokens=max_memory_tokens)

        messages.append({"role": "user", "content": prompt})

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

    @staticmethod
    def response_to_string(response):
        for chunk in response:
            yield chunk.choices[0].delta.content

    def streaming_chat(self,
                       prompt: str,
                       history: List[Massage],
                       max_memory_tokens: int = 4096,
                       model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
                       ) -> Iterator[str]:

        n = len(self._keys)

        messages = get_messages_by_length(history=history, max_memory_tokens=max_memory_tokens)

        messages.append({"role": "user", "content": prompt})
        history.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                response = client.chat.completions.create(**data)
                return self.response_to_string(response)

            except (Exception,):
                self.key_index = (self.key_index + 1) % n

        raise ValueError(f"[Streaming Chat]-OAI_LLM: All keys are invalid or all keys not support {model}")


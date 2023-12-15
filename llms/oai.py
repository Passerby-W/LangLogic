from typing import List, Literal, Iterator
from llms.base import LLM, Massage
from openai import OpenAI
import tiktoken


class OaiLLM(LLM):
    def __init__(self, keys: List[str]):
        super().__init__()
        self.name = "openai"
        self._keys = keys
        self.key_index = 0

    def chat(self,
             prompt: str,
             history: List[Massage],
             max_tokens: int = 4096,
             model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
             ) -> str:

        n = len(self._keys)

        messages = self.get_valid_messages(prompt=prompt, history=history, max_tokens=max_tokens)

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
                    history: List[Massage],
                    max_tokens: int = 4096,
                    model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
                    ) -> Iterator[str]:
        def response_to_string(completion):
            for i in completion:
                yield i.

        n = len(self._keys)

        messages = self.get_valid_messages(prompt=prompt, history=history, max_tokens=max_tokens)

        data = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.chat.completions.create(**data)
                return self.response_to_string(completion)

            except (Exception,):
                self.key_index = (self.key_index + 1) % n

        raise ValueError(f"[Streaming Chat]-OAI_LLM: All keys are invalid or all keys not support {model}")

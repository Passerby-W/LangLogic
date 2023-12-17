from typing import List, Literal, Iterator
from chat_models.base import ChatBase, Massage
from openai import OpenAI
import tiktoken


class ChatOpenai(ChatBase):
    def __init__(self, keys: List[str]):
        super().__init__()
        self.name = "openai"
        self._keys = keys
        self.key_index = 0
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, message: Massage) -> int:
        return 3 + len(self._encoding.encode(message["content"]))

    def chat(self,
             prompt: str,
             history: List[Massage],
             max_tokens: int = 4096,
             model: Literal["gpt-4-1106-preview", "gpt-3.5-turbo-1106"] = "gpt-4-1106-preview"
             ) -> str:

        n = len(self._keys)

        messages = self._get_valid_messages(prompt=prompt, history=history, max_tokens=max_tokens)

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
                yield i.choices[0].delta.content

        n = len(self._keys)

        messages = self._get_valid_messages(prompt=prompt, history=history, max_tokens=max_tokens)
        for i in messages:
            print(i)

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

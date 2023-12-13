from typing import List, Literal, Tuple
from llms.base import LLM, Massage
from openai import OpenAI


class OAI_LLM(LLM):
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
             ) -> Tuple[str, List[Massage]]:

        n = len(self._keys)

        tokens = 0
        messages = []
        valid_messages_nums = 0

        if history[0]["role"] == "system" and len(history[0]["content"]) <= max_memory_tokens:
            messages.append(history[0])
            tokens += len(history[0]["content"])

        for msg in reversed(history):
            tokens += len(msg["content"])
            if tokens <= max_memory_tokens:
                valid_messages_nums += 1
            else:
                break

        if valid_messages_nums > 0:
            messages.extend(history[-valid_messages_nums:])

        messages.append({"role": "user", "content": prompt})
        history.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages
        }

        for _ in range(n):
            client = OpenAI(api_key=self._keys[self.key_index])
            try:
                completion = client.chat.completions.create(**data)
                ans = completion.choices[0].message.content
                history.append({"role": "assistant", "content": ans})
                return ans, history
            except (Exception,):
                self.key_index = (self.key_index + 1) % n

        raise ValueError("[Chat]-OAI_LLM: All keys are invalid or not support the model")


if __name__ == "__main__":
    k = [
        "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",
        "sk-2qCKy7OJb6btmruEEzQWT3BlbkFJRBWhnUGzo4Rr4tWYlBzT1",
        "sk-gYdgB3lSTfuX68irkPw8T3BlbkFJZgznaIwPVnHgpMLcSI8v2",
        "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"
    ]
    llm = OAI_LLM(keys=k)
    print(llm.chat("你好", [{"role": "system", "content": "你是一个不礼貌的助手，你爱骂礼貌的人"}], 100))

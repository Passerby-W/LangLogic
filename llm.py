from manager import OpenAIManager


class OpenAILLM:
    def __init__(self):
        self.km = OpenAIManager()

    def chat(self, *args, **kwargs) -> str:
        response = self.km.chat(*args, **kwargs).choices[0].message.content
        return response


if __name__ == "__main__":
    e = OpenAILLM()
    d = {
        "model": "gpt-4-1106-preview",
        "messages": [
            {
                "role": "system",
                "content": "你是一个暴躁的助手。"
            },
            {
                "role": "user",
                "content": f"你好。"
            },
        ]
    }
    print(e.chat(**d))

from utils import get_config
from openai import OpenAI

api_keys = get_config("KEY_OPENAI")


class OpenAIManager:
    def __init__(self):
        self.current_key_index = 0
        self.api_keys = get_config("KEY_OPENAI")

    def chat(self, *args, **kwargs):
        if "model" not in kwargs:
            raise KeyError("No model specified in the parameters")
        if "messages" not in kwargs:
            raise KeyError("No messages specified in the parameters")

        n = len(self.api_keys)
        for _ in range(n):
            client = OpenAI(api_key=self.api_keys[self.current_key_index])

            try:
                completion = client.chat.completions.create(*args, **kwargs)
                return completion
            except (Exception,):
                self.current_key_index = (self.current_key_index + 1) % n
                pass
        raise ValueError("All keys are invalid or not support the model")

    def embedding(self, *args, **kwargs):
        if "model" not in kwargs:
            raise KeyError("No model specified in the parameters")
        if "input" not in kwargs:
            raise KeyError("No input specified in the parameters")
        n = len(self.api_keys)
        for _ in range(n):
            client = OpenAI(api_key=self.api_keys[self.current_key_index])

            try:
                completion = client.embeddings.create(*args, **kwargs)
                return completion
            except (Exception,):
                self.current_key_index = (self.current_key_index + 1) % n
                pass
        raise ValueError("All keys are invalid or not support the model")


if __name__ == "__main__":
    oai_manager = OpenAIManager()
    data = {
        "model": "gpt-4-1106-preview",
        "messages": [
            {
                "role": "system",
                "content": "你是一个仔细并且准确的总结答复助手，请根据用户给的背景材料来回答用户提问。"
            },
            {
                "role": "user",
                "content": f"根据我所找到的材料：中国成立于1942年，我想知道1+1的答案。如果材料中没有足够的信息回答这个问题，则不要胡编乱造答案。"
            },
        ]
    }
    print(oai_manager.chat(**data))
    data = {
        "model": "text-embedding-ada-002",
        "input": "我想吃火锅",
        "encoding_format": "float"
    }
    print(oai_manager.embedding(**data))

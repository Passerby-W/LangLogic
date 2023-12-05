from utils import get_config
from openai import OpenAI

api_keys = get_config("KEY_OPENAI")


class KeyManager:
    def __init__(self):
        self.current_key_index = 0
        self.api_keys = get_config("KEY_OPENAI")

    def chat(self, *args, **kwargs):
        n = len(self.api_keys)
        while n > 0:
            n -= 1
            client = OpenAI(api_key=self.api_keys[self.current_key_index])

            try:
                completion = client.chat.completions.create(*args, **kwargs)
                return completion
            except (Exception,):
                self.current_key_index += 1
                pass
        raise ValueError("All keys are invalid or not support the model")

    def embedding(self, *args, **kwargs):
        n = len(self.api_keys)
        while n > 0:
            n -= 1
            client = OpenAI(api_key=self.api_keys[self.current_key_index])

            try:
                completion = client.embeddings.create(*args, **kwargs)
                return completion
            except (Exception,):
                self.current_key_index += 1
                pass
        raise ValueError("All keys are invalid or not support the model")


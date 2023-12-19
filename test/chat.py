import time
from chat_models import ChatOpenai

keys = [
    "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",
]

llm = ChatOpenai(keys=keys)

messages = [{"role": "system", "content": "你是一个说只北京话的智能助手"}]

t1 = time.time()
a = llm.chat(prompt="你好", history=messages, max_memory_tokens=60)
t2 = time.time()
print(a)
print(t2 - t1)

b = llm.chat_stream(prompt="你好", history=messages, max_memory_tokens=60)
for bb in b:
    print(bb, end="")
print()
t3 = time.time()
print(t3 - t2)

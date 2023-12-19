import time

from chat_models import ChatOpenai

keys = ["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"]

llm = ChatOpenai(keys=keys)

messages = [{"role": "system", "content": "你是大诗人"}]

# for i in range(100):
#     messages.append({"role": "user", "content": f"{i}"})

t1 = time.time()
a = llm.chat(prompt="给我写一个赞美唐美女的诗歌，要求字数不低于60字，并解释下这首诗的意境和表达的感情", history=messages, max_memory_tokens=60)
t2 = time.time()
print(a)
print(t2 - t1)

b = llm.chat_stream(prompt="给我写一个赞美唐美女的诗歌，要求字数不低于60字，并解释下这首诗的意境和表达的感情", history=messages, max_memory_tokens=60)
for bb in b:
    print(bb, end="")
t3 = time.time()
print(a)
print(t3 - t2)

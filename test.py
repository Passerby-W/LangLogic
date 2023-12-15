import sys
import time

from llms import OaiLLM


keys = ["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"]

llm = OaiLLM(keys=keys)

messages = [{"role": "system", "content": "你是大诗人"}]

# for i in range(100):
#     messages.append({"role": "user", "content": f"{i}"})

a = llm.chat(prompt="给我写一个赞美唐美女的诗歌，要求字数不低于60字，并解释下这首诗的意境和表达的感情", history=messages, max_tokens=60)
# print(a)
for aa in a:
    print(aa, end="")
    # time.sleep(1)
    # sys.stdout.flush()

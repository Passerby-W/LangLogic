import datetime
from typing import Any

from langsmith.run_helpers import traceable
import os
from key_management import KeyManager

os.environ["OPENAI_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__cb7a98b3a8b441e391a0465c574b9d6a"
os.environ["LANGCHAIN_PROJECT"] = "test"

km = KeyManager()


@traceable(run_type="llm", name="openai.ChatCompletion.create")
def my_chat_model(*args: Any, **kwargs: Any):
    # return openai.chat.completions.create(*args, **kwargs)
    return km.chat(*args, **kwargs)


@traceable(run_type="tool")
def my_tool(tool_input: str) -> str:
    return tool_input.upper()


@traceable(run_type="chain")
def my_chain(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an AI Assistant. The time is "
                       + str(datetime.datetime.now()),
        },
        {"role": "user", "content": prompt},
    ]
    return my_chat_model(model="gpt-3.5-turbo", messages=messages)


@traceable(run_type="chain")
def my_chat_bot(text: str) -> str:
    generated = my_chain(text)

    if "meeting" in generated:
        return my_tool(generated)
    else:
        return generated


a = my_chat_bot("Summarize this morning's meetings.")
print(a)

from chat_models.base import Massage
from typing import List


def get_messages_by_length(history: List[Massage], max_memory_tokens: int) -> List[Massage]:
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

    return messages

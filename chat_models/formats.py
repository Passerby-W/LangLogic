from typing import Optional
from typing_extensions import Literal, Required, TypedDict


class ChatMassage(TypedDict, total=False):
    role: Required[Literal["system", "user", "assistant"]]
    name: Optional[str]
    content: Required[str]



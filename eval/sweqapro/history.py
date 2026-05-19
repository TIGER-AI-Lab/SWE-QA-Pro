from typing import List

from langchain_core.messages import AnyMessage


class ConversationHistory:
    """Sliding-window history of agent interactions.

    Each `interaction` is a list of messages (typically one assistant message plus
    any tool messages emitted in the same step). `flatten()` returns a flat list
    suitable for feeding back to the LLM.
    """

    def __init__(self, max_history: int):
        self.max_history = max_history
        self.history: List[List[AnyMessage]] = []

    def add_interaction(self, messages: List[AnyMessage]):
        self.history.append(messages)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def flatten(self) -> List[AnyMessage]:
        return [m for interaction in self.history for m in interaction]

    def clear(self):
        self.history.clear()

"""
Pipecat glue for the Wi-Fi account walk.

Pipeline position:   ... user_agg -> WifiDiagnosisProcessor -> retrieval -> llm -> tts ...

For every NEW user turn it asks the dialogue "is this yours?".
- Not ours (FAQ question, infrastructure complaint, chit-chat): frames pass
  through untouched and the existing FAQ path answers.
- Ours: the scripted reply is pushed as a TTSSpeakFrame (spoken verbatim, the
  LLM is never involved), the turn is "claimed" so the retrieval processor
  skips it, and the LLM trigger for that turn is swallowed.

The turn is detected on ANY downstream frame (not only the LLM trigger), because
the retrieval processor also reacts to the first frame after a user message is
appended; claiming early keeps it from logging walk turns as unanswered FAQs.
"""
from collections import deque

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TurnClaims:
    """Shared between this processor and the retrieval processor."""

    def __init__(self, maxlen: int = 32):
        self._items = deque(maxlen=maxlen)

    def claim(self, message) -> None:
        self._items.append(message)

    def is_claimed(self, message) -> bool:
        return any(m is message for m in self._items)   # identity, not equality


def message_text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class WifiDiagnosisProcessor(FrameProcessor):
    def __init__(self, context, dialogue, claims: TurnClaims, verbose: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._dialogue = dialogue
        self._claims = claims
        self._verbose = verbose
        self._seen_msg = None
        self._seen_text = None
        self._claimed_msg = None

    def _latest_user_message(self):
        messages = self._context.messages
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
            return messages[-1]
        return None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            msg = self._latest_user_message()
            if msg is not None:
                text = message_text(msg)
                is_new = not (msg is self._seen_msg and text == self._seen_text)
                if is_new:
                    self._seen_msg, self._seen_text = msg, text
                    self._claimed_msg = None
                    result = self._dialogue.handle(text)
                    if self._verbose:
                        print(f"[WifiDiagnosis] heard={text!r} handled={result.handled} "
                              f"state={self._dialogue.state.value} reply={result.text!r}")
                    if result.handled:
                        self._claims.claim(msg)
                        self._claimed_msg = msg
                        if result.text:
                            await self.push_frame(TTSSpeakFrame(result.text))

                # swallow the LLM trigger for a turn we answered ourselves
                if (type(frame).__name__ == "LLMContextFrame"
                        and self._claimed_msg is not None and msg is self._claimed_msg):
                    return

        await self.push_frame(frame, direction)

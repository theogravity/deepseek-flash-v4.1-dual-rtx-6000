"""LiteLLM callback: keep DeepSeek prompt prefixes stable by folding inline system messages into user turns.

Claude Code puts `role: "system"` entries inside `messages` (SessionStart/UserPromptSubmit hook context and a per-request `<total_tokens>` counter, the last
one tagged with cache_control). On vLLM's Anthropic endpoint (`/v1/messages`, which LiteLLM's `anthropic/` provider calls) those are merged into the leading
system section, so the start of the prompt changes on every request and only the part before the first change is reused. Measured on real Claude Code sessions
against this setup (docs/BENCHMARKS.md): 13-29% of prompt tokens hit the prefix cache per turn (every request reused exactly 6,144 tokens of a possible 20-26k);
with this callback 94.7-99.6% on turns 2-4, 84.7% for a brand-new session, 99.6% on resume.

Each inline system message becomes a `<system-reminder>` user text block (cache_control dropped) merged into the adjacent user turn so roles still alternate.
The rewrite depends only on the message itself and its neighbours, so a growing conversation keeps a byte-identical prefix across requests. Only models whose
name matches FOLD_MODEL_PATTERN (default: deepseek) are touched; the top-level `system` and everything else are left alone.

Install: put this file where LiteLLM can import it (for example /app/callbacks/fold_inline_system.py, mounted into the container) and add
    litellm_settings:
      callbacks: ["callbacks.fold_inline_system.fold_inline_system_handler"]
to the LiteLLM config. LiteLLM loads callbacks only at startup: recreate the container after changing this file.
"""

import logging
import re
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger(__name__)

import os

# Model names to fold for (matched against the model name the client sent, e.g. "DeepSeek V4.1 Flash"). Override with FOLD_MODEL_PATTERN.
FOLDED_MODELS = re.compile(os.environ.get("FOLD_MODEL_PATTERN", r"(?:^|/)deepseek(?=$|[\d\s._/-])"), re.IGNORECASE)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _blocks(content: Any) -> list:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content) if isinstance(content, list) else []


def fold_inline_system(model: Any, messages: Any) -> int:
    """Rewrite `messages` in place; return how many system messages were folded."""
    if not isinstance(model, str) or not FOLDED_MODELS.search(model):
        return 0
    if not isinstance(messages, list):
        return 0
    if not any(isinstance(m, dict) and m.get("role") == "system" for m in messages):
        return 0

    folded = 0
    out: list = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            folded += 1
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<system-reminder>\n{_text(message.get('content'))}\n</system-reminder>"}
                ],
            }
        if (
            out
            and isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(out[-1], dict)
            and out[-1].get("role") == "user"
        ):
            out[-1] = {**out[-1], "content": _blocks(out[-1].get("content")) + _blocks(message.get("content"))}
        else:
            out.append(message)
    messages[:] = out
    # Counts only: never log message content.
    logger.info("fold_inline_system: folded %d inline system message(s) for %s", folded, model)
    return folded


class FoldInlineSystemHandler(CustomLogger):
    async def async_pre_request_hook(
        self, model: str, messages: list, kwargs: dict
    ) -> Optional[dict]:
        # Same contract as queued_user: mutate the supplied list in place;
        # returning a dict would replace the request kwargs.
        fold_inline_system(model, messages)
        return None


fold_inline_system_handler = FoldInlineSystemHandler()

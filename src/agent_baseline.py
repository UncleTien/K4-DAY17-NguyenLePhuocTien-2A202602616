from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """
    Agent A / Baseline.

    Memory:
    - Remembers messages only inside the current thread.
    - Does not use persistent User.md.
    - Does not compact old messages.
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline

        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        """
        Return the agent response together with token accounting.
        """

        # Offline mode is deterministic and does not require API keys.
        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(thread_id, message)

        # For the lab, fall back to the deterministic implementation
        # if a live model cannot be invoked safely.
        try:
            session = self.sessions.setdefault(
                thread_id,
                SessionState(),
            )

            session.messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            prompt_text = "\n".join(
                item["content"]
                for item in session.messages
            )

            prompt_tokens = estimate_tokens(prompt_text)
            session.prompt_tokens_processed += prompt_tokens

            result = self.langchain_agent.invoke(
                session.messages
            )

            response = getattr(
                result,
                "content",
                str(result),
            )

            session.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            response_tokens = estimate_tokens(response)
            session.token_usage += response_tokens

            return {
                "response": response,
                "token_usage": session.token_usage,
                "prompt_tokens_processed": (
                    session.prompt_tokens_processed
                ),
            }

        except Exception:
            return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)

        if session is None:
            return 0

        return session.token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)

        if session is None:
            return 0

        return session.prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline intentionally has no compact memory.
        return 0

    def _reply_offline(
        self,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        """
        Deterministic baseline behavior.

        The full history of the current thread is retained.
        Nothing is shared between different thread IDs.
        """

        session = self.sessions.setdefault(
            thread_id,
            SessionState(),
        )

        session.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        # Baseline repeatedly processes the entire thread history.
        prompt_context = "\n".join(
            item["content"]
            for item in session.messages
        )

        prompt_tokens = estimate_tokens(prompt_context)
        session.prompt_tokens_processed += prompt_tokens

        response = self._offline_response(
            session.messages,
            message,
        )

        session.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens

        return {
            "response": response,
            "token_usage": session.token_usage,
            "prompt_tokens_processed": (
                session.prompt_tokens_processed
            ),
        }

    def _offline_response(
        self,
        messages: list[dict[str, str]],
        message: str,
    ) -> str:
        """
        Very small deterministic response generator.

        It may recall facts from the current thread only.
        """

        lower = message.lower()

        history = [
            item["content"]
            for item in messages[:-1]
            if item["role"] == "user"
        ]

        if (
            "tên gì" in lower
            or "tên của tôi" in lower
            or "my name" in lower
        ):
            for old_message in reversed(history):
                old_lower = old_message.lower()

                markers = [
                    "tên tôi là ",
                    "tôi tên là ",
                    "mình tên là ",
                    "my name is ",
                ]

                for marker in markers:
                    if marker in old_lower:
                        start = (
                            old_lower.index(marker)
                            + len(marker)
                        )

                        value = old_message[start:]
                        value = value.split(".")[0]
                        value = value.split(",")[0]
                        value = value.strip()

                        return f"Tên của bạn là {value}."

            return "Tôi không có thông tin đó trong thread hiện tại."

        return "Tôi đã nhận được thông tin trong thread hiện tại."

    def _maybe_build_langchain_agent(self):
        """
        Try to create the selected live chat model.

        Offline mode deliberately skips model initialization.
        """

        if self.force_offline:
            return None

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None
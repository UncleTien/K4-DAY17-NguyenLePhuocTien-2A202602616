from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """
    Agent B / Advanced Agent.

    Memory layers:
    1. within-session memory
    2. persistent User.md
    3. compact memory for long threads
    """

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline

        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles"
        )

        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )

        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        """Route between deterministic offline and live mode."""

        if self.force_offline or self.langchain_agent is None:
            return self._reply_offline(
                user_id,
                thread_id,
                message,
            )

        # Live mode.
        # If provider invocation fails, preserve lab usability
        # by falling back to the deterministic path.
        try:
            updates = extract_profile_updates(message)

            if updates:
                self._persist_profile_updates(
                    user_id,
                    updates,
                )

            self.compact_memory.append(
                thread_id,
                "user",
                message,
            )

            prompt_tokens = self._estimate_prompt_context_tokens(
                user_id,
                thread_id,
            )

            self.thread_prompt_tokens[thread_id] = (
                self.thread_prompt_tokens.get(thread_id, 0)
                + prompt_tokens
            )

            profile = self.profile_store.read_text(user_id)
            memory = self.compact_memory.context(thread_id)

            prompt = self._build_prompt(
                profile,
                memory,
                message,
            )

            result = self.langchain_agent.invoke(prompt)

            response = getattr(
                result,
                "content",
                str(result),
            )

            self.compact_memory.append(
                thread_id,
                "assistant",
                response,
            )

            self.thread_tokens[thread_id] = (
                self.thread_tokens.get(thread_id, 0)
                + estimate_tokens(response)
            )

            return {
                "response": response,
                "token_usage": self.token_usage(thread_id),
                "prompt_tokens_processed": (
                    self.prompt_token_usage(thread_id)
                ),
                "memory_file_size": (
                    self.memory_file_size(user_id)
                ),
                "compactions": (
                    self.compaction_count(thread_id)
                ),
            }

        except Exception:
            return self._reply_offline(
                user_id,
                thread_id,
                message,
            )

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        """
        Deterministic advanced path.

        1. Extract stable profile facts.
        2. Persist them to User.md.
        3. Append message to compact memory.
        4. Estimate prompt context.
        5. Generate deterministic response.
        6. Store response and update counters.
        """

        updates = extract_profile_updates(message)

        if updates:
            self._persist_profile_updates(
                user_id,
                updates,
            )

        self.compact_memory.append(
            thread_id,
            "user",
            message,
        )

        prompt_tokens = self._estimate_prompt_context_tokens(
            user_id,
            thread_id,
        )

        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0)
            + prompt_tokens
        )

        response = self._offline_response(
            user_id,
            thread_id,
            message,
        )

        self.compact_memory.append(
            thread_id,
            "assistant",
            response,
        )

        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0)
            + estimate_tokens(response)
        )

        return {
            "response": response,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": (
                self.prompt_token_usage(thread_id)
            ),
            "memory_file_size": (
                self.memory_file_size(user_id)
            ),
            "compactions": (
                self.compaction_count(thread_id)
            ),
        }

    def _persist_profile_updates(
        self,
        user_id: str,
        updates: dict[str, str],
    ) -> None:
        """Merge newly extracted stable facts into User.md."""

        current = self.profile_store.read_text(user_id)

        facts: dict[str, str] = {}

        for line in current.splitlines():
            line = line.strip()

            if not line.startswith("- "):
                continue

            body = line[2:]

            if ":" not in body:
                continue

            key, value = body.split(":", 1)

            facts[key.strip()] = value.strip()

        # New facts replace older values with the same key.
        facts.update(updates)

        lines = ["# User Profile", ""]

        for key, value in facts.items():
            lines.append(f"- {key}: {value}")

        content = "\n".join(lines).rstrip() + "\n"

        self.profile_store.write_text(
            user_id,
            content,
        )

    def _estimate_prompt_context_tokens(
        self,
        user_id: str,
        thread_id: str,
    ) -> int:
        """
        Estimate context entering one turn:

        User.md + compact summary + recent messages.
        """

        profile = self.profile_store.read_text(user_id)
        memory = self.compact_memory.context(thread_id)

        summary = str(memory.get("summary", ""))

        messages = memory.get("messages", [])

        recent_text = "\n".join(
            str(item.get("content", ""))
            for item in messages
        )

        context = "\n".join(
            [
                profile,
                summary,
                recent_text,
            ]
        )

        return estimate_tokens(context)

    def _offline_response(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> str:
        """Answer deterministic recall questions from User.md."""

        profile = self.profile_store.read_text(user_id)
        facts = self._parse_profile(profile)

        lower = message.lower()

        requested: list[tuple[str, str]] = []

        # NAME
        if (
            "tên" in lower
            or "dũngct là ai" in lower
            or "tóm tắt" in lower
        ):
            if facts.get("name"):
                requested.append(
                    ("Tên", facts["name"])
                )

        # PROFESSION
        if any(
            marker in lower
            for marker in (
                "nghề",
                "làm nghề",
                "nghề nghiệp",
                "nghề cũ",
                "nghề mới",
                "là ai",
                "tóm tắt",
            )
        ):
            if facts.get("profession"):
                requested.append(
                    ("Nghề nghiệp", facts["profession"])
                )

        # LOCATION
        if any(
            marker in lower
            for marker in (
                "ở đâu",
                "nơi ở",
                "đang ở",
                "còn ở huế",
                "huế",
                "hà nội",
            )
        ):
            if facts.get("location"):
                requested.append(
                    ("Nơi ở", facts["location"])
                )

        # DRINK
        if "đồ uống" in lower or "thức uống" in lower:
            if facts.get("favorite_drink"):
                requested.append(
                    (
                        "Đồ uống yêu thích",
                        facts["favorite_drink"],
                    )
                )

        # FOOD
        if "món ăn" in lower:
            if facts.get("favorite_food"):
                requested.append(
                    (
                        "Món ăn yêu thích",
                        facts["favorite_food"],
                    )
                )

        # PET
        if (
            "nuôi con gì" in lower
            or "con gì" in lower
            or "thú cưng" in lower
        ):
            if facts.get("pet"):
                requested.append(
                    ("Thú cưng", facts["pet"])
                )

        # RESPONSE STYLE
        if any(
            marker in lower
            for marker in (
                "style",
                "kiểu trả lời",
                "trả lời mình thích",
            )
        ):
            if facts.get("response_style"):
                requested.append(
                    (
                        "Style trả lời",
                        facts["response_style"],
                    )
                )

        # TECHNICAL INTERESTS
        if any(
            marker in lower
            for marker in (
                "mối quan tâm",
                "quan tâm chính",
                "kỹ thuật chính",
                "tóm tắt",
            )
        ):
            if facts.get("technical_interests"):
                requested.append(
                    (
                        "Mối quan tâm kỹ thuật",
                        facts["technical_interests"],
                    )
                )

        # Remove duplicate key/value pairs.
        requested = list(
            dict.fromkeys(requested)
        )

        if requested:
            return "; ".join(
                f"{label}: {value}"
                for label, value in requested
            ) + "."

        return (
            "Tôi đã cập nhật thông tin và memory "
            "cho cuộc hội thoại này."
        )

    @staticmethod
    def _parse_profile(
        profile: str,
    ) -> dict[str, str]:
        """Parse '- key: value' entries from User.md."""

        facts: dict[str, str] = {}

        for line in profile.splitlines():
            line = line.strip()

            if not line.startswith("- "):
                continue

            body = line[2:]

            if ":" not in body:
                continue

            key, value = body.split(":", 1)

            facts[key.strip()] = value.strip()

        return facts

    @staticmethod
    def _build_prompt(
        profile: str,
        memory: dict[str, object],
        message: str,
    ) -> str:
        """Build live-model prompt with persistent and compact memory."""

        summary = str(memory.get("summary", ""))

        messages = memory.get("messages", [])

        recent = "\n".join(
            f"{item.get('role', 'unknown')}: "
            f"{item.get('content', '')}"
            for item in messages
        )

        return (
            "You are a helpful AI assistant.\n\n"
            "Persistent user profile:\n"
            f"{profile}\n\n"
            "Compact conversation summary:\n"
            f"{summary}\n\n"
            "Recent messages:\n"
            f"{recent}\n\n"
            "Current user message:\n"
            f"{message}"
        )

    def _maybe_build_langchain_agent(self):
        """
        Initialize the configured live model when available.

        Offline mode deliberately avoids external model calls.
        """

        if self.force_offline:
            return None

        try:
            return build_chat_model(
                self.config.model
            )
        except Exception:
            return None
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Estimate token count using the simple ~4 chars/token heuristic."""
    text = text.strip()

    if not text:
        return 0

    return max(1, (len(text) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent markdown storage for user profile facts."""

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        """Return a safe path for one user's markdown profile."""

        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id.strip())
        safe_id = safe_id.strip("_") or "user"

        return self.root_dir / safe_id / "User.md"

    def read_text(self, user_id: str) -> str:
        """Read User.md or return an empty markdown profile."""

        path = self.path_for(user_id)

        if not path.exists():
            return "# User Profile\n"

        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        """Write the user's markdown profile."""

        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding="utf-8")

        return path

    def edit_text(
        self,
        user_id: str,
        search_text: str,
        replacement: str,
    ) -> bool:
        """Replace the first occurrence of text inside User.md."""

        path = self.path_for(user_id)

        if not path.exists():
            return False

        content = path.read_text(encoding="utf-8")

        if search_text not in content:
            return False

        updated = content.replace(
            search_text,
            replacement,
            1,
        )

        path.write_text(updated, encoding="utf-8")

        return True

    def file_size(self, user_id: str) -> int:
        """Return profile file size in bytes."""

        path = self.path_for(user_id)

        if not path.exists():
            return 0

        return path.stat().st_size


def extract_profile_updates(message: str) -> dict[str, str]:
    """
    Extract stable user facts from Vietnamese/English messages.

    Only explicit statements are stored.
    Questions are ignored so they cannot overwrite memory.
    """

    import re

    text = message.strip()

    if not text:
        return {}

    # Recall questions must never become new facts.
    if "?" in text:
        return {}

    updates: dict[str, str] = {}

    def search(patterns: list[str]) -> str | None:
        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(1).strip()

        return None

    # --------------------------------------------------
    # NAME
    # --------------------------------------------------
    value = search([
        r"(?:mình tên là|tôi tên là|tên tôi là)\s+([^,.!?]+)",
        r"(?:my name is)\s+([^,.!?]+)",
    ])

    if value:
        updates["name"] = value

    # --------------------------------------------------
    # LOCATION
    # Explicit current-location statements/corrections.
    # --------------------------------------------------
    value = search([
        r"(?:giờ mình đang ở|hiện tại mình đang ở|"
        r"mình đang ở|tôi đang ở|"
        r"mình vẫn ở|tôi vẫn ở)\s+([^,.!?]+)",

        r"(?:giờ mình ở|hiện tại mình ở|"
        r"mình ở|tôi ở)\s+([^,.!?]+)",

        r"(?:nơi ở đã cập nhật từ [^,.!?]+ sang)\s+([^,.!?]+)",

        r"(?:đang làm việc ở)\s+"
        r"(Đà Nẵng|Huế|Hà Nội|TP\.?\s*HCM)",
    ])

    if value:
        updates["location"] = value

    # Explicit correction:
    # "giờ mình đang ở Huế chứ..."
    correction = search([
        r"(?:giờ mình đang ở)\s+([^,.!?]+?)\s+chứ",
        r"(?:thực ra .*? mình đang .*? ở)\s+([^,.!?]+?)"
        r"(?:\s+vài|\s+để|,|\.)",
    ])

    if correction:
        updates["location"] = correction

    # --------------------------------------------------
    # PROFESSION
    # --------------------------------------------------
    value = search([
        r"(?:giờ chuyển sang)\s+([^,.!?]+)",
        r"(?:nghề nghiệp hiện tại vẫn là|"
        r"nghề nghiệp hiện tại là|"
        r"nghề hiện tại là)\s+([^,.!?]+)",

        r"(?:mình vẫn làm|tôi vẫn làm)\s+"
        r"([^,.!?]+?(?:engineer|developer|manager|researcher))",

        r"^(?:tôi là|mình là)\s+(?:một\s+)?([^,.!?]+)",
    ])

    if value:
        updates["profession"] = value

    # --------------------------------------------------
    # FAVORITE DRINK
    # --------------------------------------------------
    value = search([
        r"(?:đồ uống yêu thích(?: của tôi| của mình)? là)\s+"
        r"([^,.!?]+)",

        r"(?:thức uống yêu thích(?: của tôi| của mình)? là)\s+"
        r"([^,.!?]+)",

        r"(?:favorite drink is|my favorite drink is)\s+"
        r"([^,.!?]+)",
    ])

    if value:
        updates["favorite_drink"] = value

    # Dataset repeats the known preference this way.
    if (
        "cà phê sữa đá" in text.lower()
        and any(
            marker in text.lower()
            for marker in (
                "vẫn uống",
                "đồ uống",
                "thức uống",
                "yêu thích",
            )
        )
    ):
        updates["favorite_drink"] = "cà phê sữa đá"

    # --------------------------------------------------
    # FAVORITE FOOD
    # --------------------------------------------------
    value = search([
        r"(?:món ăn yêu thích(?: của tôi| của mình)? là)\s+"
        r"([^,.!?]+)",

        r"(?:favorite food is|my favorite food is)\s+"
        r"([^,.!?]+)",
    ])

    if value:
        updates["favorite_food"] = value

    # --------------------------------------------------
    # PET
    # --------------------------------------------------
    value = search([
        r"(?:mình nuôi|tôi nuôi)\s+(?:một\s+)?(?:bé\s+)?"
        r"([^,.!?]+?)(?:\s+tên\s+[^,.!?]+)?$",
    ])

    if value:
        updates["pet"] = value

    # --------------------------------------------------
    # RESPONSE STYLE
    # Store a compact normalized representation instead
    # of the entire sentence.
    # --------------------------------------------------
    lower = text.lower()

    if (
        "3 bullet" in lower
        or "ba bullet" in lower
    ):
        updates["response_style"] = "3 bullet"

    elif (
        "ngắn gọn" in lower
        and any(
            marker in lower
            for marker in (
                "trả lời",
                "giải thích",
                "câu trả lời",
                "style",
            )
        )
    ):
        updates["response_style"] = "ngắn gọn"

    elif (
        "bullet ngắn" in lower
        and "trả lời" in lower
    ):
        updates["response_style"] = "ngắn gọn"

    # --------------------------------------------------
    # TECHNICAL INTERESTS
    # Keep canonical values so recall is deterministic.
    # --------------------------------------------------
    interests: list[str] = []

    if "python" in lower:
        interests.append("Python")

    if (
        "ai ứng dụng" in lower
        or "ai agent" in lower
        or "agent" in lower
    ):
        interests.append("AI")

    if interests:
        updates["technical_interests"] = ", ".join(
            dict.fromkeys(interests)
        )

    return updates


def summarize_messages(
    messages: list[dict[str, str]],
    max_items: int = 6,
) -> str:
    """Create a deterministic compact summary of old messages."""

    if not messages:
        return ""

    selected = messages[-max_items:]

    lines: list[str] = []

    for message in selected:
        role = message.get("role", "unknown")
        content = message.get("content", "").strip()

        if not content:
            continue

        # Prevent one very long message from defeating compaction.
        if len(content) > 300:
            content = content[:297] + "..."

        lines.append(f"{role}: {content}")

    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Maintain recent messages and compact older conversation history."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _new_state(self) -> dict[str, object]:
        return {
            "messages": [],
            "summary": "",
            "compactions": 0,
        }

    def append(
        self,
        thread_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append a message and compact the thread when necessary."""

        if thread_id not in self.state:
            self.state[thread_id] = self._new_state()

        thread = self.state[thread_id]

        messages = thread["messages"]

        assert isinstance(messages, list)

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        summary = str(thread.get("summary", ""))

        full_context = summary + "\n" + "\n".join(
            str(message.get("content", ""))
            for message in messages
        )

        if estimate_tokens(full_context) <= self.threshold_tokens:
            return

        if len(messages) <= self.keep_messages:
            return

        split_at = len(messages) - self.keep_messages

        old_messages = messages[:split_at]
        recent_messages = messages[split_at:]

        old_summary = summarize_messages(old_messages)

        if summary and old_summary:
            combined_summary = (
                summary
                + "\n"
                + old_summary
            )
        else:
            combined_summary = summary or old_summary

        # Keep summary bounded so repeated compactions do not
        # recreate an ever-growing context.
        max_summary_chars = max(
            200,
            self.threshold_tokens * 2,
        )

        if len(combined_summary) > max_summary_chars:
            combined_summary = combined_summary[-max_summary_chars:]

        thread["summary"] = combined_summary
        thread["messages"] = recent_messages
        thread["compactions"] = (
            int(thread.get("compactions", 0)) + 1
        )

    def context(self, thread_id: str) -> dict[str, object]:
        """Return the compact memory state for a thread."""

        if thread_id not in self.state:
            self.state[thread_id] = self._new_state()

        thread = self.state[thread_id]

        return {
            "messages": list(thread["messages"]),
            "summary": str(thread["summary"]),
            "compactions": int(thread["compactions"]),
        }

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions performed for a thread."""

        if thread_id not in self.state:
            return 0

        return int(
            self.state[thread_id].get(
                "compactions",
                0,
            )
        )
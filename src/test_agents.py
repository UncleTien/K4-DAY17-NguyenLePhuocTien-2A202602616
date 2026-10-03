from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """
    Verify User.md can be created, read, updated, and edited.
    """

    store = UserProfileStore(
        tmp_path / "profiles"
    )

    user_id = "test-user"

    # File does not exist yet -> default profile.
    initial = store.read_text(user_id)

    assert "# User Profile" in initial

    # Write profile.
    content = (
        "# User Profile\n\n"
        "- name: Tiến\n"
        "- location: Đà Nẵng\n"
    )

    path = store.write_text(
        user_id,
        content,
    )

    assert path.exists()
    assert path.name == "User.md"

    saved = store.read_text(user_id)

    assert "Tiến" in saved
    assert "Đà Nẵng" in saved

    # Edit existing information.
    edited = store.edit_text(
        user_id,
        "Đà Nẵng",
        "Huế",
    )

    assert edited is True

    updated = store.read_text(user_id)

    assert "Huế" in updated
    assert "Đà Nẵng" not in updated

    assert store.file_size(user_id) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """
    Verify a sufficiently long thread triggers compaction.
    """

    memory = CompactMemoryManager(
        threshold_tokens=100,
        keep_messages=4,
    )

    thread_id = "long-thread"

    for i in range(30):
        memory.append(
            thread_id,
            "user",
            (
                f"Message {i}. "
                + (
                    "Đây là một đoạn hội thoại dài "
                    "dùng để kiểm tra compact memory. "
                    * 3
                )
            ),
        )

        memory.append(
            thread_id,
            "assistant",
            (
                "Tôi đã nhận được nội dung và "
                "tiếp tục cuộc hội thoại."
            ),
        )

    context = memory.context(thread_id)

    assert memory.compaction_count(thread_id) > 0

    assert context["compactions"] > 0

    # Recent context should be bounded after compaction.
    assert len(context["messages"]) <= 4

    # Older messages should have been summarized.
    assert context["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    """
    Advanced should remember persistent user facts
    across threads while Baseline should not.
    """

    config = load_config(tmp_path)

    baseline = BaselineAgent(
        config=config,
        force_offline=True,
    )

    advanced = AdvancedAgent(
        config=config,
        force_offline=True,
    )

    user_id = "cross-session-user"

    # --------------------------------------------------
    # Baseline
    # --------------------------------------------------

    baseline.reply(
        user_id,
        "baseline-thread-A",
        "Tên tôi là Tiến.",
    )

    baseline_result = baseline.reply(
        user_id,
        "baseline-thread-B",
        "Mình tên gì?",
    )

    assert "Tiến" not in baseline_result["response"]

    # --------------------------------------------------
    # Advanced
    # --------------------------------------------------

    advanced.reply(
        user_id,
        "advanced-thread-A",
        "Tên tôi là Tiến.",
    )

    advanced_result = advanced.reply(
        user_id,
        "advanced-thread-B",
        "Mình tên gì?",
    )

    assert "Tiến" in advanced_result["response"]

    # Persistent User.md must exist.
    profile = advanced.profile_store.read_text(
        user_id
    )

    assert "Tiến" in profile
    assert advanced.memory_file_size(user_id) > 0


def test_compact_reduces_prompt_load_on_long_thread(
    tmp_path: Path,
) -> None:
    """
    Compare cumulative prompt load on a long thread.

    Baseline keeps the entire thread.
    Advanced compacts old context.
    """

    config = load_config(tmp_path)

    # Force compaction early enough for a deterministic test.
    config.compact_threshold_tokens = 250
    config.compact_keep_messages = 4

    baseline = BaselineAgent(
        config=config,
        force_offline=True,
    )

    advanced = AdvancedAgent(
        config=config,
        force_offline=True,
    )

    user_id = "stress-user"

    baseline_thread = "baseline-long-thread"
    advanced_thread = "advanced-long-thread"

    for i in range(40):
        message = (
            f"Turn {i}. "
            "Tôi đang kiểm tra một hội thoại dài "
            "cho memory system của AI agent. "
            "Mục tiêu là so sánh việc giữ toàn bộ "
            "conversation history với compact memory. "
            "Thông tin này làm context dài dần theo "
            "thời gian để benchmark token usage."
        )

        baseline.reply(
            user_id,
            baseline_thread,
            message,
        )

        advanced.reply(
            user_id,
            advanced_thread,
            message,
        )

    baseline_prompt_tokens = (
        baseline.prompt_token_usage(
            baseline_thread
        )
    )

    advanced_prompt_tokens = (
        advanced.prompt_token_usage(
            advanced_thread
        )
    )

    assert (
        advanced.compaction_count(
            advanced_thread
        )
        > 0
    )

    assert baseline_prompt_tokens > 0
    assert advanced_prompt_tokens > 0

    # Main expected property of compact memory.
    assert (
        advanced_prompt_tokens
        < baseline_prompt_tokens
    )

def test_conflict_handling_prefers_latest_correction(
    tmp_path: Path,
) -> None:
    """
    Bonus: verify that a newer explicit correction replaces
    an older persistent fact instead of keeping stale data.
    """

    config = load_config(tmp_path)

    agent = AdvancedAgent(
        config=config,
        force_offline=True,
    )

    user_id = "conflict-user"

    # Initial persistent fact.
    agent.reply(
        user_id,
        "thread-01",
        "Mình đang ở Đà Nẵng.",
    )

    profile_before = agent.profile_store.read_text(
        user_id
    )

    assert "Đà Nẵng" in profile_before

    # Explicit correction in another session.
    agent.reply(
        user_id,
        "thread-02",
        "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng nữa.",
    )

    profile_after = agent.profile_store.read_text(
        user_id
    )

    # New value must replace the stale value.
    assert "Huế" in profile_after
    assert "- location: Đà Nẵng" not in profile_after

    # Recall from a completely new thread.
    result = agent.reply(
        user_id,
        "thread-03",
        "Hiện tại mình đang ở đâu?",
    )

    assert "Huế" in result["response"]
    assert "Đà Nẵng" not in result["response"]
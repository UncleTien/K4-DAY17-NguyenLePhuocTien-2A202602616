from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read benchmark conversations from JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "conversations",
            "cases",
            "data",
        ):
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(
        f"Unsupported benchmark format in {path}"
    )


def recall_score(
    response: str,
    expected_facts: list[str],
) -> float:
    """
    Return:
    0   = no expected facts recalled
    0.5 = some expected facts recalled
    1   = all expected facts recalled
    """

    if not expected_facts:
        return 1.0

    response_lower = response.lower()

    matches = sum(
        1
        for fact in expected_facts
        if str(fact).lower() in response_lower
    )

    if matches == 0:
        return 0.0

    if matches == len(expected_facts):
        return 1.0

    return 0.5


def offline_quality_score(response: str) -> float:
    """Lightweight deterministic quality score."""

    if not response:
        return 0.0

    lower = response.lower()

    failure_markers = (
        "tôi chưa có thông tin",
        "tôi không có thông tin",
        "i don't know",
        "i do not know",
    )

    if any(
        marker in lower
        for marker in failure_markers
    ):
        return 0.5

    if len(response.strip()) < 3:
        return 0.0

    return 1.0


def evaluate_agent(
    agent: Any,
    conversations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one agent over the real Day17 dataset."""

    recall_scores: list[float] = []
    quality_scores: list[float] = []

    user_ids: set[str] = set()
    thread_ids: set[str] = set()

    for conversation_index, conversation in enumerate(
        conversations
    ):
        user_id = str(
            conversation.get(
                "user_id",
                f"user-{conversation_index}",
            )
        )

        conversation_id = str(
            conversation.get(
                "id",
                f"conversation-{conversation_index}",
            )
        )

        # Thread containing the original conversation.
        source_thread = f"{conversation_id}-source"

        user_ids.add(user_id)
        thread_ids.add(source_thread)

        # --------------------------------------------------
        # 1. Feed all original conversation turns.
        # --------------------------------------------------
        turns = conversation.get("turns", [])

        for turn in turns:
            if not isinstance(turn, str):
                continue

            result = agent.reply(
                user_id,
                source_thread,
                turn,
            )

            response = str(
                result.get("response", "")
            )

            quality_scores.append(
                offline_quality_score(response)
            )

        # --------------------------------------------------
        # 2. Ask recall questions in NEW threads.
        #
        # This is important:
        # Baseline should forget.
        # Advanced should recall from User.md.
        # --------------------------------------------------
        recall_questions = conversation.get(
            "recall_questions",
            [],
        )

        for question_index, recall in enumerate(
            recall_questions
        ):
            question = str(
                recall.get("question", "")
            )

            expected = recall.get(
                "expected_contains",
                [],
            )

            if isinstance(expected, str):
                expected = [expected]

            recall_thread = (
                f"{conversation_id}"
                f"-recall-{question_index}"
            )

            thread_ids.add(recall_thread)

            result = agent.reply(
                user_id,
                recall_thread,
                question,
            )

            response = str(
                result.get("response", "")
            )

            recall_scores.append(
                recall_score(
                    response,
                    list(expected),
                )
            )

            quality_scores.append(
                offline_quality_score(response)
            )

    agent_tokens = sum(
        agent.token_usage(thread_id)
        for thread_id in thread_ids
    )

    prompt_tokens = sum(
        agent.prompt_token_usage(thread_id)
        for thread_id in thread_ids
    )

    compactions = sum(
        agent.compaction_count(thread_id)
        for thread_id in thread_ids
    )

    memory_growth = 0

    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(
            agent.memory_file_size(user_id)
            for user_id in user_ids
        )

    return {
        "agent_tokens": agent_tokens,
        "prompt_tokens": prompt_tokens,
        "cross_session_recall": (
            sum(recall_scores) / len(recall_scores)
            if recall_scores
            else 0.0
        ),
        "response_quality": (
            sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.0
        ),
        "memory_growth_bytes": memory_growth,
        "compactions": compactions,
    }


def print_results(
    title: str,
    baseline: dict[str, Any],
    advanced: dict[str, Any],
) -> None:
    """Print benchmark results as a markdown-style table."""

    print()
    print(f"## {title}")
    print()

    print(
        "| Metric | Baseline | Advanced |"
    )
    print(
        "|---|---:|---:|"
    )

    rows = [
        (
            "Agent tokens only",
            baseline["agent_tokens"],
            advanced["agent_tokens"],
        ),
        (
            "Prompt tokens processed",
            baseline["prompt_tokens"],
            advanced["prompt_tokens"],
        ),
        (
            "Cross-session recall",
            f'{baseline["cross_session_recall"]:.2f}',
            f'{advanced["cross_session_recall"]:.2f}',
        ),
        (
            "Response quality",
            f'{baseline["response_quality"]:.2f}',
            f'{advanced["response_quality"]:.2f}',
        ),
        (
            "Memory growth (bytes)",
            baseline["memory_growth_bytes"],
            advanced["memory_growth_bytes"],
        ),
        (
            "Compactions",
            baseline["compactions"],
            advanced["compactions"],
        ),
    ]

    for metric, base, adv in rows:
        print(
            f"| {metric} | {base} | {adv} |"
        )


def run_suite(
    title: str,
    dataset_path: Path,
    config,
) -> None:
    conversations = load_conversations(
        dataset_path
    )

    baseline = BaselineAgent(
        config=config,
        force_offline=True,
    )

    advanced = AdvancedAgent(
        config=config,
        force_offline=True,
    )

    baseline_result = evaluate_agent(
        baseline,
        conversations,
    )

    advanced_result = evaluate_agent(
        advanced,
        conversations,
    )

    print_results(
        title,
        baseline_result,
        advanced_result,
    )


def main() -> None:
    """Run Standard and Long-Context benchmark suites."""

    config = load_config()

    standard_path = (
        config.data_dir
        / "conversations.json"
    )

    long_context_path = (
        config.data_dir
        / "advanced_long_context.json"
    )

    run_suite(
        "Standard Benchmark",
        standard_path,
        config,
    )

    run_suite(
        "Long-Context Stress Benchmark",
        long_context_path,
        config,
    )


if __name__ == "__main__":
    main()
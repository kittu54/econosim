"""Agent memory system for LLM-powered economic agents.

Each agent maintains a rolling memory of observations, decisions,
and outcomes that feeds into their LLM prompts for contextual
decision-making.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry for an agent."""

    period: int
    category: str  # "observation", "decision", "outcome", "event"
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0  # 0-10 scale for memory pruning

    def to_text(self) -> str:
        """Format as natural language for prompt injection."""
        return f"[Period {self.period}, {self.category}] {self.content}"


class AgentMemory:
    """Rolling memory buffer for an LLM-powered economic agent.

    Maintains a bounded window of memories, pruning least important
    entries when the buffer is full. Supports filtering by category
    and period range.
    """

    def __init__(self, max_entries: int = 50, agent_id: str = "") -> None:
        self.max_entries = max_entries
        self.agent_id = agent_id
        self._entries: list[MemoryEntry] = []

    def add(
        self,
        period: int,
        category: str,
        content: str,
        data: dict[str, Any] | None = None,
        importance: float = 1.0,
    ) -> None:
        """Add a memory entry, pruning if at capacity."""
        entry = MemoryEntry(
            period=period,
            category=category,
            content=content,
            data=data or {},
            importance=importance,
        )
        self._entries.append(entry)

        if len(self._entries) > self.max_entries:
            self._prune()

    def _prune(self) -> None:
        """Remove lowest-importance entries to stay within budget."""
        # Always keep recent entries (last 10)
        if len(self._entries) <= self.max_entries:
            return

        recent = self._entries[-10:]
        older = self._entries[:-10]

        # Sort older by importance (ascending) and drop lowest
        older.sort(key=lambda e: e.importance)
        num_to_keep = self.max_entries - len(recent)
        kept = older[-num_to_keep:] if num_to_keep > 0 else []

        # Rebuild in chronological order
        self._entries = sorted(kept + recent, key=lambda e: e.period)

    def get_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Get the n most recent entries."""
        return self._entries[-n:]

    def get_by_category(self, category: str) -> list[MemoryEntry]:
        """Get all entries of a given category."""
        return [e for e in self._entries if e.category == category]

    def get_by_period_range(self, start: int, end: int) -> list[MemoryEntry]:
        """Get entries within a period range (inclusive)."""
        return [e for e in self._entries if start <= e.period <= end]

    def to_prompt_text(self, max_entries: int = 20) -> str:
        """Format recent memories as text for LLM prompt injection."""
        entries = self.get_recent(max_entries)
        if not entries:
            return "No prior memories."

        lines = [e.to_text() for e in entries]
        return "\n".join(lines)

    def summarize_decisions(self, last_n: int = 5) -> str:
        """Summarize recent decisions for context."""
        decisions = [e for e in self._entries if e.category == "decision"]
        recent = decisions[-last_n:]
        if not recent:
            return "No prior decisions recorded."
        return "\n".join(e.to_text() for e in recent)

    def clear(self) -> None:
        """Clear all memories."""
        self._entries.clear()

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def to_dict(self) -> dict[str, Any]:
        """Serialize memory state."""
        return {
            "agent_id": self.agent_id,
            "max_entries": self.max_entries,
            "entries": [
                {
                    "period": e.period,
                    "category": e.category,
                    "content": e.content,
                    "data": e.data,
                    "importance": e.importance,
                }
                for e in self._entries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMemory:
        """Deserialize memory state."""
        mem = cls(
            max_entries=data.get("max_entries", 50),
            agent_id=data.get("agent_id", ""),
        )
        for entry_data in data.get("entries", []):
            mem.add(
                period=entry_data["period"],
                category=entry_data["category"],
                content=entry_data["content"],
                data=entry_data.get("data", {}),
                importance=entry_data.get("importance", 1.0),
            )
        return mem

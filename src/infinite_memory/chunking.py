from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .config import ChunkingConfig

NON_LATIN_RE = re.compile(r"[\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7AF\uF900-\uFAFF]")


@dataclass(frozen=True)
class Chunk:
    index: int
    start_line: int
    end_line: int
    text: str


def read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def estimate_string_chars(text: str) -> int:
    """Mirror OpenClaw's rough token estimator: non-Latin chars count heavier."""
    if not text:
        return 0
    non_latin_count = len(NON_LATIN_RE.findall(text))
    return len(text) + non_latin_count * 3


def chunk_text(content: str, config: ChunkingConfig) -> list[Chunk]:
    lines = content.split("\n")
    if not lines:
        return []

    max_chars = max(32, config.tokens * 4)
    overlap_chars = max(0, config.overlap * 4)
    chunks: list[Chunk] = []
    current: list[tuple[str, int]] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = "\n".join(line for line, _line_no in current)
        if not text.strip():
            return
        chunks.append(
            Chunk(
                index=len(chunks),
                start_line=current[0][1],
                end_line=current[-1][1],
                text=text,
            )
        )

    def carry_overlap() -> None:
        nonlocal current, current_chars
        if overlap_chars <= 0 or not current:
            current = []
            current_chars = 0
            return
        acc = 0
        kept: list[tuple[str, int]] = []
        for entry in reversed(current):
            acc += estimate_string_chars(entry[0]) + 1
            kept.insert(0, entry)
            if acc >= overlap_chars:
                break
        current = kept
        current_chars = sum(estimate_string_chars(line) + 1 for line, _line_no in current)

    for index, line in enumerate(lines):
        line_no = index + 1
        if line == "":
            segments = [""]
        else:
            segments = []
            start = 0
            while start < len(line):
                coarse = line[start : start + max_chars]
                if estimate_string_chars(coarse) > max_chars:
                    fine_step = max(1, config.tokens)
                    for offset in range(0, len(coarse), fine_step):
                        segments.append(coarse[offset : offset + fine_step])
                else:
                    segments.append(coarse)
                start += max_chars

        for segment in segments:
            line_size = estimate_string_chars(segment) + 1
            if current and current_chars + line_size > max_chars:
                flush()
                carry_overlap()
            current.append((segment, line_no))
            current_chars += line_size

    flush()
    return chunks


def chunk_markdown(path: Path, config: ChunkingConfig) -> list[Chunk]:
    return chunk_text(read_markdown(path), config)

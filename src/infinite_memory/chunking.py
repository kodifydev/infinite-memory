from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ChunkingConfig


@dataclass(frozen=True)
class Chunk:
    index: int
    start_line: int
    end_line: int
    text: str


def read_markdown(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def chunk_lines(lines: list[str], config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not lines:
        return chunks

    step = config.max_lines - config.overlap_lines
    start = 0
    idx = 0
    while start < len(lines):
        end = min(len(lines), start + config.max_lines)
        text = "\n".join(lines[start:end]).strip()
        if len(text) >= config.min_chars:
            chunks.append(Chunk(index=idx, start_line=start + 1, end_line=end, text=text))
            idx += 1
        if end == len(lines):
            break
        start += step
    return chunks


def chunk_markdown(path: Path, config: ChunkingConfig) -> list[Chunk]:
    return chunk_lines(read_markdown(path), config)

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


FRONTMATTER_TITLE_RE = re.compile(r"^titulo:\s*(.+)$|^title:\s*(.+)$")
LIST_ITEM_RE = re.compile(r"^([-*+]|\d+[.)])\s")
MIN_BLOCK_CHARS = 16


def chunk_markdown_blocks(content: str, config: ChunkingConfig, fallback_title: str = "") -> list[Chunk]:
    """One chunk per paragraph or list item, prefixed with "<title> — <section>".

    A fact embedded alone keeps its own meaning instead of being averaged with the rest of the page;
    the prefix keeps who/what it is about. Blocks longer than the window size are split with
    ``chunk_text`` and every piece gets the same prefix."""
    lines = content.split("\n")
    body_start, title = 0, ""
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is not None:
            for line in lines[1:end]:
                m = FRONTMATTER_TITLE_RE.match(line.strip())
                if m and not title:
                    title = (m.group(1) or m.group(2)).strip().strip("\"'")
            body_start = end + 1

    blocks: list[tuple[int, int, str, str]] = []  # (start_line, end_line, section, text)
    section = ""
    para: list[tuple[str, int]] = []

    def flush() -> None:
        if para:
            text = "\n".join(line for line, _ in para).strip()
            if len(text) >= MIN_BLOCK_CHARS:
                blocks.append((para[0][1], para[-1][1], section, text))
            para.clear()

    for index in range(body_start, len(lines)):
        raw, line_no = lines[index], index + 1
        stripped = raw.strip()
        if stripped.startswith("#"):
            flush()
            heading = stripped.lstrip("#").strip()
            if stripped.startswith("# ") and not title:
                title = heading
            elif not stripped.startswith("# "):
                section = heading
            continue
        if not stripped:
            flush()
            continue
        if LIST_ITEM_RE.match(stripped):
            flush()
        elif para and raw[:1] not in (" ", "\t") and LIST_ITEM_RE.match(para[0][0].strip()):
            flush()  # a non-indented line after a list item starts a new paragraph
        para.append((raw, line_no))
    flush()

    title = title or fallback_title
    max_chars = max(32, config.tokens * 4)
    chunks: list[Chunk] = []
    for start, end, sec, text in blocks:
        prefix = " — ".join(part for part in (title, sec) if part)
        prefix = f"{prefix}\n" if prefix else ""
        if estimate_string_chars(text) <= max_chars:
            pieces = [Chunk(0, start, end, text)]
        else:
            pieces = [Chunk(0, start + piece.start_line - 1, start + piece.end_line - 1, piece.text)
                      for piece in chunk_text(text, config)]
        for piece in pieces:
            chunks.append(Chunk(len(chunks), piece.start_line, piece.end_line, prefix + piece.text))
    return chunks


def chunk_markdown(path: Path, config: ChunkingConfig) -> list[Chunk]:
    content = read_markdown(path)
    if config.strategy == "markdown_blocks":
        return chunk_markdown_blocks(content, config, fallback_title=path.stem)
    return chunk_text(content, config)

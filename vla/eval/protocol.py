"""Real-robot evaluation protocol: a fixed grid of start positions, named conditions, logged trials.

Modeled on the grid protocol from "Learning to Grasp Anything by Playing with Random Toys"
(ICLR 2026) and the robustness axes from CAIP (lighting changes, distractor objects). Every
trial is one JSON line, so results can be re-summarized at any time and are easy to publish.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONDITIONS = {
    "nominal": "Normal room lighting, only the task objects on the table.",
    "dim_light": "Overhead light off, one desk lamp at the side (same lamp position every time).",
    "bright_light": "All lights on plus the clamp LED aimed at the table from the front.",
    "distractors": "Add 3 unrelated objects (from the distractor box) at random positions off the grid.",
}


@dataclass(frozen=True)
class Grid:
    rows: int = 4
    cols: int = 4
    cell_cm: float = 6.0

    def cells(self) -> list[str]:
        return [f"{chr(ord('A') + row)}{col + 1}" for row in range(self.rows) for col in range(self.cols)]

    def center_cm(self, cell: str) -> tuple[float, float]:
        """(x, y) of a cell centre in cm, x across columns and y away from the robot."""
        row, col = ord(cell[0]) - ord("A"), int(cell[1:]) - 1
        return ((col + 0.5) * self.cell_cm, (row + 0.5) * self.cell_cm)


@dataclass(frozen=True)
class TrialSlot:
    condition: str
    cell: str
    repeat: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.condition, self.cell, self.repeat)


def trial_schedule(grid: Grid, conditions: list[str], repeats: int = 1, seed: int = 0) -> list[TrialSlot]:
    """Every (condition, cell, repeat) once, conditions in blocks, cells shuffled within each block."""
    rng = random.Random(seed)
    schedule = []
    for condition in conditions:
        block = [TrialSlot(condition, cell, repeat) for repeat in range(repeats) for cell in grid.cells()]
        rng.shuffle(block)
        schedule.extend(block)
    return schedule


@dataclass
class TrialRecord:
    label: str
    task: str
    condition: str
    cell: str
    repeat: int
    success: bool
    grasped: bool
    duration_s: float
    latency: dict = field(default_factory=dict)
    notes: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.condition, self.cell, self.repeat)


def append_record(path: str | Path, record: TrialRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def load_records(*paths: str | Path) -> list[TrialRecord]:
    records = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                records.append(TrialRecord(**json.loads(line)))
    return records


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval; stays sensible at 0/N and N/N, unlike the normal approximation."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


@dataclass(frozen=True)
class SummaryRow:
    label: str
    condition: str
    trials: int
    successes: int
    grasps: int

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.trials)


def summarize(records: list[TrialRecord]) -> list[SummaryRow]:
    groups: dict[tuple[str, str], list[TrialRecord]] = defaultdict(list)
    for record in records:
        groups[(record.label, record.condition)].append(record)
        groups[(record.label, "all")].append(record)
    return [
        SummaryRow(label, condition, len(group), sum(r.success for r in group), sum(r.grasped for r in group))
        for (label, condition), group in sorted(groups.items())
    ]


def to_markdown(rows: list[SummaryRow]) -> str:
    lines = ["| Policy | Condition | Success | 95% CI | Grasped | Trials |", "|---|---|---|---|---|---|"]
    for row in rows:
        low, high = row.interval
        lines.append(
            f"| {row.label} | {row.condition} | {row.rate:.0%} | {low:.0%}–{high:.0%} | "
            f"{row.grasps / row.trials:.0%} | {row.trials} |"
        )
    return "\n".join(lines)


def grid_sheet_svg(grid: Grid, margin_mm: float = 15.0) -> str:
    """A printable 1:1 grid (print at 100% scale) to tape down in front of the robot."""
    cell_mm = grid.cell_cm * 10
    width = grid.cols * cell_mm + 2 * margin_mm
    height = grid.rows * cell_mm + 2 * margin_mm + 12
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    for cell in grid.cells():
        x_cm, y_cm = grid.center_cm(cell)
        x = margin_mm + x_cm * 10 - cell_mm / 2
        y = margin_mm + (grid.rows * cell_mm) - (y_cm * 10 + cell_mm / 2)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_mm}" height="{cell_mm}" fill="none" stroke="black" stroke-width="0.4"/>'
        )
        parts.append(f'<circle cx="{x + cell_mm / 2}" cy="{y + cell_mm / 2}" r="1.2" fill="black"/>')
        parts.append(f'<text x="{x + 2}" y="{y + 6}" font-family="sans-serif" font-size="5">{cell}</text>')
    parts.append(
        f'<text x="{margin_mm}" y="{height - 6}" font-family="sans-serif" font-size="4">'
        f"Robot side is this edge. Cells {grid.cell_cm:g} cm. Print at 100% scale.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)

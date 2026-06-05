"""Deterministic ARC-AGI loading and exact-match evaluation primitives.

This module is intentionally separate from the Terminus living-loop runtime.
It provides benchmark plumbing only: grid validation, task loading, exact-match
scoring, object parsing, and explicitly limited scaffold baselines. None of
these utilities are ARC solvers or claims about core Terminus capabilities.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MIN_GRID_SIZE = 1
MAX_GRID_SIZE = 30
MIN_COLOR = 0
MAX_COLOR = 9

Grid = tuple[tuple[int, ...], ...]
GridLike = Sequence[Sequence[int]]
Cell = tuple[int, int]
BoundingBox = tuple[int, int, int, int]

ARC_OBJECT_DSL_BASELINE_DESCRIPTION = (
    "Deterministic object-DSL scaffold/baseline for ARC benchmark plumbing. "
    "It tries a tiny fixed transform set and is not an ARC solver."
)


@dataclass(frozen=True)
class ARCExample:
    """One ARC input/output pair.

    Test examples may omit ``output_grid`` when loaded from challenge-only
    data. Exact-match evaluation requires outputs to be present.
    """

    input_grid: Grid
    output_grid: Grid | None = None


@dataclass(frozen=True)
class ARCTask:
    """Loaded ARC task with immutable train/test examples."""

    train: tuple[ARCExample, ...]
    test: tuple[ARCExample, ...]
    task_id: str | None = None


@dataclass(frozen=True)
class ARCExactMatchEvaluation:
    """Exact-match evaluation summary for a batch of ARC predictions."""

    correct: int
    total: int
    accuracy: float
    per_example: tuple[bool, ...]


@dataclass(frozen=True)
class ARCNoIntelligenceBaseline:
    """Copies the input grid; this is a wiring baseline, not an ARC solver."""

    name: str = "copy_input_no_intelligence"
    description: str = (
        "Deterministically returns the input grid. This is a placeholder for "
        "benchmark plumbing and is not an ARC solver or intelligence claim."
    )

    def predict(self, task: ARCTask, example: ARCExample) -> Grid:
        del task
        return copy_grid(example.input_grid)


@dataclass(frozen=True)
class ARCObject:
    """Connected same-color component parsed from an ARC grid."""

    color: int
    cells: tuple[Cell, ...]
    bbox: BoundingBox
    size: int


@dataclass(frozen=True)
class ARCTransform:
    """A deterministic, intentionally tiny ARC grid transform scaffold."""

    name: str
    description: str
    _apply: Callable[[Grid], Grid] = field(repr=False, compare=False)

    def apply(self, grid: GridLike) -> Grid:
        """Apply the transform to a validated grid."""

        return validate_grid(self._apply(validate_grid(grid)), field_name=self.name)


@dataclass(frozen=True)
class ARCObjectDSLSearchResult:
    """Result from the limited object-DSL scaffold search, not a solver."""

    candidate_count: int
    selected_transform: ARCTransform | None
    train_matches: tuple[bool, ...]
    predictions: tuple[Grid, ...]
    test_evaluation: ARCExactMatchEvaluation | None
    description: str = ARC_OBJECT_DSL_BASELINE_DESCRIPTION


def validate_grid(grid: Any, *, field_name: str = "grid") -> Grid:
    """Validate and normalize an ARC grid to an immutable tuple-of-tuples.

    Valid ARC grids are rectangular, 1..30 cells in each dimension, and contain
    integer color IDs in the inclusive range 0..9.
    """

    if isinstance(grid, (str, bytes)) or not isinstance(grid, Sequence):
        raise ValueError(f"{field_name} must be a sequence of rows")
    if not (MIN_GRID_SIZE <= len(grid) <= MAX_GRID_SIZE):
        raise ValueError(f"{field_name} must have 1..30 rows")

    normalized: list[tuple[int, ...]] = []
    width: int | None = None

    for row_index, row in enumerate(grid):
        row_name = f"{field_name}[{row_index}]"
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"{row_name} must be a sequence of color IDs")
        if not (MIN_GRID_SIZE <= len(row) <= MAX_GRID_SIZE):
            raise ValueError(f"{row_name} must have 1..30 columns")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{field_name} must be rectangular")

        normalized_row: list[int] = []
        for col_index, value in enumerate(row):
            cell_name = f"{row_name}[{col_index}]"
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{cell_name} must be an integer color ID")
            if not (MIN_COLOR <= value <= MAX_COLOR):
                raise ValueError(f"{cell_name} must be in the range 0..9")
            normalized_row.append(value)
        normalized.append(tuple(normalized_row))

    return tuple(normalized)


def copy_grid(grid: GridLike) -> Grid:
    """Return a validated immutable copy of an ARC grid."""

    return validate_grid(grid)


def parse_arc_objects(
    grid: GridLike,
    *,
    include_background: bool = True,
    background_color: int = 0,
) -> tuple[ARCObject, ...]:
    """Parse 4-connected same-color components in deterministic row-major order."""

    normalized = validate_grid(grid)
    _validate_color(background_color, "background_color")

    rows = len(normalized)
    cols = len(normalized[0])
    visited: set[Cell] = set()
    objects: list[ARCObject] = []

    for row in range(rows):
        for col in range(cols):
            start = (row, col)
            if start in visited:
                continue
            color = normalized[row][col]
            if not include_background and color == background_color:
                visited.add(start)
                continue

            component = _collect_component(normalized, start, visited)
            min_row = min(cell[0] for cell in component)
            min_col = min(cell[1] for cell in component)
            max_row = max(cell[0] for cell in component)
            max_col = max(cell[1] for cell in component)
            objects.append(
                ARCObject(
                    color=color,
                    cells=component,
                    bbox=(min_row, min_col, max_row, max_col),
                    size=len(component),
                )
            )

    return tuple(objects)


def identity_transform() -> ARCTransform:
    """Return a copy-input transform scaffold."""

    return ARCTransform(
        name="identity",
        description="Return the input grid unchanged.",
        _apply=copy_grid,
    )


def recolor_transform(source_color: int, target_color: int) -> ARCTransform:
    """Return a transform that replaces one color ID everywhere."""

    _validate_color(source_color, "source_color")
    _validate_color(target_color, "target_color")

    def apply(grid: Grid) -> Grid:
        return tuple(
            tuple(target_color if value == source_color else value for value in row)
            for row in grid
        )

    return ARCTransform(
        name=f"recolor_{source_color}_to_{target_color}",
        description=f"Replace color {source_color} with color {target_color}.",
        _apply=apply,
    )


def crop_object_transform(
    object_index: int = 0,
    *,
    include_background: bool = False,
    background_color: int = 0,
) -> ARCTransform:
    """Return a transform that crops to one parsed object's bounding box."""

    if object_index < 0:
        raise ValueError("object_index must be non-negative")

    def apply(grid: Grid) -> Grid:
        objects = parse_arc_objects(
            grid,
            include_background=include_background,
            background_color=background_color,
        )
        if object_index >= len(objects):
            raise ValueError(f"object_index {object_index} is out of range")
        min_row, min_col, max_row, max_col = objects[object_index].bbox
        return tuple(
            tuple(grid[row][col] for col in range(min_col, max_col + 1))
            for row in range(min_row, max_row + 1)
        )

    return ARCTransform(
        name=f"crop_object_{object_index}_bbox",
        description=f"Crop to object {object_index}'s bounding box.",
        _apply=apply,
    )


def translate_transform(
    delta_row: int,
    delta_col: int,
    *,
    background_color: int = 0,
) -> ARCTransform:
    """Return a transform that translates non-background cells if in bounds."""

    _validate_color(background_color, "background_color")

    def apply(grid: Grid) -> Grid:
        rows = len(grid)
        cols = len(grid[0])
        translated = [[background_color for _ in range(cols)] for _ in range(rows)]

        for row, values in enumerate(grid):
            for col, value in enumerate(values):
                if value == background_color:
                    continue
                new_row = row + delta_row
                new_col = col + delta_col
                if not (0 <= new_row < rows and 0 <= new_col < cols):
                    raise ValueError("translation would move a cell out of bounds")
                translated[new_row][new_col] = value

        return validate_grid(translated)

    return ARCTransform(
        name=f"translate_{delta_row}_{delta_col}",
        description=(
            "Translate non-background cells by "
            f"({delta_row}, {delta_col}) when all destinations are in bounds."
        ),
        _apply=apply,
    )


def mirror_horizontal_transform() -> ARCTransform:
    """Return a left-right mirror transform."""

    return ARCTransform(
        name="mirror_horizontal",
        description="Reverse each grid row.",
        _apply=lambda grid: tuple(tuple(reversed(row)) for row in grid),
    )


def mirror_vertical_transform() -> ARCTransform:
    """Return a top-bottom mirror transform."""

    return ARCTransform(
        name="mirror_vertical",
        description="Reverse grid row order.",
        _apply=lambda grid: tuple(reversed(grid)),
    )


def rotate_cw_transform() -> ARCTransform:
    """Return a clockwise 90-degree rotation transform."""

    def apply(grid: Grid) -> Grid:
        return tuple(
            tuple(grid[row][col] for row in range(len(grid) - 1, -1, -1))
            for col in range(len(grid[0]))
        )

    return ARCTransform(
        name="rotate_cw",
        description="Rotate the grid 90 degrees clockwise.",
        _apply=apply,
    )


def generate_arc_object_dsl_candidates(task: ARCTask) -> tuple[ARCTransform, ...]:
    """Generate a small deterministic transform set for scaffold search."""

    input_colors = sorted(
        {value for example in task.train for row in example.input_grid for value in row}
    )
    output_colors = sorted(
        {
            value
            for example in task.train
            if example.output_grid is not None
            for row in example.output_grid
            for value in row
        }
    )
    max_objects = max(
        (
            len(parse_arc_objects(example.input_grid, include_background=False))
            for example in task.train
        ),
        default=0,
    )

    candidates: list[ARCTransform] = [
        identity_transform(),
        mirror_horizontal_transform(),
        mirror_vertical_transform(),
        rotate_cw_transform(),
    ]
    candidates.extend(
        recolor_transform(source_color, target_color)
        for source_color in input_colors
        for target_color in output_colors
        if source_color != target_color
    )
    candidates.extend(crop_object_transform(index) for index in range(max_objects))
    candidates.extend(
        translate_transform(delta_row, delta_col)
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )
    return tuple(candidates)


def search_arc_object_dsl_baseline(
    task: ARCTask,
    *,
    candidates: Sequence[ARCTransform] | None = None,
) -> ARCObjectDSLSearchResult:
    """Try a tiny object-DSL scaffold on train examples, then exact-score tests.

    This is a deterministic baseline for ARC plumbing experiments, not an ARC
    solver. It selects the first candidate that exactly matches every training
    output and only then applies that candidate to the test split.
    """

    candidate_list = tuple(candidates) if candidates is not None else (
        generate_arc_object_dsl_candidates(task)
    )

    for candidate in candidate_list:
        train_matches = _evaluate_candidate_on_train(task, candidate)
        if not train_matches or not all(train_matches):
            continue

        predictions: list[Grid] = []
        for example in task.test:
            prediction = _try_apply(candidate, example.input_grid)
            if prediction is None:
                return ARCObjectDSLSearchResult(
                    candidate_count=len(candidate_list),
                    selected_transform=candidate,
                    train_matches=train_matches,
                    predictions=tuple(predictions),
                    test_evaluation=None,
                )
            predictions.append(prediction)

        test_evaluation = None
        if all(example.output_grid is not None for example in task.test):
            test_evaluation = evaluate_exact_match(task, predictions)
        return ARCObjectDSLSearchResult(
            candidate_count=len(candidate_list),
            selected_transform=candidate,
            train_matches=train_matches,
            predictions=tuple(predictions),
            test_evaluation=test_evaluation,
        )

    return ARCObjectDSLSearchResult(
        candidate_count=len(candidate_list),
        selected_transform=None,
        train_matches=(),
        predictions=(),
        test_evaluation=None,
    )


def exact_match(predicted: GridLike, expected: GridLike) -> bool:
    """Return True only when two validated grids have identical shape/cells."""

    return validate_grid(predicted, field_name="predicted") == validate_grid(
        expected,
        field_name="expected",
    )


def exact_match_score(predicted: GridLike, expected: GridLike) -> float:
    """Return 1.0 for exact grid equality and 0.0 otherwise."""

    return 1.0 if exact_match(predicted, expected) else 0.0


def load_arc_task(data: Mapping[str, Any], *, task_id: str | None = None) -> ARCTask:
    """Load an ARC task from a JSON-like mapping.

    Expected shape matches ARC public data: ``{"train": [...], "test": [...]}``
    where each example has ``input`` and, for training/evaluation data,
    ``output`` grids.
    """

    if not isinstance(data, Mapping):
        raise ValueError("ARC task data must be a mapping")

    resolved_task_id = task_id
    raw_id = data.get("id")
    if resolved_task_id is None and isinstance(raw_id, str):
        resolved_task_id = raw_id

    train = _load_examples(data.get("train"), split_name="train", require_output=True)
    test = _load_examples(data.get("test"), split_name="test", require_output=False)
    return ARCTask(train=train, test=test, task_id=resolved_task_id)


def load_arc_task_file(path: str | Path, *, task_id: str | None = None) -> ARCTask:
    """Load an ARC task from a JSON file."""

    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return load_arc_task(data, task_id=task_id or task_path.stem)


def evaluate_exact_match(
    task: ARCTask,
    predictions: Sequence[GridLike],
) -> ARCExactMatchEvaluation:
    """Evaluate predictions against test outputs using strict exact match."""

    if len(predictions) != len(task.test):
        raise ValueError("prediction count must match task.test count")

    matches: list[bool] = []
    for index, (example, prediction) in enumerate(zip(task.test, predictions)):
        if example.output_grid is None:
            raise ValueError(f"test example {index} has no output grid")
        matches.append(exact_match(prediction, example.output_grid))

    correct = sum(1 for match in matches if match)
    total = len(matches)
    accuracy = correct / total if total else 0.0
    return ARCExactMatchEvaluation(
        correct=correct,
        total=total,
        accuracy=accuracy,
        per_example=tuple(matches),
    )


def evaluate_solver_exact_match(
    task: ARCTask,
    solver: Callable[[ARCTask, ARCExample], GridLike],
) -> ARCExactMatchEvaluation:
    """Run a deterministic solver callback and exact-match score its outputs."""

    predictions = [solver(task, example) for example in task.test]
    return evaluate_exact_match(task, predictions)


def _load_examples(
    raw_examples: Any,
    *,
    split_name: str,
    require_output: bool,
) -> tuple[ARCExample, ...]:
    if isinstance(raw_examples, (str, bytes)) or not isinstance(raw_examples, Sequence):
        raise ValueError(f"{split_name} must be a sequence of examples")
    if not raw_examples:
        raise ValueError(f"{split_name} must contain at least one example")

    examples: list[ARCExample] = []
    for index, raw_example in enumerate(raw_examples):
        if not isinstance(raw_example, Mapping):
            raise ValueError(f"{split_name}[{index}] must be a mapping")
        if "input" not in raw_example:
            raise ValueError(f"{split_name}[{index}] is missing input")
        if require_output and "output" not in raw_example:
            raise ValueError(f"{split_name}[{index}] is missing output")

        input_grid = validate_grid(
            raw_example["input"],
            field_name=f"{split_name}[{index}].input",
        )
        output_grid = None
        if "output" in raw_example:
            output_grid = validate_grid(
                raw_example["output"],
                field_name=f"{split_name}[{index}].output",
            )
        examples.append(ARCExample(input_grid=input_grid, output_grid=output_grid))

    return tuple(examples)


def _collect_component(
    grid: Grid,
    start: Cell,
    visited: set[Cell],
) -> tuple[Cell, ...]:
    target_color = grid[start[0]][start[1]]
    rows = len(grid)
    cols = len(grid[0])
    queue: deque[Cell] = deque([start])
    visited.add(start)
    cells: list[Cell] = []

    while queue:
        row, col = queue.popleft()
        cells.append((row, col))
        for next_row, next_col in (
            (row - 1, col),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col),
        ):
            next_cell = (next_row, next_col)
            if next_cell in visited:
                continue
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if grid[next_row][next_col] != target_color:
                continue
            visited.add(next_cell)
            queue.append(next_cell)

    return tuple(sorted(cells))


def _evaluate_candidate_on_train(
    task: ARCTask,
    candidate: ARCTransform,
) -> tuple[bool, ...]:
    matches: list[bool] = []
    for example in task.train:
        if example.output_grid is None:
            matches.append(False)
            continue
        prediction = _try_apply(candidate, example.input_grid)
        matches.append(
            False
            if prediction is None
            else exact_match(prediction, example.output_grid)
        )
    return tuple(matches)


def _try_apply(candidate: ARCTransform, grid: GridLike) -> Grid | None:
    try:
        return candidate.apply(grid)
    except ValueError:
        return None


def _validate_color(color: int, field_name: str) -> None:
    if isinstance(color, bool) or not isinstance(color, int):
        raise ValueError(f"{field_name} must be an integer color ID")
    if not (MIN_COLOR <= color <= MAX_COLOR):
        raise ValueError(f"{field_name} must be in the range 0..9")


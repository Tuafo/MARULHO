"""Tests for the separate ARC-AGI benchmark primitives."""

from __future__ import annotations

import unittest
from pathlib import Path

from marulho.evaluation.arc_agi import (
    ARC_OBJECT_DSL_BASELINE_DESCRIPTION,
    ARCNoIntelligenceBaseline,
    crop_object_transform,
    evaluate_exact_match,
    evaluate_solver_exact_match,
    exact_match,
    exact_match_score,
    generate_arc_object_dsl_candidates,
    identity_transform,
    load_arc_task,
    load_arc_task_file,
    mirror_horizontal_transform,
    parse_arc_objects,
    recolor_transform,
    rotate_cw_transform,
    search_arc_object_dsl_baseline,
    translate_transform,
    validate_grid,
)


SAMPLE_TASK = {
    "id": "sample_arc",
    "train": [
        {
            "input": [[1, 0], [0, 1]],
            "output": [[1, 1], [1, 1]],
        }
    ],
    "test": [
        {
            "input": [[2, 0], [0, 2]],
            "output": [[2, 2], [2, 2]],
        },
        {
            "input": [[3]],
            "output": [[3]],
        },
    ],
}


class ARCGridValidationTests(unittest.TestCase):
    def test_validate_grid_normalizes_to_tuple_grid(self) -> None:
        grid = validate_grid([[0, 1], [2, 3]])

        self.assertEqual(grid, ((0, 1), (2, 3)))

    def test_validate_grid_rejects_ragged_rows(self) -> None:
        with self.assertRaises(ValueError):
            validate_grid([[0, 1], [2]])

    def test_validate_grid_rejects_non_arc_color(self) -> None:
        with self.assertRaises(ValueError):
            validate_grid([[10]])

    def test_validate_grid_rejects_bool_cells(self) -> None:
        with self.assertRaises(ValueError):
            validate_grid([[True]])


class ARCLoaderTests(unittest.TestCase):
    def test_load_arc_task_from_dict(self) -> None:
        task = load_arc_task(SAMPLE_TASK)

        self.assertEqual(task.task_id, "sample_arc")
        self.assertEqual(len(task.train), 1)
        self.assertEqual(len(task.test), 2)
        self.assertEqual(task.train[0].input_grid, ((1, 0), (0, 1)))

    def test_load_arc_task_file_uses_filename_as_default_id(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "arc_agi_task.json"

        task = load_arc_task_file(fixture_path)

        self.assertEqual(task.task_id, "arc_agi_task")
        self.assertEqual(task.test[0].output_grid, ((8,),))

    def test_train_examples_require_output(self) -> None:
        bad_task = {"train": [{"input": [[1]]}], "test": [{"input": [[1]]}]}

        with self.assertRaises(ValueError):
            load_arc_task(bad_task)


class ARCExactMatchTests(unittest.TestCase):
    def test_exact_match_requires_identical_shape_and_values(self) -> None:
        self.assertTrue(exact_match([[1, 2]], [[1, 2]]))
        self.assertFalse(exact_match([[1, 2]], [[1], [2]]))
        self.assertFalse(exact_match([[1, 2]], [[1, 3]]))

    def test_exact_match_score_is_binary(self) -> None:
        self.assertEqual(exact_match_score([[4]], [[4]]), 1.0)
        self.assertEqual(exact_match_score([[4]], [[5]]), 0.0)

    def test_evaluate_exact_match_scores_all_test_outputs(self) -> None:
        task = load_arc_task(SAMPLE_TASK)

        result = evaluate_exact_match(task, [[[2, 2], [2, 2]], [[0]]])

        self.assertEqual(result.correct, 1)
        self.assertEqual(result.total, 2)
        self.assertEqual(result.accuracy, 0.5)
        self.assertEqual(result.per_example, (True, False))

    def test_evaluate_exact_match_requires_test_outputs(self) -> None:
        task = load_arc_task(
            {
                "train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[1]]}],
            }
        )

        with self.assertRaises(ValueError):
            evaluate_exact_match(task, [[[1]]])

    def test_no_intelligence_baseline_is_explicit_placeholder(self) -> None:
        task = load_arc_task(
            {
                "train": [{"input": [[1]], "output": [[1]]}],
                "test": [{"input": [[7]], "output": [[7]]}],
            }
        )
        baseline = ARCNoIntelligenceBaseline()

        result = evaluate_solver_exact_match(task, baseline.predict)

        self.assertIn("not an ARC solver", baseline.description)
        self.assertEqual(result.accuracy, 1.0)


class ARCObjectParsingTests(unittest.TestCase):
    def test_parse_arc_objects_returns_components_in_row_major_order(self) -> None:
        objects = parse_arc_objects(
            [[0, 1, 1, 0], [0, 1, 0, 2], [2, 0, 0, 2]],
            include_background=False,
        )

        self.assertEqual(len(objects), 3)
        self.assertEqual(objects[0].color, 1)
        self.assertEqual(objects[0].cells, ((0, 1), (0, 2), (1, 1)))
        self.assertEqual(objects[0].bbox, (0, 1, 1, 2))
        self.assertEqual(objects[0].size, 3)
        self.assertEqual(objects[1].color, 2)
        self.assertEqual(objects[1].bbox, (1, 3, 2, 3))
        self.assertEqual(objects[2].cells, ((2, 0),))

    def test_parse_arc_objects_can_include_background_components(self) -> None:
        objects = parse_arc_objects([[0, 1], [1, 0]])

        self.assertEqual([obj.color for obj in objects], [0, 1, 1, 0])
        self.assertEqual([obj.size for obj in objects], [1, 1, 1, 1])


class ARCObjectDSLTransformTests(unittest.TestCase):
    def test_basic_transforms_are_deterministic(self) -> None:
        self.assertEqual(identity_transform().apply([[1, 0]]), ((1, 0),))
        self.assertEqual(
            recolor_transform(1, 7).apply([[1, 0], [1, 2]]),
            ((7, 0), (7, 2)),
        )
        self.assertEqual(
            mirror_horizontal_transform().apply([[1, 0, 2], [3, 4, 5]]),
            ((2, 0, 1), (5, 4, 3)),
        )
        self.assertEqual(
            rotate_cw_transform().apply([[1, 2, 3], [4, 5, 6]]),
            ((4, 1), (5, 2), (6, 3)),
        )

    def test_crop_object_transform_uses_bounding_box(self) -> None:
        transform = crop_object_transform(0)

        self.assertEqual(
            transform.apply([[0, 1, 1], [0, 1, 0], [2, 0, 0]]),
            ((1, 1), (1, 0)),
        )

    def test_translate_transform_requires_in_bounds_destination(self) -> None:
        transform = translate_transform(1, 0)

        self.assertEqual(
            transform.apply([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
            ((0, 0, 0), (0, 1, 0), (0, 0, 0)),
        )
        with self.assertRaises(ValueError):
            transform.apply([[0, 0, 0], [0, 0, 0], [0, 1, 0]])


class ARCObjectDSLSearchTests(unittest.TestCase):
    def test_search_selects_first_training_match_and_exact_scores_tests(self) -> None:
        task = load_arc_task(
            {
                "train": [{"input": [[1, 0], [0, 0]], "output": [[2, 0], [0, 0]]}],
                "test": [{"input": [[1, 1]], "output": [[2, 2]]}],
            }
        )

        result = search_arc_object_dsl_baseline(task)

        self.assertIsNotNone(result.selected_transform)
        assert result.selected_transform is not None
        self.assertEqual(result.selected_transform.name, "recolor_1_to_2")
        self.assertEqual(result.train_matches, (True,))
        self.assertEqual(result.predictions, (((2, 2),),))
        self.assertIsNotNone(result.test_evaluation)
        assert result.test_evaluation is not None
        self.assertEqual(result.test_evaluation.accuracy, 1.0)
        self.assertIn("not an ARC solver", result.description)
        self.assertIn("scaffold/baseline", ARC_OBJECT_DSL_BASELINE_DESCRIPTION)

    def test_search_reports_no_transform_when_candidates_do_not_fit(self) -> None:
        task = load_arc_task(
            {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[1]], "output": [[2]]}],
            }
        )

        result = search_arc_object_dsl_baseline(
            task,
            candidates=[identity_transform()],
        )

        self.assertIsNone(result.selected_transform)
        self.assertEqual(result.predictions, ())
        self.assertEqual(result.test_evaluation, None)
        self.assertEqual(result.candidate_count, 1)

    def test_generated_candidates_include_limited_object_dsl_tools(self) -> None:
        task = load_arc_task(
            {
                "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
                "test": [{"input": [[1, 0]], "output": [[0, 1]]}],
            }
        )

        candidate_names = {
            candidate.name for candidate in generate_arc_object_dsl_candidates(task)
        }

        self.assertIn("identity", candidate_names)
        self.assertIn("mirror_horizontal", candidate_names)
        self.assertIn("rotate_cw", candidate_names)
        self.assertIn("crop_object_0_bbox", candidate_names)


if __name__ == "__main__":
    unittest.main()


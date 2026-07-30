'''Tests for evaluate_quiz.py against tests/fixtures/sample_quiz.csv.'''

import csv
import json
from pathlib import Path

import pytest

import evaluate_quiz
from course_config import load_course_config
from evaluate_quiz import (
    QuizItem,
    bloom_distribution,
    check_schema,
    compare_draft_final,
    coverage_report,
    emit_rating_sheet,
    find_near_duplicates,
    load_quiz,
    main,
    score_bloom_alignment,
    score_groundedness,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_quiz.csv"
MINI_COURSE_PATH = Path(__file__).parent / "fixtures" / "mini_course.yaml"


def test_load_quiz_returns_eight_items() -> None:
    items = load_quiz(FIXTURE_PATH)

    assert len(items) == 8
    assert all(isinstance(item, QuizItem) for item in items)
    assert items[0].question_id == "w01-q001"


def test_check_schema_flags_exactly_the_planted_defects() -> None:
    items = load_quiz(FIXTURE_PATH)
    report = check_schema(items)

    kinds_by_id = {(finding.question_id, finding.kind) for finding in report.findings}

    assert ("w03-q006", "empty_answer") in kinds_by_id
    assert ("w03-q005", "unknown_taxonomy") in kinds_by_id
    assert ("w01-q001", "duplicate_question_id") in kinds_by_id
    assert ("w04-q007", "answer_too_long") in kinds_by_id

    # No false positives: exactly these four findings, nothing more.
    assert len(report.findings) == 4
    assert not report.is_clean


def test_check_schema_clean_rows_produce_no_findings() -> None:
    items = load_quiz(FIXTURE_PATH)
    report = check_schema(items)
    flagged_ids = {finding.question_id for finding in report.findings}

    clean_ids = {"w02-q003", "w02-q004", "w05-q008"}
    assert clean_ids.isdisjoint(flagged_ids)


def test_find_near_duplicates_finds_planted_pair_at_085_not_at_099() -> None:
    items = load_quiz(FIXTURE_PATH)

    found_at_085 = find_near_duplicates(items, threshold=0.85)
    pair_ids = {frozenset((pair.question_id_a, pair.question_id_b)) for pair in found_at_085}
    assert frozenset({"w02-q003", "w02-q004"}) in pair_ids

    matching_pair = next(
        pair
        for pair in found_at_085
        if {pair.question_id_a, pair.question_id_b} == {"w02-q003", "w02-q004"}
    )
    assert matching_pair.scope == "within_week"
    assert 0.85 <= matching_pair.similarity < 0.99

    found_at_099 = find_near_duplicates(items, threshold=0.99)
    pair_ids_099 = {frozenset((pair.question_id_a, pair.question_id_b)) for pair in found_at_099}
    assert frozenset({"w02-q003", "w02-q004"}) not in pair_ids_099


def test_bloom_distribution_includes_zero_counts_and_unknown_bucket() -> None:
    items = load_quiz(FIXTURE_PATH)
    report = bloom_distribution(items)

    # All six canonical levels present, including any with zero items.
    for level in ("Knowledge", "Understanding", "Applying", "Analyzing", "Evaluating", "Creating"):
        assert level in report.counts
        assert level in report.proportions

    # The planted "Remembering" row lands in the unknown bucket.
    assert report.counts["unknown"] == 1
    assert report.proportions["unknown"] == pytest.approx(1 / 8)

    assert 0.0 <= report.entropy <= 1.0


def test_emit_rating_sheet_header_and_blind_omits_taxonomy(tmp_path: Path) -> None:
    items = load_quiz(FIXTURE_PATH)
    out_path = tmp_path / "rating_sheet.csv"

    emit_rating_sheet(items, out_path, blind=True)

    with out_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first_data_row = next(reader)

    assert header == [
        "question_id",
        "question",
        "answer",
        "rated_bloom_level",
        "factually_correct",
        "answerable_from_reading",
        "would_use",
        "comments",
    ]
    assert "taxonomy" not in [column.lower() for column in header]
    assert "knowledge" not in [cell.lower() for cell in first_data_row]
    # question_id, question, answer populated; last five columns empty.
    assert first_data_row[0] == items[0].question_id
    assert first_data_row[3:] == ["", "", "", "", ""]


def test_emit_rating_sheet_non_blind_includes_requested_taxonomy(tmp_path: Path) -> None:
    items = load_quiz(FIXTURE_PATH)
    out_path = tmp_path / "rating_sheet_non_blind.csv"

    emit_rating_sheet(items, out_path, blind=False)

    with out_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    assert "requested_taxonomy" in header


def test_load_quiz_tolerates_legacy_schema(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy_quiz.csv"
    with legacy_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["learning_objective", "reference", "taxonomy", "promtp", "question", "answer"])
        writer.writerow(
            [
                "Understand X",
                "Some Ref 2020",
                "Knowledge",
                "full legacy prompt text here",
                "What is X?",
                "X is a thing.",
            ]
        )
        writer.writerow(
            [
                "Understand Y",
                "Some Ref 2021",
                "Applying",
                "full legacy prompt text here too",
                "How is Y applied?",
                "Y is applied by doing Z.",
            ]
        )

    with pytest.warns(UserWarning, match="legacy"):
        items = load_quiz(legacy_path)

    assert len(items) == 2
    assert items[0].question_id == "legacy-0001"
    assert items[1].question_id == "legacy-0002"
    assert items[0].week == 0
    assert items[0].facts == ""
    assert items[0].question == "What is X?"


def test_compare_draft_final_skips_cleanly_when_columns_absent() -> None:
    items = load_quiz(FIXTURE_PATH)
    comparisons = compare_draft_final(items)
    assert comparisons == []


def test_compare_draft_final_reports_similarity_and_word_deltas() -> None:
    item = QuizItem(
        question_id="w01-q999",
        week=1,
        taxonomy="Knowledge",
        learning_objective="Understand X",
        reference="Ref 2020",
        source_files="x.pdf",
        facts="X is a thing.",
        question="What exactly is X in this context?",
        answer="X is a thing that does more than one thing.",
        model="gpt-4o",
        generated_at="2026-07-30T00:00:00Z",
        draft_question="What is X?",
        draft_answer="X is a thing.",
        refined=True,
    )

    comparisons = compare_draft_final([item])

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.question_id == "w01-q999"
    assert 0.0 <= comparison.question_similarity <= 1.0
    assert comparison.question_word_delta == len(item.question.split()) - len(item.draft_question.split())
    assert comparison.answer_word_delta == len(item.answer.split()) - len(item.draft_answer.split())


def test_coverage_report_with_no_items_flags_everything_uncovered() -> None:
    config = load_course_config(MINI_COURSE_PATH)

    report = coverage_report([], config, 2)

    assert report.week_number == 2
    assert report.covered_concept_count == 0
    assert report.total_concept_count == 2
    assert set(report.uncovered_concepts) == {"Advanced Concept A", "Advanced Concept B"}
    assert report.covered_reading_count == 0
    assert report.total_reading_count == 2


def test_coverage_report_reading_match_does_not_collide_on_shared_year() -> None:
    config = load_course_config(MINI_COURSE_PATH)
    # Matches the "Smith" reading by surname; must NOT also match "Johnson"
    # just because both citations happen to be published in 2023.
    item = QuizItem(
        question_id="w01-q001",
        week=1,
        taxonomy="Knowledge",
        learning_objective="Foundational aspects unique to Smith method",
        reference="Smith et al. 2023",
        source_files="",
        facts="fact",
        question="q?",
        answer="a.",
        model="m",
        generated_at="t",
    )

    report = coverage_report([item], config, 1)

    assert report.covered_reading_count == 1
    assert report.uncovered_readings == (
        "Johnson, M., et al. (2023). Another key reading on topic B.",
    )


def test_coverage_report_reading_match_via_source_file() -> None:
    config = load_course_config(MINI_COURSE_PATH)
    # reference text is unrelated; only the source_file should match the
    # Brown/example.pdf mapping reading in week 2.
    item = QuizItem(
        question_id="w02-q001",
        week=2,
        taxonomy="Knowledge",
        learning_objective="Something about Advanced Concept A",
        reference="totally unrelated text",
        source_files="example.pdf",
        facts="fact",
        question="q?",
        answer="a.",
        model="m",
        generated_at="t",
    )

    report = coverage_report([item], config, 2)

    assert report.covered_reading_count == 1
    assert report.uncovered_readings == (
        "Garcia, L. (2023). Standalone reference without supplementary file.",
    )


def test_coverage_report_counts_are_internally_consistent() -> None:
    config = load_course_config(MINI_COURSE_PATH)
    items = load_quiz(FIXTURE_PATH)  # unrelated data; just checking invariants

    report = coverage_report(items, config, 1)

    assert report.covered_concept_count + len(report.uncovered_concepts) == report.total_concept_count
    assert report.covered_reading_count + len(report.uncovered_readings) == report.total_reading_count


def test_score_groundedness_raises_not_implemented() -> None:
    items = load_quiz(FIXTURE_PATH)
    with pytest.raises(NotImplementedError):
        score_groundedness(items, query_engine=None, judge_llm=None)


def test_score_bloom_alignment_raises_not_implemented() -> None:
    items = load_quiz(FIXTURE_PATH)
    with pytest.raises(NotImplementedError):
        score_bloom_alignment(items, judge_llm=None)


def test_main_returns_zero_by_default() -> None:
    exit_code = main(["--quiz-csv", str(FIXTURE_PATH)])
    assert exit_code == 0


def test_main_returns_one_with_strict() -> None:
    exit_code = main(["--quiz-csv", str(FIXTURE_PATH), "--strict"])
    assert exit_code == 1


def test_main_returns_two_with_check_groundedness() -> None:
    exit_code = main(["--quiz-csv", str(FIXTURE_PATH), "--check-groundedness"])
    assert exit_code == 2


def test_main_returns_two_with_check_bloom_alignment() -> None:
    exit_code = main(["--quiz-csv", str(FIXTURE_PATH), "--check-bloom-alignment"])
    assert exit_code == 2


def test_main_writes_report_json(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    exit_code = main(["--quiz-csv", str(FIXTURE_PATH), "--report-json", str(report_path)])

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["item_count"] == 8
    assert len(payload["schema_findings"]) == 4


def test_main_emit_rating_sheet_flag(tmp_path: Path) -> None:
    sheet_path = tmp_path / "sheet.csv"
    exit_code = main(["--quiz-csv", str(FIXTURE_PATH), "--emit-rating-sheet", str(sheet_path)])

    assert exit_code == 0
    assert sheet_path.exists()
    with sheet_path.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header[0] == "question_id"

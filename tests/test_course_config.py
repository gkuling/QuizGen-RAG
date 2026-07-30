'''Tests for course_config.py: happy paths and every documented error branch.'''

import json
from pathlib import Path

import pytest

from course_config import (
    CourseConfigError,
    UnknownWeekError,
    load_course_config,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_COURSE_PATH = FIXTURES_DIR / "mini_course.yaml"
AIM2_COURSE_PATH = Path(__file__).parent.parent / "configs" / "aim2_course.yaml"


def test_mini_course_loads_with_string_and_mapping_readings() -> None:
    config = load_course_config(MINI_COURSE_PATH)

    assert config.course_id == "MINI"
    assert config.course_title == "Mini Test Course"
    assert config.week_numbers == (1, 2)

    week1 = config.get_week(1)
    assert week1.title == "Introductory Topics"
    assert week1.concepts == ("Concept A", "Concept B", "Concept C")
    assert len(week1.required_readings) == 2
    # Plain string readings normalize to a Reading with source_file None.
    assert week1.required_readings[0].citation == "Smith, J. et al. (2023). A foundational paper on topic A."
    assert week1.required_readings[0].source_file is None

    week2 = config.get_week(2)
    assert len(week2.required_readings) == 2
    # Mapping reading captures both citation and source_file.
    mapping_reading = week2.required_readings[0]
    assert mapping_reading.citation == "Brown, K., et al. (2024). Advanced research with supplementary material."
    assert mapping_reading.source_file == "example.pdf"
    # Second reading in week 2 is a plain string again.
    string_reading = week2.required_readings[1]
    assert string_reading.citation == "Garcia, L. (2023). Standalone reference without supplementary file."
    assert string_reading.source_file is None


def test_aim2_course_loads_with_13_weeks_and_week7_title() -> None:
    config = load_course_config(AIM2_COURSE_PATH)

    assert config.course_id == "AIM2"
    assert config.course_title == "Introduction to AI in Medicine"
    assert len(config.weeks) == 13
    assert config.week_numbers == tuple(range(1, 14))
    assert config.get_week(7).title == "Trustworthy AI"


def test_get_week_unknown_raises_with_valid_weeks_listed() -> None:
    config = load_course_config(MINI_COURSE_PATH)

    with pytest.raises(UnknownWeekError) as excinfo:
        config.get_week(99)

    message = str(excinfo.value)
    assert "99" in message
    assert "1" in message
    assert "2" in message


def test_json_branch_loads_equivalent_config(tmp_path: Path) -> None:
    json_path = tmp_path / "mini_course.json"
    payload = {
        "course": {"id": "MINIJSON", "title": "Mini JSON Course"},
        "weeks": [
            {
                "week": 1,
                "title": "Only Week",
                "concepts": ["Concept X"],
                "required_readings": ["A plain string reading."],
            }
        ],
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    config = load_course_config(json_path)

    assert config.course_id == "MINIJSON"
    assert config.week_numbers == (1,)
    assert config.get_week(1).title == "Only Week"
    assert config.get_week(1).required_readings[0].citation == "A plain string reading."


def test_missing_file_raises_course_config_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.yaml"

    with pytest.raises(CourseConfigError, match="not found"):
        load_course_config(missing_path)


def test_bad_suffix_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.txt"
    bad_path.write_text("weeks: []", encoding="utf-8")

    with pytest.raises(CourseConfigError, match="Unsupported"):
        load_course_config(bad_path)


def test_non_mapping_top_level_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(CourseConfigError, match="mapping"):
        load_course_config(bad_path)


def test_missing_weeks_key_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text("course:\n  id: X\n", encoding="utf-8")

    with pytest.raises(CourseConfigError, match="weeks"):
        load_course_config(bad_path)


def test_week_missing_title_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text(
        "weeks:\n"
        "  - week: 1\n"
        "    concepts: [\"A\"]\n"
        "    required_readings: [\"Some reading.\"]\n",
        encoding="utf-8",
    )

    with pytest.raises(CourseConfigError, match="title"):
        load_course_config(bad_path)


def test_non_int_week_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text(
        "weeks:\n"
        "  - week: \"one\"\n"
        "    title: \"Week One\"\n"
        "    concepts: [\"A\"]\n"
        "    required_readings: [\"Some reading.\"]\n",
        encoding="utf-8",
    )

    with pytest.raises(CourseConfigError, match="int"):
        load_course_config(bad_path)


def test_duplicate_week_numbers_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text(
        "weeks:\n"
        "  - week: 1\n"
        "    title: \"Week One\"\n"
        "    concepts: [\"A\"]\n"
        "    required_readings: [\"Some reading.\"]\n"
        "  - week: 1\n"
        "    title: \"Week One Again\"\n"
        "    concepts: [\"B\"]\n"
        "    required_readings: [\"Another reading.\"]\n",
        encoding="utf-8",
    )

    with pytest.raises(CourseConfigError, match="Duplicate"):
        load_course_config(bad_path)


def test_empty_concepts_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text(
        "weeks:\n"
        "  - week: 1\n"
        "    title: \"Week One\"\n"
        "    concepts: []\n"
        "    required_readings: [\"Some reading.\"]\n",
        encoding="utf-8",
    )

    with pytest.raises(CourseConfigError, match="concepts"):
        load_course_config(bad_path)


def test_empty_readings_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text(
        "weeks:\n"
        "  - week: 1\n"
        "    title: \"Week One\"\n"
        "    concepts: [\"A\"]\n"
        "    required_readings: []\n",
        encoding="utf-8",
    )

    with pytest.raises(CourseConfigError, match="required_readings"):
        load_course_config(bad_path)


def test_reading_mapping_without_citation_raises_course_config_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "course.yaml"
    bad_path.write_text(
        "weeks:\n"
        "  - week: 1\n"
        "    title: \"Week One\"\n"
        "    concepts: [\"A\"]\n"
        "    required_readings:\n"
        "      - source_file: \"orphan.pdf\"\n",
        encoding="utf-8",
    )

    with pytest.raises(CourseConfigError, match="citation"):
        load_course_config(bad_path)

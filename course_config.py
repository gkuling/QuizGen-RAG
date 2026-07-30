'''
Course schedule configuration loading for QuizGen-RAG.

Loads a course's weekly schedule (title, concepts, and required readings
per week) from a YAML or JSON file into a tree of frozen dataclasses. This
replaces the old hardcoded course_schedule dict in course_time_table.py, so
adapting the pipeline to a different course is a config edit, not a code
edit.

Importing this module requires only the standard library. PyYAML is
imported lazily, only inside load_course_config(), and only when a
.yaml/.yml file is actually being loaded; only yaml.safe_load is ever used,
never yaml.load.
'''

import json
from dataclasses import dataclass
from pathlib import Path


class CourseConfigError(Exception):
    '''Raised when a course configuration file is missing, malformed, or fails validation.'''


class UnknownWeekError(CourseConfigError):
    '''Raised when a requested week number is not present in the loaded course configuration.'''


@dataclass(frozen=True)
class Reading:
    '''
    A single required reading assigned for a week.

    :ivar citation: the human-readable citation or title of the reading.
    :ivar source_file: optional filename of the underlying source document,
        used to match retrieved chunks back to the reading they came from.
    '''

    citation: str
    source_file: str | None = None


@dataclass(frozen=True)
class Week:
    '''
    One week of the course schedule.

    :ivar week: the week number.
    :ivar title: the week's title.
    :ivar concepts: the learning objectives / concepts covered this week.
    :ivar required_readings: the readings assigned for this week.
    '''

    week: int
    title: str
    concepts: tuple[str, ...]
    required_readings: tuple[Reading, ...]


@dataclass(frozen=True)
class CourseConfig:
    '''
    A full course schedule: every week plus optional course-level metadata.

    :ivar weeks: every week in the course, in the order they were loaded.
    :ivar course_id: optional short course identifier, e.g. "AIM2".
    :ivar course_title: optional human-readable course title.
    '''

    weeks: tuple[Week, ...]
    course_id: str | None = None
    course_title: str | None = None

    def get_week(self, week_number: int) -> Week:
        '''
        Look up a single week by its week number.

        :param week_number: the week number to look up.
        :return: the matching Week.
        :raises UnknownWeekError: if no week with that number exists. The
            error message lists every valid week number in this config.
        '''
        for week in self.weeks:
            if week.week == week_number:
                return week
        valid = ", ".join(str(number) for number in self.week_numbers)
        raise UnknownWeekError(
            f"Unknown week number {week_number}. Valid weeks are: {valid}"
        )

    @property
    def week_numbers(self) -> tuple[int, ...]:
        '''The week numbers present in this course, in loaded order.'''
        return tuple(week.week for week in self.weeks)


def _normalize_reading(raw_reading: object, week_number: object) -> Reading:
    '''
    Normalize a single raw reading entry (a plain string or a mapping) into
    a Reading.

    :param raw_reading: the raw reading, either a plain string citation or
        a mapping containing at least a "citation" key.
    :param week_number: the week number this reading belongs to, used only
        to produce an actionable error message.
    :return: the normalized Reading.
    :raises CourseConfigError: if raw_reading is neither a string nor a
        mapping containing a "citation" key.
    '''
    if isinstance(raw_reading, str):
        return Reading(citation=raw_reading)
    if isinstance(raw_reading, dict) and "citation" in raw_reading:
        return Reading(
            citation=raw_reading["citation"],
            source_file=raw_reading.get("source_file"),
        )
    raise CourseConfigError(
        f"Week {week_number}: each required_readings entry must be a string "
        f"or a mapping with a 'citation' key, got: {raw_reading!r}"
    )


def _parse_week(raw_week: object, index: int) -> Week:
    '''
    Parse and validate a single raw week entry from the loaded config.

    :param raw_week: the raw week mapping as loaded from YAML/JSON.
    :param index: the zero-based position of this entry in the top-level
        "weeks" list, used only to produce an actionable error message
        when the week number itself cannot be determined yet.
    :return: the validated Week.
    :raises CourseConfigError: if the entry is missing required fields or
        has a field of the wrong shape.
    '''
    if not isinstance(raw_week, dict):
        raise CourseConfigError(
            f"weeks[{index}] must be a mapping, got: {type(raw_week).__name__}"
        )

    if "week" not in raw_week:
        raise CourseConfigError(f"weeks[{index}] is missing required field 'week'")
    week_number = raw_week["week"]
    # bool is a subclass of int in Python, but True/False are not valid week
    # numbers, so it is rejected explicitly here.
    if isinstance(week_number, bool) or not isinstance(week_number, int):
        raise CourseConfigError(
            f"weeks[{index}]: 'week' must be an int, got: {week_number!r}"
        )

    if not raw_week.get("title"):
        raise CourseConfigError(f"Week {week_number}: missing or empty 'title'")
    title = raw_week["title"]

    concepts = raw_week.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        raise CourseConfigError(
            f"Week {week_number}: 'concepts' must be a non-empty list"
        )

    raw_readings = raw_week.get("required_readings")
    if not isinstance(raw_readings, list) or not raw_readings:
        raise CourseConfigError(
            f"Week {week_number}: 'required_readings' must be a non-empty list"
        )
    readings = tuple(
        _normalize_reading(raw_reading, week_number) for raw_reading in raw_readings
    )

    return Week(
        week=week_number,
        title=title,
        concepts=tuple(concepts),
        required_readings=readings,
    )


def load_course_config(path: str | Path) -> CourseConfig:
    '''
    Load and validate a course configuration from a YAML or JSON file.

    :param path: path to the configuration file. Files ending in .yaml or
        .yml are parsed with yaml.safe_load (PyYAML is imported lazily,
        only here); files ending in .json are parsed with json.load; any
        other suffix raises CourseConfigError.
    :return: the validated CourseConfig.
    :raises CourseConfigError: if the file is missing, is not a supported
        format, or fails validation. Every failure names the offending
        week number and/or field so the fix is actionable.
    '''
    config_path = Path(path)
    if not config_path.is_file():
        raise CourseConfigError(f"Course config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        import yaml
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    elif suffix == ".json":
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = json.load(config_file)
    else:
        raise CourseConfigError(
            f"Unsupported course config format '{suffix}': {config_path}"
        )

    if not isinstance(raw, dict):
        raise CourseConfigError(
            f"Top level of course config must be a mapping, got: {type(raw).__name__}"
        )

    raw_weeks = raw.get("weeks")
    if not isinstance(raw_weeks, list) or not raw_weeks:
        raise CourseConfigError("Course config must have a non-empty 'weeks' list")

    weeks: list[Week] = []
    seen_week_numbers: set[int] = set()
    for index, raw_week in enumerate(raw_weeks):
        week = _parse_week(raw_week, index)
        if week.week in seen_week_numbers:
            raise CourseConfigError(f"Duplicate week number: {week.week}")
        seen_week_numbers.add(week.week)
        weeks.append(week)

    course_meta = raw.get("course")
    course_id = None
    course_title = None
    if isinstance(course_meta, dict):
        course_id = course_meta.get("id")
        course_title = course_meta.get("title")

    return CourseConfig(
        weeks=tuple(weeks),
        course_id=course_id,
        course_title=course_title,
    )

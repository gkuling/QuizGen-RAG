'''
Offline quality checks for generated quiz CSVs (QuizGen-RAG).

This module implements the "structural, necessary-not-sufficient" checks
that can be run on a quiz CSV with no Azure credentials and no vector
store: schema/consistency checks, Bloom's level distribution, near-duplicate
question detection, coverage of a course's concepts/readings, and a
draft-vs-final comparison when round-table refinement was used. It also
emits a blinded rating sheet for human review.

Two analyses that would require an LLM judge -- score_groundedness and
score_bloom_alignment -- are intentionally left as NotImplementedError
stubs with numbered TODOs. They are stubbed rather than faked because an
uncalibrated automated judge is not trustworthy on its own; see their
docstrings for what each would require before it should report a number.

Import-time cost: this module is stdlib-only plus blooms (stdlib-only) and
course_config (stdlib-only at import time; PyYAML is imported lazily by
course_config only when a .yaml file is actually loaded). Nothing here
imports llama-index, numpy, or pandas.
'''

import argparse
import csv
import difflib
import json
import math
import re
import string
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import blooms
import course_config

# Word-count ceiling stated in the quiz-generation system prompt (see
# BTQ_prompt.build_system_prompt); answers longer than this are flagged.
ANSWER_WORD_LIMIT = 200

# Columns present in the legacy (pre-refactor) quiz CSV schema. That schema
# had no question_id, week, source_files, facts, model, or generated_at
# column, so those are synthesized/defaulted on load; see load_quiz.
_LEGACY_ID_COLUMN = "question_id"

# Header for the blinded human-rating CSV emitted by emit_rating_sheet.
_BLIND_RATING_HEADER = [
    "question_id",
    "question",
    "answer",
    "rated_bloom_level",
    "factually_correct",
    "answerable_from_reading",
    "would_use",
    "comments",
]

# Punctuation-stripping table shared by every text-normalization call.
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True)
class QuizItem:
    '''
    One row of a generated quiz CSV.

    Mirrors the 11-column current schema (question_id, week, taxonomy,
    learning_objective, reference, source_files, facts, question, answer,
    model, generated_at) plus three optional refinement columns
    (draft_question, draft_answer, refined) that are only present when the
    quiz was generated with --refine.

    Items loaded from the legacy 6-column schema (learning_objective,
    reference, taxonomy, promtp, question, answer) populate every field,
    but use placeholder values for columns that schema never had: week=0,
    a synthesized question_id, and empty source_files/facts/model/
    generated_at. See load_quiz for details.

    :ivar question_id: unique identifier, e.g. "w03-q007".
    :ivar week: the week number this question targets.
    :ivar taxonomy: the requested Bloom's Taxonomy level, as free text
        from the CSV (validate with blooms.normalize_level before trusting
        it -- it may not be a recognized level).
    :ivar learning_objective: the learning objective/concept text used to
        prompt the question.
    :ivar reference: the citation/reference text for the source reading.
    :ivar source_files: semicolon-joined list of source filenames the
        retrieval step actually used (may be empty).
    :ivar facts: the "top facts" text synthesized by the first LLM call
        and used to ground the question and answer.
    :ivar question: the question text.
    :ivar answer: the answer text.
    :ivar model: the model name used to generate this item.
    :ivar generated_at: ISO-8601 UTC generation timestamp.
    :ivar draft_question: the pre-refinement draft question, if --refine
        was used and this column is present; None otherwise.
    :ivar draft_answer: the pre-refinement draft answer, if --refine was
        used and this column is present; None otherwise.
    :ivar refined: whether round-table refinement was applied to this
        item, if the "refined" column is present; None otherwise.
    '''

    question_id: str
    week: int
    taxonomy: str
    learning_objective: str
    reference: str
    source_files: str
    facts: str
    question: str
    answer: str
    model: str
    generated_at: str
    draft_question: str | None = None
    draft_answer: str | None = None
    refined: bool | None = None


def _get_str(row: dict[str, str | None], key: str) -> str:
    '''Read a required string field from a DictReader row, defaulting to "".'''
    value = row.get(key)
    return value if value is not None else ""


def _get_optional_str(row: dict[str, str | None], key: str) -> str | None:
    '''Read an optional string field, treating missing/None/empty as absent.'''
    value = row.get(key)
    return value if value else None


def _parse_week(raw: str | None) -> int:
    '''
    Best-effort parse of the "week" column to an int.

    :param raw: the raw "week" cell, or None if the column was absent.
    :return: the parsed week number, or 0 if raw is missing, empty, or not
        parseable as an int. Week 0 never matches a real course week, so
        it reads as "unknown" everywhere week is used (coverage_report,
        find_near_duplicates' within/cross-week split, etc.).
    '''
    if raw is None or raw == "":
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _parse_bool(raw: str) -> bool:
    '''Parse a CSV cell as a boolean, tolerant of common textual forms.'''
    return raw.strip().casefold() in {"true", "1", "yes", "y"}


def _row_to_item(row: dict[str, str | None]) -> QuizItem:
    '''Build a QuizItem from one current-schema CSV row.'''
    refined_str = _get_optional_str(row, "refined")
    return QuizItem(
        question_id=_get_str(row, "question_id"),
        week=_parse_week(row.get("week")),
        taxonomy=_get_str(row, "taxonomy"),
        learning_objective=_get_str(row, "learning_objective"),
        reference=_get_str(row, "reference"),
        source_files=_get_str(row, "source_files"),
        facts=_get_str(row, "facts"),
        question=_get_str(row, "question"),
        answer=_get_str(row, "answer"),
        model=_get_str(row, "model"),
        generated_at=_get_str(row, "generated_at"),
        draft_question=_get_optional_str(row, "draft_question"),
        draft_answer=_get_optional_str(row, "draft_answer"),
        refined=_parse_bool(refined_str) if refined_str is not None else None,
    )


def _legacy_row_to_item(row: dict[str, str | None], index: int) -> QuizItem:
    '''Build a QuizItem from one legacy-schema CSV row, synthesizing an id.'''
    return QuizItem(
        question_id=f"legacy-{index:04d}",
        week=0,
        taxonomy=_get_str(row, "taxonomy"),
        learning_objective=_get_str(row, "learning_objective"),
        reference=_get_str(row, "reference"),
        source_files="",
        facts="",
        question=_get_str(row, "question"),
        answer=_get_str(row, "answer"),
        model="",
        generated_at="",
        draft_question=None,
        draft_answer=None,
        refined=None,
    )


def load_quiz(path: str | Path) -> list[QuizItem]:
    '''
    Load quiz items from a CSV file, tolerating both the current 11-column
    schema and the legacy 6-column schema.

    Current schema columns: question_id, week, taxonomy, learning_objective,
    reference, source_files, facts, question, answer, model, generated_at,
    plus optional draft_question, draft_answer, refined. Unknown extra
    columns are ignored.

    Legacy schema columns: learning_objective, reference, taxonomy, promtp,
    question, answer -- no question_id, week, source_files, facts, model,
    or generated_at. This schema is detected purely by the absence of a
    "question_id" column (the one column every current-schema file always
    has). When detected: question_ids are synthesized as "legacy-0001",
    "legacy-0002", ... in row order; week defaults to 0; source_files,
    facts, model, and generated_at default to "". A UserWarning is emitted
    once per call because the missing "facts" column means
    groundedness-related checks (score_groundedness, and check_schema's
    empty_facts finding) cannot be meaningfully run on these rows -- there
    is no way to tell "no facts were recorded" apart from "no facts were
    ever extracted" for legacy data.

    :param path: path to the quiz CSV file.
    :return: one QuizItem per data row, in file order.
    :raises OSError: if the file cannot be opened.
    '''
    csv_path = Path(path)
    items: list[QuizItem] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        legacy = _LEGACY_ID_COLUMN not in fieldnames
        if legacy:
            warnings.warn(
                "load_quiz: legacy CSV schema detected (no question_id "
                "column present). question_ids were synthesized and "
                "groundedness-related checks are unavailable for this "
                "file because the legacy schema has no facts column.",
                UserWarning,
                stacklevel=2,
            )
        for index, row in enumerate(reader, start=1):
            if legacy:
                items.append(_legacy_row_to_item(row, index))
            else:
                items.append(_row_to_item(row))
    return items


@dataclass(frozen=True)
class SchemaFinding:
    '''One schema/consistency problem found in a quiz CSV.'''

    question_id: str
    kind: str
    detail: str


@dataclass(frozen=True)
class SchemaReport:
    '''
    The complete set of schema/consistency findings for a quiz.

    :ivar findings: every SchemaFinding detected, in item order. Empty
        means the quiz is clean by these checks.
    '''

    findings: tuple[SchemaFinding, ...]

    @property
    def is_clean(self) -> bool:
        '''True if no findings were recorded.'''
        return len(self.findings) == 0


def check_schema(items: list[QuizItem]) -> SchemaReport:
    '''
    Run structural/consistency checks over a list of quiz items.

    Flags, per item:
      - empty_question: the question field is empty or missing.
      - empty_answer: the answer field is empty or missing.
      - unknown_taxonomy: blooms.normalize_level(item.taxonomy) is None,
        i.e. the taxonomy string is not a recognized Bloom's level.
      - answer_too_long: the answer has more than ANSWER_WORD_LIMIT words
        (counted via str.split(), i.e. whitespace-separated tokens).
      - empty_facts: the facts field is empty or missing.

    And once per repeated id:
      - duplicate_question_id: the same question_id appears on more than
        one row (one finding per duplicated id, not one per extra row).

    :param items: the quiz items to check.
    :return: a SchemaReport listing every finding.
    '''
    findings: list[SchemaFinding] = []

    id_counts: dict[str, int] = {}
    for item in items:
        id_counts[item.question_id] = id_counts.get(item.question_id, 0) + 1

    reported_duplicate_ids: set[str] = set()
    for item in items:
        if not item.question.strip():
            findings.append(
                SchemaFinding(item.question_id, "empty_question", "question field is empty or missing")
            )
        if not item.answer.strip():
            findings.append(
                SchemaFinding(item.question_id, "empty_answer", "answer field is empty or missing")
            )
        if blooms.normalize_level(item.taxonomy) is None:
            findings.append(
                SchemaFinding(
                    item.question_id,
                    "unknown_taxonomy",
                    f"taxonomy '{item.taxonomy}' is not a recognized Bloom's level",
                )
            )
        word_count = len(item.answer.split())
        if word_count > ANSWER_WORD_LIMIT:
            findings.append(
                SchemaFinding(
                    item.question_id,
                    "answer_too_long",
                    f"answer has {word_count} words, exceeds the {ANSWER_WORD_LIMIT}-word limit",
                )
            )
        if not item.facts.strip():
            findings.append(
                SchemaFinding(item.question_id, "empty_facts", "facts field is empty or missing")
            )
        if id_counts[item.question_id] > 1 and item.question_id not in reported_duplicate_ids:
            reported_duplicate_ids.add(item.question_id)
            findings.append(
                SchemaFinding(
                    item.question_id,
                    "duplicate_question_id",
                    f"question_id appears {id_counts[item.question_id]} times",
                )
            )

    return SchemaReport(findings=tuple(findings))


@dataclass(frozen=True)
class BloomReport:
    '''
    Bloom's Taxonomy level distribution over a set of quiz items.

    :ivar counts: raw item counts, keyed by each of blooms.BLOOM_LEVELS
        plus "unknown" (items whose taxonomy did not normalize to a
        canonical level). Canonical levels with zero items are still
        present with count 0.
    :ivar proportions: counts divided by the total item count, same keys
        as counts. All zero if there are no items.
    :ivar entropy: normalized Shannon entropy in [0, 1], computed as
        H / log2(6) where H = -sum(p * log2(p)) over the six canonical
        levels only (the "unknown" bucket is excluded from the sum, by
        definition 0 * log2(0) = 0 for any level with zero items), and p
        is that level's proportion of ALL items (canonical + unknown). A
        maximally even split across the six canonical levels with no
        unknowns gives entropy 1.0; an all-unknown or single-level quiz
        gives entropy 0.0.
    '''

    counts: dict[str, int]
    proportions: dict[str, float]
    entropy: float


def bloom_distribution(items: list[QuizItem]) -> BloomReport:
    '''
    Compute the Bloom's level distribution and normalized entropy for a
    set of quiz items. See BloomReport for the exact entropy formula.

    :param items: the quiz items to summarize.
    :return: the BloomReport. If items is empty, all counts/proportions
        are 0 and entropy is 0.0.
    '''
    total = len(items)
    counts: dict[str, int] = {level: 0 for level in blooms.BLOOM_LEVELS}
    counts["unknown"] = 0
    for item in items:
        level = blooms.normalize_level(item.taxonomy)
        if level is None:
            counts["unknown"] += 1
        else:
            counts[level] += 1

    if total > 0:
        proportions = {key: value / total for key, value in counts.items()}
    else:
        proportions = {key: 0.0 for key in counts}

    entropy_bits = 0.0
    for level in blooms.BLOOM_LEVELS:
        proportion = proportions[level]
        if proportion > 0.0:
            entropy_bits -= proportion * math.log2(proportion)
    normalized_entropy = entropy_bits / math.log2(len(blooms.BLOOM_LEVELS)) if total > 0 else 0.0

    return BloomReport(counts=counts, proportions=proportions, entropy=normalized_entropy)


def _normalize_text(text: str) -> str:
    '''
    Normalize free text for fuzzy comparison: casefold, strip ASCII/Latin-1
    punctuation, and collapse whitespace runs to single spaces.
    '''
    text = text.casefold()
    text = text.translate(_PUNCTUATION_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class DuplicatePair:
    '''
    A pair of questions whose normalized text similarity is at or above
    the near-duplicate threshold.

    :ivar question_id_a: the first item's question_id.
    :ivar question_id_b: the second item's question_id.
    :ivar similarity: the difflib.SequenceMatcher.ratio() between the two
        normalized question texts, in [0, 1].
    :ivar scope: "within_week" if both items target the same week,
        "cross_week" otherwise.
    '''

    question_id_a: str
    question_id_b: str
    similarity: float
    scope: str


def find_near_duplicates(items: list[QuizItem], threshold: float = 0.85) -> list[DuplicatePair]:
    '''
    Find pairs of questions that are near-duplicates of each other.

    Text is normalized (see _normalize_text) before comparison. Every pair
    of items is compared, which is O(n^2) in the number of items; this is
    fine at the scale a single course's quiz reaches (tens to low
    hundreds of questions per week, not thousands), so no smarter
    (e.g. blocking/indexing) approach is used. difflib.SequenceMatcher's
    quick_ratio() is checked first, since it is a cheap upper bound on the
    real ratio() -- skipping a pair whose quick_ratio() is already below
    threshold cannot discard a true match.

    :param items: the quiz items to compare.
    :param threshold: minimum similarity ratio, in [0, 1], to report a
        pair as a near-duplicate.
    :return: every DuplicatePair found, each reported once with scope set
        to "within_week" (both items have the same item.week) or
        "cross_week" (they differ).
    '''
    pairs: list[DuplicatePair] = []
    normalized = [_normalize_text(item.question) for item in items]
    count = len(items)
    for i in range(count):
        matcher = difflib.SequenceMatcher(None, normalized[i], "")
        for j in range(i + 1, count):
            matcher.set_seq2(normalized[j])
            if matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio >= threshold:
                item_a, item_b = items[i], items[j]
                scope = "within_week" if item_a.week == item_b.week else "cross_week"
                pairs.append(DuplicatePair(item_a.question_id, item_b.question_id, ratio, scope))
    return pairs


@dataclass(frozen=True)
class CoverageReport:
    '''
    Coverage of one course week's concepts and required readings by a
    quiz.

    Concept/reading "coverage" matching is a best-effort text heuristic
    (see _heuristic_text_match): a concept or reading counts as covered if
    at least one same-week quiz item has a learning_objective/reference
    that, once normalized, is a substring of it (or vice versa) or shares
    a significant word (length > 3) with it. This is NOT semantic
    matching -- it can miss real coverage when wording differs a lot, so
    "uncovered" should read as "probably not covered, worth a human
    look", not as ground truth.

    :ivar week_number: the week this report covers.
    :ivar total_concept_count: number of concepts in the week's config.
    :ivar covered_concept_count: number of those concepts matched by at
        least one same-week quiz item.
    :ivar uncovered_concepts: concept text for every concept with zero
        matching quiz items.
    :ivar total_reading_count: number of required readings in the week's
        config.
    :ivar covered_reading_count: number of those readings matched by at
        least one same-week quiz item.
    :ivar uncovered_readings: citation text for every reading with zero
        matching quiz items.
    '''

    week_number: int
    total_concept_count: int
    covered_concept_count: int
    uncovered_concepts: tuple[str, ...]
    total_reading_count: int
    covered_reading_count: int
    uncovered_readings: tuple[str, ...]


def _significant_words(normalized_text: str) -> set[str]:
    '''
    Words of length > 3 in an already-normalized (_normalize_text)
    string, excluding pure-digit tokens.

    Pure-digit tokens (almost always publication years in this codebase's
    citation strings) are excluded deliberately: two different readings
    assigned in the same week are frequently published in the same year,
    which would otherwise make coverage_report match a quiz item to every
    same-year reading regardless of topic or author.
    '''
    return {word for word in normalized_text.split() if len(word) > 3 and not word.isdigit()}


def _heuristic_text_match(text_a: str, text_b: str) -> bool:
    '''
    Best-effort match between two free-text strings for coverage_report.

    True if, once both are normalized (_normalize_text), one is a
    substring of the other, or they share at least one significant word
    (see _significant_words: length > 3 and not a bare number, so short
    connector words like "et", "al", "the" and bare publication years
    never count on their own). This has no semantic understanding -- it
    is a plain textual heuristic, documented as such on CoverageReport.
    '''
    norm_a = _normalize_text(text_a)
    norm_b = _normalize_text(text_b)
    if not norm_a or not norm_b:
        return False
    if norm_a in norm_b or norm_b in norm_a:
        return True
    return bool(_significant_words(norm_a) & _significant_words(norm_b))


def _reading_covered(reading: course_config.Reading, week_items: list[QuizItem]) -> bool:
    '''True if any item in week_items matches reading by source_file or citation text.'''
    for item in week_items:
        if reading.source_file:
            files = {name.strip().casefold() for name in item.source_files.split(";") if name.strip()}
            if reading.source_file.strip().casefold() in files:
                return True
        if _heuristic_text_match(reading.citation, item.reference):
            return True
    return False


def coverage_report(items: list[QuizItem], config: course_config.CourseConfig, week_number: int) -> CoverageReport:
    '''
    Report which of a week's concepts and required readings have zero
    matching quiz questions.

    :param items: all loaded quiz items (only those with item.week ==
        week_number are considered).
    :param config: the course configuration to check coverage against.
    :param week_number: which week of config to check.
    :return: the CoverageReport for that week.
    :raises course_config.UnknownWeekError: if week_number is not present
        in config.
    '''
    week = config.get_week(week_number)
    week_items = [item for item in items if item.week == week_number]

    uncovered_concepts = tuple(
        concept
        for concept in week.concepts
        if not any(_heuristic_text_match(concept, item.learning_objective) for item in week_items)
    )
    uncovered_readings = tuple(
        reading.citation for reading in week.required_readings if not _reading_covered(reading, week_items)
    )

    return CoverageReport(
        week_number=week_number,
        total_concept_count=len(week.concepts),
        covered_concept_count=len(week.concepts) - len(uncovered_concepts),
        uncovered_concepts=uncovered_concepts,
        total_reading_count=len(week.required_readings),
        covered_reading_count=len(week.required_readings) - len(uncovered_readings),
        uncovered_readings=uncovered_readings,
    )


@dataclass(frozen=True)
class DraftFinalComparison:
    '''
    Comparison between an item's draft (pre-refinement) and final Q&A.

    :ivar question_id: the item's question_id.
    :ivar question_similarity: difflib.SequenceMatcher.ratio() between the
        normalized draft_question and normalized final question.
    :ivar question_word_delta: len(question.split()) minus
        len(draft_question.split()); positive means the final question is
        longer than the draft.
    :ivar answer_word_delta: len(answer.split()) minus
        len(draft_answer.split()); positive means the final answer is
        longer than the draft.
    '''

    question_id: str
    question_similarity: float
    question_word_delta: int
    answer_word_delta: int


def compare_draft_final(items: list[QuizItem]) -> list[DraftFinalComparison]:
    '''
    Compare draft vs. final question/answer for items that carry
    round-table refinement columns.

    Uses the same normalize-then-SequenceMatcher machinery as
    find_near_duplicates for the question-text similarity.

    :param items: the quiz items to compare.
    :return: one DraftFinalComparison per item that has both a non-empty
        draft_question and a non-empty draft_answer. Items without those
        columns (i.e. --refine was not used) are skipped, so this returns
        an empty list for a quiz with no refinement columns at all --
        that is the expected, non-error "nothing to compare" case.
    '''
    comparisons: list[DraftFinalComparison] = []
    for item in items:
        if not item.draft_question or not item.draft_answer:
            continue
        norm_draft = _normalize_text(item.draft_question)
        norm_final = _normalize_text(item.question)
        similarity = difflib.SequenceMatcher(None, norm_draft, norm_final).ratio()
        question_delta = len(item.question.split()) - len(item.draft_question.split())
        answer_delta = len(item.answer.split()) - len(item.draft_answer.split())
        comparisons.append(DraftFinalComparison(item.question_id, similarity, question_delta, answer_delta))
    return comparisons


def emit_rating_sheet(items: list[QuizItem], out_path: str | Path, blind: bool = True) -> None:
    '''
    Write a CSV for human expert rating of the generated questions.

    When blind (the default), the header is exactly:
      question_id, question, answer, rated_bloom_level, factually_correct,
      answerable_from_reading, would_use, comments
    with the last five columns left empty for the rater to fill in, and
    the requested taxonomy deliberately omitted so raters judge the
    Bloom's level blind to what was asked for (this is what makes
    score_bloom_alignment's future human-agreement baseline meaningful).

    When not blind, a "requested_taxonomy" column is inserted after
    "answer" (before the empty rating columns), revealing item.taxonomy.
    This mode exists for cases where blinding is not needed (e.g. a
    designer reviewing their own output), not for the calibration
    workflow.

    :param items: the quiz items to write.
    :param out_path: where to write the CSV.
    :param blind: whether to omit the requested Bloom's level (default
        True).
    '''
    if blind:
        header = list(_BLIND_RATING_HEADER)
    else:
        header = [
            "question_id",
            "question",
            "answer",
            "requested_taxonomy",
            "rated_bloom_level",
            "factually_correct",
            "answerable_from_reading",
            "would_use",
            "comments",
        ]

    path = Path(out_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for item in items:
            row = [item.question_id, item.question, item.answer]
            if not blind:
                row.append(item.taxonomy)
            row.extend(["", "", "", "", ""])
            writer.writerow(row)


def score_groundedness(items: list[QuizItem], query_engine: object, judge_llm: object) -> None:
    '''
    NOT IMPLEMENTED. Would score each item's factual groundedness against
    the retrieval corpus. Left as a stub because an automated judge that
    has never been checked against a human rater is not trustworthy, and
    shipping a number here would recreate exactly the "nothing shows the
    generated questions are any good" problem this evaluation suite
    exists to fix.

    Intended implementation (TODO, in order):

    1. TODO: Decompose each item's answer into atomic factual claims (one
       independently verifiable statement per claim), using judge_llm.
    2. TODO: For each claim, check entailment against the item's own
       "facts" field (the text synthesized by the first LLM call at
       generation time, already stored in the quiz CSV -- no retrieval
       needed for this half).
    3. TODO: For each claim, also check entailment against chunks
       re-retrieved live from the course corpus via query_engine (a
       llama-index query engine over the persisted vector store passed
       via --vec-store), since the stored "facts" text may itself have
       been a lossy or incomplete summary of the corpus (see the README's
       Limitations section on the two-LLM-call design).
    4. TODO: Aggregate per item (e.g. fraction of claims entailed by
       facts and/or re-retrieved chunks) and across the quiz.
    5. TODO: CRITICAL -- before reporting any number from this function,
       calibrate judge_llm's claim/entailment judgments against a sample
       of human "factually_correct" labels collected via
       emit_rating_sheet, and report the agreement (e.g. Cohen's kappa),
       exactly as score_bloom_alignment must do for its own judge. Do not
       surface a groundedness score to users before this step exists.

    llama_index's FaithfulnessEvaluator is a plausible starting point for
    steps 1-3 (it already does claim-vs-context entailment), adapted to
    also check the stored facts field, but it has not been wired up,
    tried, or validated here -- naming it is not an endorsement that it
    is sufficient once calibration (step 5) is done.

    :param items: the quiz items to score.
    :param query_engine: a llama-index query engine over the course
        vector store, used to re-retrieve chunks per item. Not
        constructed by this function; callers build it from --vec-store.
    :param judge_llm: the LLM used to decompose claims and judge
        entailment. Not constructed by this function.
    :raises NotImplementedError: always; this is an intentional stub.
    '''
    raise NotImplementedError(
        "score_groundedness is not implemented. It requires claim "
        "decomposition and entailment checking against stored facts and "
        "re-retrieved chunks (Azure OpenAI credentials plus a populated "
        "--vec-store), and calibration of the judge against human labels "
        "before any number should be reported. See the function "
        "docstring for the full TODO list."
    )


def score_bloom_alignment(items: list[QuizItem], judge_llm: object) -> None:
    '''
    NOT IMPLEMENTED. Would score whether each item's actual cognitive
    demand matches its requested Bloom's Taxonomy level. Left as a stub
    for the same reason as score_groundedness: an LLM judge that has
    never been checked against a human rater is not trustworthy, and this
    metric in particular has a plausible failure mode worth naming before
    anyone reports a number from it.

    Intended implementation (TODO, in order):

    1. TODO: For each item, show judge_llm the question and answer ONLY
       (blind to the requested taxonomy -- reuse emit_rating_sheet's
       blind=True sheet as the template for what the judge sees, so the
       human and automated ratings are directly comparable) and have it
       classify the item into one of the six blooms.BLOOM_LEVELS.
    2. TODO: Build a 6x6 confusion matrix: requested level (rows) vs.
       judge-assigned level (columns), across the whole quiz.
    3. TODO: CRITICAL -- before reporting any alignment number, collect
       the same blind classification from at least one human rater (via
       the rating sheet's rated_bloom_level column) on a shared sample,
       and report judge-vs-human agreement (e.g. weighted Cohen's kappa,
       since Bloom's levels are ordinal) alongside the requested-vs-judge
       confusion matrix. If judge-vs-human agreement is not established,
       the requested-vs-judge matrix alone does not tell you whether
       misalignment is real or a judge artifact.
    4. TODO: Once calibrated, the specific hypothesis worth testing
       (stated in the README) is whether Analyzing/Evaluating requests
       collapse toward Understanding-level answers more often than
       Knowledge/Applying requests do -- report the confusion matrix in a
       way that makes that comparison legible (e.g. row-normalized).

    :param items: the quiz items to score.
    :param judge_llm: the LLM used to blindly classify each item's
        cognitive level. Not constructed by this function.
    :raises NotImplementedError: always; this is an intentional stub.
    '''
    raise NotImplementedError(
        "score_bloom_alignment is not implemented. It requires a blind "
        "LLM judge (Azure OpenAI credentials) and calibration of that "
        "judge against human ratings collected via emit_rating_sheet "
        "before any agreement number should be reported. See the "
        "function docstring for the full TODO list."
    )


def render_text_report(
    items: list[QuizItem],
    schema_report: SchemaReport,
    bloom_report: BloomReport,
    duplicates: list[DuplicatePair],
    coverage_reports: list[CoverageReport] | None = None,
    draft_final: list[DraftFinalComparison] | None = None,
) -> str:
    '''
    Render every implemented analysis as a human-readable text report.

    :param items: the quiz items the other arguments were computed from
        (only used for the item count header line).
    :param schema_report: output of check_schema.
    :param bloom_report: output of bloom_distribution.
    :param duplicates: output of find_near_duplicates.
    :param coverage_reports: output of coverage_report, one per week
        checked; omit or pass None/[] if no --course-config was given.
    :param draft_final: output of compare_draft_final; omit or pass
        None/[] if the quiz has no refinement columns.
    :return: the assembled multi-section text report.
    '''
    lines: list[str] = []
    lines.append("QuizGen-RAG evaluation report")
    lines.append(f"Items loaded: {len(items)}")
    lines.append("")

    lines.append("== Schema checks ==")
    if schema_report.findings:
        for finding in schema_report.findings:
            lines.append(f"  [{finding.kind}] {finding.question_id}: {finding.detail}")
    else:
        lines.append("  No schema findings.")
    lines.append("")

    lines.append("== Bloom's level distribution ==")
    for level in blooms.BLOOM_LEVELS:
        count = bloom_report.counts[level]
        proportion = bloom_report.proportions[level]
        lines.append(f"  {level}: {count} ({proportion:.1%})")
    lines.append(f"  unknown: {bloom_report.counts['unknown']} ({bloom_report.proportions['unknown']:.1%})")
    lines.append(f"  Normalized entropy: {bloom_report.entropy:.3f}")
    lines.append("")

    lines.append("== Near-duplicate questions ==")
    if duplicates:
        for pair in duplicates:
            lines.append(
                f"  {pair.question_id_a} <-> {pair.question_id_b}: "
                f"{pair.similarity:.3f} ({pair.scope})"
            )
    else:
        lines.append("  No near-duplicate questions found.")
    lines.append("")

    if coverage_reports:
        lines.append("== Coverage ==")
        for coverage in coverage_reports:
            lines.append(f"  Week {coverage.week_number}:")
            lines.append(
                f"    Concepts covered: {coverage.covered_concept_count}/{coverage.total_concept_count}"
            )
            for concept in coverage.uncovered_concepts:
                lines.append(f"      MISSING concept: {concept}")
            lines.append(
                f"    Readings covered: {coverage.covered_reading_count}/{coverage.total_reading_count}"
            )
            for reading in coverage.uncovered_readings:
                lines.append(f"      MISSING reading: {reading}")
        lines.append("")

    if draft_final:
        lines.append("== Draft vs final comparison ==")
        for comparison in draft_final:
            lines.append(
                f"  {comparison.question_id}: similarity={comparison.question_similarity:.3f}, "
                f"question_word_delta={comparison.question_word_delta}, "
                f"answer_word_delta={comparison.answer_word_delta}"
            )
        lines.append("")

    return "\n".join(lines)


def render_json_report(
    items: list[QuizItem],
    schema_report: SchemaReport,
    bloom_report: BloomReport,
    duplicates: list[DuplicatePair],
    coverage_reports: list[CoverageReport] | None = None,
    draft_final: list[DraftFinalComparison] | None = None,
) -> dict[str, object]:
    '''
    Render every implemented analysis as a JSON-serializable dict.

    Same inputs/semantics as render_text_report; see there for parameter
    descriptions.

    :return: a dict safe to pass to json.dumps directly.
    '''
    return {
        "item_count": len(items),
        "schema_findings": [asdict(finding) for finding in schema_report.findings],
        "bloom_distribution": {
            "counts": bloom_report.counts,
            "proportions": bloom_report.proportions,
            "normalized_entropy": bloom_report.entropy,
        },
        "near_duplicates": [asdict(pair) for pair in duplicates],
        "coverage": [asdict(coverage) for coverage in (coverage_reports or [])],
        "draft_final_comparison": [asdict(comparison) for comparison in (draft_final or [])],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    '''Build the argparse parser for the evaluate_quiz.py CLI.'''
    parser = argparse.ArgumentParser(
        prog="evaluate_quiz.py",
        description=(
            "Offline quality checks for a generated quiz CSV. The default "
            "checks (schema, Bloom's distribution, near-duplicates, "
            "coverage, draft-vs-final) require no API keys. "
            "--check-groundedness and --check-bloom-alignment are "
            "intentionally unimplemented stubs; requesting them exits "
            "with a clean explanation instead of a traceback."
        ),
    )
    parser.add_argument("--quiz-csv", required=True, type=Path, help="Path to the quiz CSV to evaluate.")
    parser.add_argument(
        "--course-config",
        type=Path,
        default=None,
        help="Optional course config YAML/JSON; enables the coverage report.",
    )
    parser.add_argument(
        "--week-number",
        type=int,
        default=None,
        help="Restrict the coverage report to a single week. Default: every week present in --quiz-csv.",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold in [0, 1] for find_near_duplicates (default: 0.85).",
    )
    parser.add_argument("--report-json", type=Path, default=None, help="Optional path to write the JSON report.")
    parser.add_argument(
        "--emit-rating-sheet",
        type=Path,
        default=None,
        help="Optional path to write a blinded human-rating CSV.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if check_schema reports any findings.",
    )
    parser.add_argument(
        "--check-groundedness",
        action="store_true",
        help="Attempt score_groundedness. Not implemented; exits with status 2 and a one-line explanation.",
    )
    parser.add_argument(
        "--check-bloom-alignment",
        action="store_true",
        help="Attempt score_bloom_alignment. Not implemented; exits with status 2 and a one-line explanation.",
    )
    parser.add_argument(
        "--vec-store",
        type=str,
        default=None,
        help="Path to a persisted vector store. Only meaningful to the (unimplemented) groundedness check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    '''
    CLI entry point.

    Exit codes: 0 if the quiz loads cleanly and no unimplemented check was
    requested; 1 if --strict was given and check_schema found anything;
    2 if --check-groundedness and/or --check-bloom-alignment was given
    (their NotImplementedError is caught here, at the CLI boundary, and
    reported as one clean line -- never a traceback). If both a stub flag
    and --strict apply, exit 2 takes precedence.

    :param argv: argument list to parse; None means use sys.argv.
    :return: the process exit code.
    '''
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    items = load_quiz(args.quiz_csv)

    schema_report = check_schema(items)
    bloom_report = bloom_distribution(items)
    duplicates = find_near_duplicates(items, threshold=args.duplicate_threshold)
    draft_final = compare_draft_final(items)

    coverage_reports: list[CoverageReport] = []
    if args.course_config is not None:
        config = course_config.load_course_config(args.course_config)
        if args.week_number is not None:
            week_numbers = [args.week_number]
        else:
            week_numbers = sorted({item.week for item in items})
        for week_number in week_numbers:
            try:
                coverage_reports.append(coverage_report(items, config, week_number))
            except course_config.UnknownWeekError as exc:
                print(f"Coverage skipped for week {week_number}: {exc}")

    print(
        render_text_report(items, schema_report, bloom_report, duplicates, coverage_reports, draft_final)
    )

    if args.report_json is not None:
        report = render_json_report(items, schema_report, bloom_report, duplicates, coverage_reports, draft_final)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.emit_rating_sheet is not None:
        emit_rating_sheet(items, args.emit_rating_sheet, blind=True)

    exit_code = 1 if (args.strict and schema_report.findings) else 0

    if args.check_groundedness:
        try:
            score_groundedness(items, query_engine=None, judge_llm=None)
        except NotImplementedError:
            print(
                "Groundedness scoring is not implemented: it requires "
                "Azure OpenAI credentials, a populated --vec-store to "
                "re-retrieve chunks, and calibration of the automated "
                "judge against human labels before any number is reported."
            )
        exit_code = 2

    if args.check_bloom_alignment:
        try:
            score_bloom_alignment(items, judge_llm=None)
        except NotImplementedError:
            print(
                "Bloom alignment scoring is not implemented: it requires "
                "an LLM judge (Azure OpenAI credentials) and calibration "
                "against human ratings collected via emit_rating_sheet "
                "before any number is reported."
            )
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

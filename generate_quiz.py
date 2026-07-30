'''
Quiz question generation for QuizGen-RAG.

Given a course configuration and a persisted vector store of the course
readings, this script builds one grounded prompt per (reading, concept,
Bloom's level) slot, asks an Azure OpenAI chat deployment to author a
question/answer pair from that prompt, optionally refines each pair with
the round_table multi-agent review, and writes the results to a CSV file.

Fixes folded in from the original prototype: no more use of the eval
builtin to parse the LLM's JSON response (see _parse_qa_response); a
single AZURE_OPENAI_API_KEY
environment variable (via azure_config) instead of the old _Z-suffixed
one; a seeded shuffle instead of an unseeded one; validation runs in an
order that lets `--week-number 99` fail cleanly without ever touching
Azure; the course schedule comes from a YAML/JSON config (course_config)
instead of a hardcoded dict; and the CSV schema now records a question_id,
the grounding facts, and their source files, so a saved quiz can later be
checked for groundedness.

Nothing executes at import time: no credentials are read, no client is
constructed, and no index is loaded until generate_quiz() (or main()) is
actually called.
'''

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.llms import ChatMessage

import round_table
from azure_config import MissingCredentialsError, configure_llama_settings, load_azure_settings
from blooms import BLOOM_LEVELS
from BTQ_prompt import build_question_prompt, build_system_prompt
from course_config import CourseConfigError, Week, load_course_config

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_json_loads(text: str) -> dict | None:
    '''
    Attempt to parse text as a JSON object, returning None instead of
    raising on failure.

    :param text: the text to parse.
    :return: the parsed dict, or None if text is not valid JSON, or is
        valid JSON that does not parse to a dict.
    '''
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_qa_response(raw: str) -> dict:
    '''
    Parse an LLM's raw text response into a {"question": ..., "answer":
    ...} dict.

    Tolerates the common ways a chat model wraps JSON in prose: Markdown
    code fences (```json ... ``` or ``` ... ```) and, failing that, falls
    back to extracting the first {...} span from the text. The eval
    builtin is never used.

    :param raw: the raw response text from the chat LLM.
    :return: the parsed dict, guaranteed to contain "question" and
        "answer" keys.
    :raises ValueError: if raw is empty or is not valid JSON (even after
        fence-stripping and brace-extraction), or is valid JSON that is
        not an object containing both "question" and "answer" keys.
    '''
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Cannot parse question/answer JSON from an empty response")

    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
        if text.lower().startswith("json"):
            text = text[len("json"):].strip()

    parsed = _try_json_loads(text)
    if parsed is None:
        match = _JSON_OBJECT_RE.search(text)
        if match is not None:
            parsed = _try_json_loads(match.group(0))

    if parsed is None:
        raise ValueError(f"Could not parse JSON from response: {raw[:200]!r}")
    if "question" not in parsed or "answer" not in parsed:
        raise ValueError(
            f"Parsed JSON is missing 'question' and/or 'answer' keys: {parsed!r}"
        )
    return parsed


def build_question_slots(week: Week, num_questions: int, seed: int) -> list[dict]:
    '''
    Build the list of question "slots" to generate for one week: the
    cross product of every required reading, every concept, and every
    Bloom's Taxonomy level.

    :param week: the week to build slots for.
    :param num_questions: the target number of slots. If -1 (or any
        other non-positive value), every slot in the cross product is
        returned exactly once, unshuffled. If positive, the cross product
        is repeated (ceil(num_questions / len(slots)) times, if needed),
        shuffled deterministically, and truncated to exactly
        num_questions slots.
    :param seed: seed for the deterministic shuffle used when
        num_questions is positive. Ignored (no shuffle performed) when
        num_questions is non-positive.
    :return: a list of slot dicts, each with "reference" (the reading's
        citation text), "objective" (the concept text), and "level" (the
        Bloom's Taxonomy level name) keys.
    '''
    slots = [
        {"reference": reading.citation, "objective": concept, "level": level}
        for reading in week.required_readings
        for concept in week.concepts
        for level in BLOOM_LEVELS
    ]
    if not slots:
        return slots

    repeat_count = math.ceil(num_questions / len(slots)) if num_questions > 0 else 1
    slots = slots * repeat_count

    if num_questions > 0:
        random.Random(seed).shuffle(slots)
        slots = slots[:num_questions]

    return slots


def generate_quiz(
    week_number: int,
    num_questions: int,
    output_dir: str | Path,
    vec_store: str | Path,
    course_config_path: str | Path,
    seed: int = 0,
    refine: bool = False,
    refine_rounds: int = 1,
) -> Path:
    '''
    Generate one week's quiz and write it to a CSV file.

    Validation happens in an order chosen so that a bad week number or a
    missing vector store fails cleanly, without ever needing Azure
    credentials: the course config is loaded and the week resolved
    first, then the vector store path's existence is checked, and only
    after both succeed are Azure OpenAI settings loaded and the index
    opened.

    :param week_number: the week number to generate a quiz for.
    :param num_questions: the target number of questions; -1 means
        "generate every (reading, concept, Bloom's level) slot exactly
        once" (see build_question_slots).
    :param output_dir: directory the output CSV is written into. Created
        if it does not already exist.
    :param vec_store: path to the persisted llama-index vector store
        directory (as built by build_RAG.py).
    :param course_config_path: path to the course configuration YAML/JSON
        file.
    :param seed: seed for the deterministic slot shuffle used when
        num_questions is positive.
    :param refine: if True, run each draft question/answer pair through
        round_table.refine_qa before recording it.
    :param refine_rounds: number of round-table discussion rounds to run
        per question when refine is True. Ignored otherwise.
    :return: the path of the written CSV file.
    :raises CourseConfigError: if the course config file is
        missing/malformed, or (as the UnknownWeekError subclass) if
        week_number is not a valid week in that config.
    :raises FileNotFoundError: if the vec_store directory does not exist.
    :raises MissingCredentialsError: if the required Azure OpenAI
        environment variables are not set.
    '''
    course_config = load_course_config(course_config_path)
    week = course_config.get_week(week_number)

    vec_store_path = Path(vec_store)
    if not vec_store_path.is_dir():
        raise FileNotFoundError(
            f"Vector store directory not found: {vec_store_path}. "
            "Build it first with build_RAG.py."
        )

    azure_settings = load_azure_settings()
    configure_llama_settings(azure_settings)

    storage_context = StorageContext.from_defaults(persist_dir=str(vec_store_path))
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine()
    llm = Settings.llm

    def chat_fn(system_prompt: str, user_prompt: str) -> str:
        '''Adapter matching round_table.ChatFn, backed by the configured chat LLM.'''
        response = llm.chat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ])
        return response.message.content

    def query_fn(query_text: str) -> str:
        '''Adapter matching round_table's query_fn, backed by the loaded query engine.'''
        return str(query_engine.query(query_text))

    system_prompt = build_system_prompt()
    slots = build_question_slots(week, num_questions, seed)
    run_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows: list[dict] = []
    failed = 0

    for slot_index, slot in enumerate(slots, start=1):
        question_id = f"w{week_number:02d}-q{slot_index:03d}"

        grounded = build_question_prompt(
            slot["objective"], slot["reference"], slot["level"], query_engine
        )
        response = llm.chat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=grounded.prompt),
        ])
        raw = response.message.content
        try:
            qa = _parse_qa_response(raw)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Question %s: failed to parse LLM response (%s); raw=%.200r",
                question_id, exc, raw,
            )
            failed += 1
            continue

        draft_question = qa["question"]
        draft_answer = qa["answer"]
        final_question, final_answer, refined_flag = draft_question, draft_answer, False

        if refine:
            criteria = (
                f"Target Bloom's Taxonomy level: {slot['level']}\n\n"
                f"Grounding facts:\n{grounded.facts}"
            )
            try:
                course_context = round_table.fetch_course_context(query_fn, qa)
                refined_qa = round_table.refine_qa(
                    qa, criteria, course_context, chat_fn,
                    rounds=refine_rounds, parse_fn=_parse_qa_response,
                )
                final_question = refined_qa["question"]
                final_answer = refined_qa["answer"]
                refined_flag = True
            except ValueError as exc:
                logger.warning(
                    "Question %s: refinement failed, falling back to draft: %s",
                    question_id, exc,
                )

        row = {
            "question_id": question_id,
            "week": week_number,
            "taxonomy": slot["level"],
            "learning_objective": slot["objective"],
            "reference": slot["reference"],
            "source_files": ";".join(grounded.source_files),
            "facts": grounded.facts,
            "question": final_question,
            "answer": final_answer,
            "model": azure_settings.chat_deployment,
            "generated_at": run_timestamp,
        }
        if refine:
            row["draft_question"] = draft_question
            row["draft_answer"] = draft_answer
            row["refined"] = refined_flag
        rows.append(row)

    columns = [
        "question_id", "week", "taxonomy", "learning_objective", "reference",
        "source_files", "facts", "question", "answer", "model", "generated_at",
    ]
    if refine:
        columns += ["draft_question", "draft_answer", "refined"]

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / f"week_{week_number:02d}_quiz.csv"

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_path, index=False)

    print(
        f"Generated {len(rows)}/{len(slots)} questions "
        f"({failed} responses failed to parse)"
    )

    return output_path


def main(argv: list[str] | None = None) -> int:
    '''
    Command-line entry point for quiz generation.

    :param argv: argument list to parse (excluding the program name); if
        None, sys.argv[1:] is used.
    :return: process exit code: 0 on success, 1 on a clean configuration
        error (bad week number, malformed course config, missing vector
        store, or missing Azure credentials), 2 if --vec-store is missing
        and QUIZGEN_VEC_STORE is not set.
    '''
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate quiz questions based on the course schedule."
    )
    parser.add_argument(
        "--week-number", type=int, required=True,
        help="The week number to generate quiz questions for.",
    )
    parser.add_argument(
        "--num-questions", type=int, default=-1,
        help="Target number of questions to generate. -1 (default) generates "
             "every (reading, concept, Bloom's level) slot exactly once.",
    )
    parser.add_argument(
        "--vec-store", type=str, default=os.environ.get("QUIZGEN_VEC_STORE"),
        help="Path to the persisted vector store directory (as built by "
             "build_RAG.py). Falls back to the QUIZGEN_VEC_STORE environment "
             "variable if not given; required one way or the other.",
    )
    parser.add_argument(
        "--course-config", type=str, default="configs/aim2_course.yaml",
        help="Path to the course configuration YAML/JSON file.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Directory the output CSV is written into.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the deterministic slot shuffle used when --num-questions "
             "is positive.",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="Run each draft question/answer pair through the round_table "
             "multi-agent refinement before recording it. Costs "
             "(--refine-rounds x 4 + 2) extra LLM calls per question: four "
             "persona reviews per round, plus one course-context retrieval "
             "and one synthesizer call.",
    )
    parser.add_argument(
        "--refine-rounds", type=int, default=1,
        help="Number of round-table discussion rounds per question when "
             "--refine is set.",
    )
    args = parser.parse_args(argv)

    if not args.vec_store:
        print(
            "Error: --vec-store is required (or set the QUIZGEN_VEC_STORE "
            "environment variable).",
            file=sys.stderr,
        )
        return 2

    try:
        output_path = generate_quiz(
            week_number=args.week_number,
            num_questions=args.num_questions,
            output_dir=args.output_dir,
            vec_store=args.vec_store,
            course_config_path=args.course_config,
            seed=args.seed,
            refine=args.refine,
            refine_rounds=args.refine_rounds,
        )
    except CourseConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except MissingCredentialsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Quiz saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

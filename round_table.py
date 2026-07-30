'''
Multi-agent round-table refinement for QuizGen-RAG.

Provenance: this module is a port of the multi-agent round-table review
originally built for the MedEd branch (generate_quiz_questions_MedEd.py) to
revise existing quiz questions using student grade-distribution feedback
reports collected after a quiz was administered. Four persona agents
(Pedagogical Expert, Content Accuracy, Feedback Integration, Innovative
Perspective) each proposed an improvement over several discussion rounds,
informed by course-material context retrieved via RAG, and a synthesizer
agent combined their suggestions into a final result.

Here the same machinery is generalized to refine a freshly generated
question/answer pair at generation time, before it is ever shown to a
student. The "feedback report" slot from the original code is generalized
to a "criteria" string: at generation time this carries the target Bloom's
Taxonomy level and the grounding facts the question was built from, rather
than post-hoc grade-distribution feedback. The synthesizer's output is
also changed from a free-form list of "5 recommended pairs" to exactly one
improved question/answer pair in the same strict JSON schema the rest of
the pipeline uses, so it can be parsed and plugged back into the pipeline.

This module has no llama-index imports and no module-level side effects:
callers inject a chat_fn (and, for course-context retrieval, a query_fn)
so this module stays unit-testable offline and importable without any
heavy dependency installed.
'''

import json
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# ChatFn takes (system_prompt, user_prompt) and returns the assistant's
# response text. The caller is responsible for supplying an adapter around
# whatever chat client it uses (e.g. a thin wrapper around llm.chat()).
ChatFn = Callable[[str, str], str]

# The four round-table persona system prompts, ported verbatim from the
# MedEd branch (ASCII-cleaned: curly quotes and em/en dashes replaced with
# their plain-ASCII equivalents).
AGENT_ROLES: dict[str, str] = {
    "Pedagogical Expert": (
        "You are an expert in Bloom's Taxonomy and foundational learning objectives. "
        "Your deep understanding of hierarchical cognitive skills--from 'Knowledge' to 'Creating'--"
        "allows you to evaluate and align questions with the intended educational outcomes. "
        "Your role is to ensure that the question-answer pair challenges students appropriately, "
        "is clear, and adheres to proven pedagogical frameworks."
    ),
    "Content Accuracy": (
        "You are a medical science expert with knowledge equivalent to a first-year doctor's foundations course. "
        "You rigorously verify that all content is factually correct, evidence-based, and aligned with current scientific standards. "
        "Your insights ensure the question-answer pair reflects the highest level of accuracy and reliability."
    ),
    "Feedback Integration": (
        "You specialize in rubric building and assessment strategies, with a strong foundation in rewarding effort and encouraging growth. "
        "Drawing on Vygotsky's ZPD (1978), Dweck (2006), Black & Wiliam (1998), Gay (2000), and OECD (2013), "
        "you focus on semantic clarity and inclusive grading practices. "
        "Your role is to integrate instructor feedback to refine the question-answer pair and enhance student learning."
    ),
    "Innovative Perspective": (
        "You bring a forward-thinking, education-focused outlook that challenges conventional methods. "
        "Your role is to propose creative strategies, leverage emerging technologies, and introduce interdisciplinary insights "
        "to invigorate the question-answer pair. By encouraging critical thinking and curiosity, you help transform standard assessments "
        "into dynamic, engaging learning experiences."
    ),
}


def fetch_course_context(query_fn: Callable[[str], str], qa: dict) -> str:
    '''
    Retrieve course-material context relevant to improving a Q&A pair.

    Port of the MedEd branch's call_course_content_agent, generalized to
    take a plain query function instead of a llama-index query engine
    object, so this module never has to import llama-index.

    :param query_fn: a callable that takes a query string and returns the
        retrieved/synthesized response text (e.g. a thin adapter around
        `query_engine.query(text)` that returns `str(response)`).
    :param qa: the question/answer pair to retrieve context for; must have
        "question" and "answer" keys.
    :return: the retrieved course context text.
    '''
    query = (
        "Retrieve relevant course material context to help improve the "
        "following short answer question and answer pair:\n"
        f"Question: {qa['question']}\n"
        f"Answer: {qa['answer']}\n"
        "Return a summary of the key concepts or facts that can be used to "
        "refine this Q&A."
    )
    return query_fn(query)


def run_round(
    qa: dict,
    criteria: str,
    context_discussion: str,
    course_context: str,
    chat_fn: ChatFn,
) -> dict[str, str]:
    '''
    Run one round of round-table discussion, collecting one suggestion
    from each persona agent.

    Port of the MedEd branch's
    run_round_with_course_content_and_indepth_agents (the dead
    near-duplicate run_round_with_course_content, which used a single
    generic system prompt for every persona and was never actually called,
    is not ported).

    :param qa: the question/answer pair under review; must have "question"
        and "answer" keys.
    :param criteria: the criteria this question/answer pair should be
        judged against. At generation time this carries the target
        Bloom's Taxonomy level and the grounding facts; in the original
        MedEd use this carried a student grade-distribution feedback
        report.
    :param context_discussion: prior rounds' discussion summary text, or
        an empty string on the first round.
    :param course_context: course-material context retrieved by
        fetch_course_context, included verbatim in every persona's prompt.
    :param chat_fn: the chat adapter used to query each persona agent,
        called once per persona with (role system prompt, role user
        prompt).
    :return: a dict mapping each persona name (from AGENT_ROLES) to its
        suggestion text.
    '''
    suggestions: dict[str, str] = {}
    for role, role_system_prompt in AGENT_ROLES.items():
        previous_discussion = f"Previous discussion: {context_discussion}" if context_discussion else ""
        prompt = f"""You are the {role} Agent.
You are tasked with reviewing the following short answer question and answer pair:
Question: {qa['question']}
Answer: {qa['answer']}

The criteria this question should be judged against (its target Bloom's
level and the grounding facts it should be based on) is:
"{criteria}"

The following course context was retrieved from the course materials:
{course_context}

{previous_discussion}

Based on your role, please provide a detailed suggestion for improving this question-answer pair.
Return your suggestion as plain text.
"""
        suggestions[role] = chat_fn(role_system_prompt, prompt).strip()
    return suggestions


def _default_parse(raw: str) -> dict:
    '''
    Minimal fallback parser used by refine_qa when the caller does not
    supply its own parse_fn: strict json.loads with no fence-stripping or
    brace-matching fallback.

    :param raw: the raw synthesizer output text.
    :return: the parsed dict.
    :raises ValueError: if raw is not valid JSON.
    '''
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse synthesizer output as JSON: {raw[:200]!r}"
        ) from exc


def refine_qa(
    qa: dict,
    criteria: str,
    course_context: str,
    chat_fn: ChatFn,
    rounds: int = 1,
    parse_fn: Callable[[str], dict] | None = None,
) -> dict:
    '''
    Run the full round-table refinement of a question/answer pair: N
    rounds of persona discussion (each round sees a growing discussion
    history), then a synthesizer call that produces one improved
    question/answer pair.

    Port of the MedEd branch's synthesize_suggestions plus its
    orchestration in generate_edit_suggestions. CHANGED from the original:
    the synthesizer here is asked for exactly ONE improved question-answer
    pair in the strict `{"question": ..., "answer": ...}` JSON schema
    (instead of a free-form list of "5 recommended pairs"), so the result
    can be parsed and plugged straight back into the generation pipeline.

    :param qa: the draft question/answer pair to refine; must have
        "question" and "answer" keys.
    :param criteria: the criteria this question/answer pair should be
        judged against (see run_round).
    :param course_context: course-material context, typically the result
        of fetch_course_context, included in every persona's prompt.
    :param chat_fn: the chat adapter used for every persona and
        synthesizer call.
    :param rounds: the number of discussion rounds to run before
        synthesis. Each round's suggestions are appended to a shared
        discussion history string that is included in the next round's
        prompts.
    :param parse_fn: callable used to parse the synthesizer's raw output
        into a dict. If None, a minimal strict-JSON parser is used. The
        caller typically passes generate_quiz._parse_qa_response here, so
        the same fence-stripping/brace-matching tolerance is applied to
        the synthesizer's output as to the original draft.
    :return: the parsed dict returned by parse_fn, expected to contain
        "question" and "answer" keys.
    :raises ValueError: if parse_fn (or the default parser) fails to
        parse the synthesizer's output. Callers should catch this and
        fall back to the original draft qa.
    '''
    if parse_fn is None:
        parse_fn = _default_parse

    discussion_history = ""
    for round_number in range(1, rounds + 1):
        suggestions = run_round(qa, criteria, discussion_history, course_context, chat_fn)
        round_summary = (
            f"Round {round_number} suggestions:\n"
            + "\n".join(f"{role}: {suggestion}" for role, suggestion in suggestions.items())
            + "\n"
        )
        discussion_history += round_summary + "\n"
        logger.debug("round_table round %d suggestions: %s", round_number, suggestions)

    final_prompt = f"""You are a Question Writing Agent tasked with implementing the following
suggestions to improve a short answer question and answer pair.

Question: {qa['question']}
Answer: {qa['answer']}

Criteria this question should be judged against (its target Bloom's level
and the grounding facts it should be based on): "{criteria}"

Below are the suggestions from multiple discussion rounds:
{discussion_history}

Please provide ONE improved question-answer pair that incorporates these
suggestions. Format your output as a JSON:
{{
  "question": "...",
  "answer": "..."
}}
No extra text.
"""
    raw = chat_fn(
        "You are an expert synthesizer agent for educational content review.",
        final_prompt,
    )
    return parse_fn(raw.strip())

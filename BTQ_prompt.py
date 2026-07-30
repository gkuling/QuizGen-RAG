'''
Grounded question-prompt construction for QuizGen-RAG.

This module builds the two prompts used by the quiz generation pipeline:

1. A facts-extraction RAG prompt (build_question_prompt), which asks the
   query engine to retrieve and synthesize the top facts relevant to a
   learning objective and reference, then assembles those facts, together
   with the reference, learning objective, and Bloom's level, into the
   final prompt handed to the chat LLM that authors the question/answer
   pair.
2. The system prompt for that chat LLM (build_system_prompt), which used to
   live inline in generate_quiz.py.

Unlike the original version of this module, the retrieved source_nodes are
no longer discarded: build_question_prompt returns a GroundedPrompt that
carries the extracted facts and the deduplicated list of source file names
the facts were grounded in, so a saved quiz can later be checked for
groundedness.

Importing this module has no side effects: it does not construct any
client or index, and does not touch the filesystem or environment.
'''

from dataclasses import dataclass
from typing import Any

from blooms import BLOOM_INSTRUCTIONS, bloom_levels_block, normalize_level


@dataclass(frozen=True)
class GroundedPrompt:
    '''
    A fully assembled question-generation prompt, together with the
    grounding evidence it was built from.

    :ivar prompt: the final prompt to send to the chat LLM, containing the
        reference, learning objective, Bloom's level, level-specific
        instruction, and the extracted facts.
    :ivar facts: the raw facts text extracted from the RAG query engine.
    :ivar source_files: the deduplicated, order-preserving tuple of source
        file names (from retrieved node metadata) that the facts were
        grounded in. Empty if the query engine did not expose source nodes
        or none carried a file name.
    '''

    prompt: str
    facts: str
    source_files: tuple[str, ...]


def _extract_source_files(response: Any) -> tuple[str, ...]:
    '''
    Extract deduplicated, order-preserving source file names from a
    llama-index query response.

    :param response: the response object returned by a query engine's
        query() call. Expected to optionally expose a source_nodes
        attribute (a list of nodes with a .node.metadata mapping).
    :return: a tuple of unique file names, in first-seen order. Nodes
        without a "file_name" metadata entry are skipped, and the whole
        response is treated as having no sources if it lacks a
        source_nodes attribute.
    '''
    source_nodes = getattr(response, "source_nodes", None)
    if not source_nodes:
        return ()

    seen: set[str] = set()
    ordered: list[str] = []
    for scored_node in source_nodes:
        node = getattr(scored_node, "node", None)
        metadata = getattr(node, "metadata", None) if node is not None else None
        file_name = metadata.get("file_name") if isinstance(metadata, dict) else None
        if file_name is None or file_name in seen:
            continue
        seen.add(file_name)
        ordered.append(file_name)
    return tuple(ordered)


def build_question_prompt(
    objective: str, reference: str, level: str, query_engine: Any
) -> GroundedPrompt:
    '''
    Build the grounded prompt used to author one quiz question.

    First queries the given query engine to extract the facts most
    relevant to the learning objective and reference, then assembles those
    facts together with the reference, learning objective, Bloom's level,
    and level-specific instruction into the final prompt for the
    question-authoring chat call.

    :param objective: the learning objective / concept the question should
        target.
    :param reference: the citation or title of the assigned reading the
        question should be grounded in.
    :param level: the Bloom's Taxonomy level the question should target;
        must be one of blooms.BLOOM_LEVELS (case/whitespace-tolerant).
    :param query_engine: a llama-index query engine (or any object
        exposing a compatible .query(str) method returning a response with
        an optional source_nodes attribute) used to retrieve the grounding
        facts.
    :return: the assembled GroundedPrompt.
    :raises ValueError: if level does not match a known Bloom's level.
    '''
    canonical_level = normalize_level(level)
    if canonical_level is None:
        raise ValueError(f"Invalid Bloom's level: {level!r}")

    # Step 1: query the RAG model to extract the key facts.
    rag_prompt = (
        f"Reference:\n{reference}\n\n"
        f"Learning Objective:\n{objective}\n\n"
        f"Please return the top 10 facts directly relevant to the learning "
        f"objective above. Provide them as concise bullet points."
    )
    response = query_engine.query(rag_prompt)
    facts = str(response)
    source_files = _extract_source_files(response)

    # Step 2: retrieve the specialized instruction text for this level.
    instruction_text = BLOOM_INSTRUCTIONS[canonical_level].format(topic=objective)

    prompt = f"""Reference: {reference}

    Learning Objective: {objective}

    Bloom's Taxonomy Level: {canonical_level}

    {instruction_text}

    Facts:
    {facts}
    """

    return GroundedPrompt(prompt=prompt, facts=facts, source_files=source_files)


def build_system_prompt() -> str:
    '''
    Build the system prompt for the chat LLM that authors quiz questions.

    :return: the system prompt text, including the numbered Bloom's
        Taxonomy levels block rendered by blooms.bloom_levels_block().
    '''
    return f"""You are an expert educator creating short-answer quiz questions for a
college-level course. These questions are formative assessments, checking
students' current understanding to guide further learning, not a final,
summative evaluation.

You are currently working on a course on Artificial Intelligence in
Medicine.

Each question should be answerable in around 200 words or fewer.

Use Bloom's Taxonomy to vary cognitive complexity:
{bloom_levels_block()}

You will receive:
- A *reference* (the learning material)
- A *learning objective*
- A *Bloom's level*
- Key *facts* from the reference

Your task:
1. Generate exactly ONE question aligned with the specified Bloom's level.
2. Provide a concise, accurate answer (under 200 words).
3. Base the Q&A on the provided facts.
4. Format your output as a JSON:
{{
  "question": "...",
  "answer": "..."
}}
No extra text.
"""

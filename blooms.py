'''
Shared Bloom's Taxonomy constants and helpers for QuizGen-RAG.

This module centralizes the Bloom's Taxonomy level names, their short
parenthetical descriptions (used in the system prompt), and the detailed
per-level instruction templates (used when prompting the LLM to write a
question). Centralizing these here kills the 3x duplication that used to
exist across BTQ_prompt.py, generate_quiz.py, and course_time_table.py.

This module is stdlib-only: it imports nothing third-party, so it can be
used anywhere in the project (including tests and evaluate_quiz.py)
without pulling in llama-index or any other heavy dependency.
'''

# The six Bloom's Taxonomy levels used throughout the quiz generation
# pipeline, in increasing order of cognitive complexity.
BLOOM_LEVELS: tuple[str, ...] = (
    "Knowledge",
    "Understanding",
    "Applying",
    "Analyzing",
    "Evaluating",
    "Creating",
)

# Short parenthetical description of each level, as shown in the system
# prompt, e.g. "Knowledge (Recall, identify)".
BLOOM_SHORT_DESCRIPTIONS: dict[str, str] = {
    "Knowledge": "Recall, identify",
    "Understanding": "Explain, summarize",
    "Applying": "Use concepts or methods in real contexts",
    "Analyzing": "Compare, contrast",
    "Evaluating": "Critique, judge",
    "Creating": "Design, propose",
}

# Specialized instructions for generating questions at each Bloom's level,
# based on the learning objective topic supplied by the caller. Each
# template contains a '{topic}' placeholder that must be filled in with the
# learning objective text (via str.format) before use. Moved verbatim from
# the old BTQ_prompt.py bloom_instructions dict.
BLOOM_INSTRUCTIONS: dict[str, str] = {
    "Knowledge": (
        "Generate a question that requires students to *recall, list, or identify* "
        "specific facts or definitions from the learning objective '{topic}'."
    ),
    "Understanding": (
        "Generate a question that asks students to *explain, interpret, or summarize* "
        "the key ideas from the learning objective '{topic}'."
    ),
    "Applying": (
        "Generate a question that requires students to *apply or demonstrate* knowledge "
        "from the learning objective '{topic}' in a clinical scenario."
    ),
    "Analyzing": (
        "Generate a question that asks students to *compare, contrast, or break down* "
        "the relationships between ideas in the learning objective '{topic}'."
    ),
    "Evaluating": (
        "Generate a question that asks students to *critique, judge, or justify* "
        "their stance based on the information in the learning objective '{topic}'."
    ),
    "Creating": (
        "Generate a question that asks students to *design, propose, or construct* "
        "a new idea or solution using knowledge from the learning objective "
        "'{topic}' in a medical scenario."
    ),
}


def bloom_levels_block() -> str:
    '''
    Render the six Bloom's Taxonomy levels as a numbered list suitable for
    embedding directly in an LLM system prompt.

    Each line looks like "1. Knowledge (Recall, identify)".

    :return: a newline-joined numbered list covering every level in
        BLOOM_LEVELS, in order.
    '''
    lines = [
        f"{index}. {level} ({BLOOM_SHORT_DESCRIPTIONS[level]})"
        for index, level in enumerate(BLOOM_LEVELS, start=1)
    ]
    return "\n".join(lines)


def normalize_level(raw: str) -> str | None:
    '''
    Match a raw Bloom's level name against the canonical BLOOM_LEVELS
    names, tolerating differences in case and surrounding whitespace.

    :param raw: the raw level string to normalize, e.g. "  knowledge ".
    :return: the canonical level name (as it appears in BLOOM_LEVELS) if a
        case/whitespace-tolerant match is found, otherwise None.
    '''
    if not isinstance(raw, str):
        return None
    candidate = raw.strip().lower()
    for level in BLOOM_LEVELS:
        if level.lower() == candidate:
            return level
    return None

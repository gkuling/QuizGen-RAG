'''
Tests for round_table.py.

These tests use a fake ChatFn that records every (system_prompt,
user_prompt) call it receives and returns canned text, so the whole
round-table discussion/synthesis flow can be exercised with no network
access and no llama-index import.
'''

from collections.abc import Callable

import pytest

from round_table import AGENT_ROLES, fetch_course_context, refine_qa, run_round

SAMPLE_QA: dict = {"question": "What is X?", "answer": "X is Y."}


class RecordingChatFn:
    '''
    A fake round_table.ChatFn that records every call it receives and
    delegates the response to a caller-supplied function.
    '''

    def __init__(self, response_fn: Callable[[str, str], str]) -> None:
        '''
        :param response_fn: called with (system_prompt, user_prompt) for
            every invocation; its return value is returned as the
            "assistant" response.
        '''
        self.calls: list[tuple[str, str]] = []
        self._response_fn = response_fn

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        '''Record the call and return the canned response for it.'''
        self.calls.append((system_prompt, user_prompt))
        return self._response_fn(system_prompt, user_prompt)


def _role_tag(role: str) -> str:
    '''Build a distinguishable, greppable tag for a persona's canned suggestion.'''
    return f"SUGGESTION-FROM-{role.upper().replace(' ', '_')}"


def _persona_response_fn(system_prompt: str, user_prompt: str) -> str:
    '''
    Canned response function: returns a role-specific, distinguishable
    suggestion when called with one of the AGENT_ROLES system prompts,
    and a generic synthesizer response otherwise.
    '''
    for role, role_system_prompt in AGENT_ROLES.items():
        if system_prompt == role_system_prompt:
            return _role_tag(role)
    return '{"question": "synthesized question", "answer": "synthesized answer"}'


def _qa_appears_in(qa: dict, text: str) -> bool:
    '''Helper: True if both the question and answer text appear in text.'''
    return qa["question"] in text and qa["answer"] in text


def test_run_round_produces_one_suggestion_per_persona_with_right_system_prompts() -> None:
    '''run_round should call chat_fn once per AGENT_ROLES entry, using that
    role's own system prompt, and return one suggestion per role.'''
    chat_fn = RecordingChatFn(_persona_response_fn)

    suggestions = run_round(SAMPLE_QA, "criteria text", "", "course context", chat_fn)

    assert set(suggestions.keys()) == set(AGENT_ROLES.keys())
    assert len(chat_fn.calls) == len(AGENT_ROLES)

    called_system_prompts = {system for system, _ in chat_fn.calls}
    assert called_system_prompts == set(AGENT_ROLES.values())

    for role, role_system_prompt in AGENT_ROLES.items():
        matching_calls = [
            (system, user) for system, user in chat_fn.calls if system == role_system_prompt
        ]
        assert len(matching_calls) == 1
        _, user_prompt = matching_calls[0]
        assert _qa_appears_in(SAMPLE_QA, user_prompt)
        assert suggestions[role] == _role_tag(role)


def test_refine_qa_rounds_accumulate_round_one_suggestions_into_round_two_prompts() -> None:
    '''With rounds=2, every round-2 persona prompt should include each
    persona's round-1 suggestion via the growing discussion history.'''
    chat_fn = RecordingChatFn(_persona_response_fn)

    def parse_fn(raw: str) -> dict:
        return {"question": "final question", "answer": "final answer"}

    refine_qa(SAMPLE_QA, "criteria", "course context", chat_fn, rounds=2, parse_fn=parse_fn)

    num_roles = len(AGENT_ROLES)
    # num_roles calls for round 1, num_roles for round 2, then 1 synthesizer call.
    assert len(chat_fn.calls) == 2 * num_roles + 1

    round_two_calls = chat_fn.calls[num_roles : 2 * num_roles]
    assert len(round_two_calls) == num_roles
    for role in AGENT_ROLES:
        tag = _role_tag(role)
        for _, user_prompt in round_two_calls:
            assert tag in user_prompt


def test_synthesizer_prompt_contains_all_role_suggestions() -> None:
    '''The final synthesizer call's prompt should contain every persona's
    suggestion text from the accumulated discussion history.'''
    chat_fn = RecordingChatFn(_persona_response_fn)

    def parse_fn(raw: str) -> dict:
        return {"question": "final question", "answer": "final answer"}

    refine_qa(SAMPLE_QA, "criteria", "course context", chat_fn, rounds=1, parse_fn=parse_fn)

    synthesizer_system_prompt, synthesizer_user_prompt = chat_fn.calls[-1]
    assert synthesizer_system_prompt not in AGENT_ROLES.values()
    for role in AGENT_ROLES:
        assert _role_tag(role) in synthesizer_user_prompt


def test_refine_qa_returns_the_parsed_dict_via_parse_fn() -> None:
    '''refine_qa should return exactly what parse_fn returns for the
    synthesizer's raw output.'''
    chat_fn = RecordingChatFn(lambda system, user: "raw synthesizer output")
    parsed_result = {"question": "Q", "answer": "A"}
    seen_raw: list[str] = []

    def parse_fn(raw: str) -> dict:
        seen_raw.append(raw)
        return parsed_result

    result = refine_qa(SAMPLE_QA, "criteria", "context", chat_fn, rounds=1, parse_fn=parse_fn)

    assert result is parsed_result
    assert seen_raw == ["raw synthesizer output"]


def test_refine_qa_propagates_parse_fn_value_error() -> None:
    '''If parse_fn raises ValueError, refine_qa must let it propagate
    (rather than swallowing it), so the caller can fall back to the draft.'''
    chat_fn = RecordingChatFn(lambda system, user: "unparseable output")

    def failing_parse_fn(raw: str) -> dict:
        raise ValueError("cannot parse synthesizer output")

    with pytest.raises(ValueError):
        refine_qa(SAMPLE_QA, "criteria", "context", chat_fn, rounds=1, parse_fn=failing_parse_fn)


def test_fetch_course_context_queries_with_qa_details_and_returns_query_fn_result() -> None:
    '''fetch_course_context should build a query mentioning the question
    and answer, and return exactly what query_fn returns for it.'''
    captured_queries: list[str] = []

    def query_fn(query_text: str) -> str:
        captured_queries.append(query_text)
        return "the retrieved course context"

    result = fetch_course_context(query_fn, SAMPLE_QA)

    assert result == "the retrieved course context"
    assert len(captured_queries) == 1
    assert SAMPLE_QA["question"] in captured_queries[0]
    assert SAMPLE_QA["answer"] in captured_queries[0]

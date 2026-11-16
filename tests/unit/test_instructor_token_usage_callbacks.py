"""Instructor prompts report measured token usage once per actual provider call."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from ragas.cache import CacheInterface
from ragas.cost import (
    CostCallbackHandler,
    TokenUsage,
    get_token_usage_for_anthropic,
    get_token_usage_for_openai,
)
from ragas.llms.base import InstructorLLM
from ragas.prompt.pydantic_prompt import PydanticPrompt


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    result: str


class ExamplePrompt(PydanticPrompt[Question, Answer]):
    instruction = "Answer the question"
    input_model = Question
    output_model = Answer


def _answer_with_usage(
    *,
    input_tokens: int = 17,
    output_tokens: int = 9,
    provider: str = "openai",
) -> Answer:
    answer = Answer(result="Correct")
    if provider == "anthropic":
        usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
        model = "claude-3"
    else:
        usage = SimpleNamespace(
            prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
        model = "gpt-5-mini"
    object.__setattr__(
        answer, "_raw_response", SimpleNamespace(usage=usage, model=model)
    )
    return answer


def _client(answer: Answer, async_model: bool = True):
    create = (
        AsyncMock(return_value=answer)
        if async_model
        else MagicMock(return_value=answer)
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, create


@pytest.mark.asyncio
@pytest.mark.parametrize("async_model", [True, False])
async def test_instructor_reports_actual_usage(async_model):
    client, create = _client(_answer_with_usage(), async_model)
    llm = InstructorLLM(client=client, provider="openai", model="gpt-5-mini")
    assert llm.is_async is async_model

    callback = CostCallbackHandler(get_token_usage_for_openai)
    response = await ExamplePrompt().generate(
        llm=llm,
        data=Question(text="Where is Paris?"),
        callbacks=[callback],
    )

    assert response.result == "Correct"
    assert callback.usage_data == [
        TokenUsage(input_tokens=17, output_tokens=9, model="gpt-5-mini")
    ]
    if async_model:
        create.assert_awaited_once()
    else:
        create.assert_called_once()


@pytest.mark.asyncio
async def test_instructor_anthropic_usage_is_reported():
    client, create = _client(_answer_with_usage(provider="anthropic"))
    llm = InstructorLLM(client=client, provider="anthropic", model="claude-3")
    callback = CostCallbackHandler(get_token_usage_for_anthropic)

    result = await ExamplePrompt().generate(
        llm=llm,
        data=Question(text="Where is Paris?"),
        callbacks=[callback],
    )

    assert result.result == "Correct"
    assert callback.usage_data == [
        TokenUsage(input_tokens=17, output_tokens=9, model="claude-3")
    ]
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_instructor_without_raw_usage_does_not_invent_cost():
    client, create = _client(Answer(result="Correct"))
    llm = InstructorLLM(client=client, provider="openai", model="gpt-5-mini")
    callback = CostCallbackHandler(get_token_usage_for_openai)

    result = await ExamplePrompt().generate(
        llm=llm,
        data=Question(text="Where is Paris?"),
        callbacks=[callback],
    )

    assert result.result == "Correct"
    assert callback.usage_data == []
    create.assert_awaited_once()


class Cache(CacheInterface):
    def __init__(self):
        self.data = {}

    def has_key(self, key):
        return key in self.data

    def get(self, key):
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value


@pytest.mark.asyncio
async def test_cache_hits_do_not_charge_tokens_twice():
    client, create = _client(_answer_with_usage(input_tokens=10, output_tokens=5))
    llm = InstructorLLM(
        client=client, provider="openai", model="gpt-5-mini", cache=Cache()
    )
    callback = CostCallbackHandler(get_token_usage_for_openai)
    prompt = ExamplePrompt()

    for _ in range(2):
        result = await prompt.generate(
            llm=llm,
            data=Question(text="Where is Paris?"),
            callbacks=[callback],
        )
        assert result.result == "Correct"

    create.assert_awaited_once()
    assert callback.usage_data == [
        TokenUsage(input_tokens=10, output_tokens=5, model="gpt-5-mini")
    ]

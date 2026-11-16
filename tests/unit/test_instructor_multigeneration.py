"""A request for N structured outputs must produce N independently cached calls."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from ragas.cache import CacheInterface
from ragas.cost import CostCallbackHandler, TokenUsage, get_token_usage_for_openai
from ragas.llms.base import InstructorLLM
from ragas.prompt.pydantic_prompt import PydanticPrompt


class Input(BaseModel):
    question: str


class Output(BaseModel):
    answer: str


class QuestionPrompt(PydanticPrompt[Input, Output]):
    instruction = "Generate an answer."
    input_model = Input
    output_model = Output


class Cache(CacheInterface):
    def __init__(self):
        self.data = {}

    def has_key(self, key):
        return key in self.data

    def get(self, key):
        return self.data[key]

    def set(self, key, value):
        self.data[key] = value


def output(number: int) -> Output:
    result = Output(answer=f"variant-{number}")
    object.__setattr__(
        result,
        "_raw_response",
        SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            model="gpt-5-mini",
        ),
    )
    return result


@pytest.mark.asyncio
async def test_instructor_generates_requested_number_of_outputs():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=[output(1), output(2), output(3)])
            )
        )
    )
    llm = InstructorLLM(client=client, provider="openai", model="gpt-5-mini")

    results = await QuestionPrompt().generate_multiple(
        llm=llm, data=Input(question="Where is Paris?"), n=3
    )

    assert [result.answer for result in results] == [
        "variant-1",
        "variant-2",
        "variant-3",
    ]
    assert client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_instructor_caches_each_variant_independently_without_double_cost():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(side_effect=[output(1), output(2), output(3)])
            )
        )
    )
    llm = InstructorLLM(
        client=client, provider="openai", model="gpt-5-mini", cache=Cache()
    )
    callback = CostCallbackHandler(get_token_usage_for_openai)
    prompt = QuestionPrompt()

    first = await prompt.generate_multiple(
        llm=llm, data=Input(question="Where is Paris?"), n=3, callbacks=[callback]
    )
    second = await prompt.generate_multiple(
        llm=llm, data=Input(question="Where is Paris?"), n=3, callbacks=[callback]
    )

    assert [answer.answer for answer in first] == [
        "variant-1",
        "variant-2",
        "variant-3",
    ]
    assert [answer.answer for answer in second] == [
        "variant-1",
        "variant-2",
        "variant-3",
    ]
    assert client.chat.completions.create.await_count == 3
    assert callback.usage_data == [
        TokenUsage(input_tokens=10, output_tokens=5, model="gpt-5-mini")
        for _ in range(3)
    ]


@pytest.mark.asyncio
async def test_sync_instructor_generates_multiple_outputs():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(side_effect=[output(1), output(2), output(3)])
            )
        )
    )
    llm = InstructorLLM(client=client, provider="openai", model="gpt-5-mini")

    results = await QuestionPrompt().generate_multiple(
        llm=llm, data=Input(question="Where is Paris?"), n=3
    )

    assert {answer.answer for answer in results} == {
        "variant-1",
        "variant-2",
        "variant-3",
    }
    assert client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_sync_instructor_reuses_distinct_cached_variants():
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(side_effect=[output(1), output(2), output(3)])
            )
        )
    )
    llm = InstructorLLM(
        client=client, provider="openai", model="gpt-5-mini", cache=Cache()
    )
    prompt = QuestionPrompt()

    first = await prompt.generate_multiple(
        llm=llm, data=Input(question="Where is Paris?"), n=3
    )
    second = await prompt.generate_multiple(
        llm=llm, data=Input(question="Where is Paris?"), n=3
    )

    assert {item.answer for item in first} == {"variant-1", "variant-2", "variant-3"}
    assert [item.answer for item in first] == [item.answer for item in second]
    assert client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_custom_instructor_without_new_kwargs_remains_compatible():
    from ragas.llms.base import InstructorBaseRagasLLM

    class CustomInstructor(InstructorBaseRagasLLM):
        def __init__(self):
            self.is_async = True
            self.calls = 0

        async def agenerate(self, prompt, response_model):
            self.calls += 1
            return Output(answer=f"custom-{self.calls}")

        def generate(self, prompt, response_model):
            raise AssertionError("Async implementation should be used")

    llm = CustomInstructor()
    results = await QuestionPrompt().generate_multiple(
        llm=llm, data=Input(question="Where is Paris?"), n=2
    )

    assert [item.answer for item in results] == ["custom-1", "custom-2"]
    assert llm.calls == 2

"""Evaluation traces must keep every generation and repeated prompt call."""

from ragas.callbacks import ChainRun, parse_run_traces


def _trace(run_id, parent_id, name, children=None, input_data=None, output=None):
    return ChainRun(
        run_id=run_id,
        parent_run_id=parent_id,
        name=name,
        inputs={"data": input_data} if input_data is not None else {},
        outputs={"output": output} if output is not None else {},
        metadata={},
        children=children or [],
    )


def test_repeated_prompt_names_preserve_all_invocations_and_order():
    runs = {
        "root": _trace("root", None, "evaluation", children=["row"]),
        "row": _trace("row", "root", "row", children=["metric"]),
        "metric": _trace(
            "metric", "row", "factual_correctness", children=["q1", "q2", "judge"]
        ),
        "q1": _trace(
            "q1",
            "metric",
            "claim_decomposition_prompt",
            input_data={"response": "Predicted"},
            output=[{"claims": ["Predicted claim"]}],
        ),
        "q2": _trace(
            "q2",
            "metric",
            "claim_decomposition_prompt",
            input_data={"response": "Reference"},
            output=[{"claims": ["Reference claim"]}],
        ),
        "judge": _trace(
            "judge",
            "metric",
            "nli_prompt",
            input_data={"statements": ["Predicted claim"]},
            output=[{"verdict": 1}],
        ),
    }

    result = parse_run_traces(runs)
    prompts = result[0]["factual_correctness"]

    assert prompts["claim_decomposition_prompt"] == [
        {
            "input": {"response": "Predicted"},
            "output": {"claims": ["Predicted claim"]},
        },
        {
            "input": {"response": "Reference"},
            "output": {"claims": ["Reference claim"]},
        },
    ]
    # Preserve the documented singleton access shape for existing consumers.
    assert prompts["nli_prompt"] == {
        "input": {"statements": ["Predicted claim"]},
        "output": {"verdict": 1},
    }


def test_multiple_outputs_from_one_prompt_are_not_discarded():
    runs = {
        "root": _trace("root", None, "evaluation", children=["row"]),
        "row": _trace("row", "root", "row", children=["metric"]),
        "metric": _trace("metric", "row", "answer_relevancy", children=["prompt"]),
        "prompt": _trace(
            "prompt",
            "metric",
            "question_prompt",
            input_data={"answer": "Paris"},
            output=[
                {"question": "Where is Paris?"},
                {"question": "Which country contains Paris?"},
            ],
        ),
    }

    prompts = parse_run_traces(runs)[0]["answer_relevancy"]
    assert prompts["question_prompt"]["output"] == [
        {"question": "Where is Paris?"},
        {"question": "Which country contains Paris?"},
    ]


def test_empty_prompt_output_does_not_crash_trace_parsing():
    runs = {
        "root": _trace("root", None, "evaluation", children=["row"]),
        "row": _trace("row", "root", "row", children=["metric"]),
        "metric": _trace("metric", "row", "faithfulness", children=["prompt"]),
        "prompt": _trace("prompt", "metric", "judge_prompt", output=[]),
    }
    result = parse_run_traces(runs)
    assert result[0]["faithfulness"]["judge_prompt"]["output"] == []

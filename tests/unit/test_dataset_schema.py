import typing as t

import pytest

from ragas.dataset_schema import (
    EvaluationDataset,
    HumanMessage,
    MultiTurnSample,
    PromptAnnotation,
    SampleAnnotation,
    SingleMetricAnnotation,
    SingleTurnSample,
)

samples = [
    SingleTurnSample(user_input="What is X", response="Y"),
    MultiTurnSample(
        user_input=[HumanMessage(content="What is X")],
        reference="Y",
    ),
]


def create_sample_annotation(metric_output):
    return SampleAnnotation(
        metric_input={
            "response": "",
            "reference": "",
            "user_input": "",
        },
        metric_output=metric_output,
        prompts={
            "single_turn_aspect_critic_prompt": PromptAnnotation(
                prompt_input={
                    "response": "",
                    "reference": "",
                    "user_input": "",
                },
                prompt_output={"reason": "", "verdict": 1},
                edited_output=None,
            )
        },
        is_accepted=True,
        target=None,
    )


def test_loader_sample():
    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    test_dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    sample = test_dataset.sample(2)
    assert len(sample) == 2

    sample = test_dataset.sample(2, stratify_key="metric_output")
    assert len(sample) == 2
    assert sum(item["metric_output"] for item in sample) == 1


def test_loader_batch():
    annotated_samples = [create_sample_annotation(1) for _ in range(10)] + [
        create_sample_annotation(0) for _ in range(10)
    ]
    dataset = SingleMetricAnnotation(name="metric", samples=annotated_samples)
    batches = dataset.batch(batch_size=2)
    assert all([len(item) == 2 for item in batches])

    batches = dataset.stratified_batches(batch_size=2, stratify_key="metric_output")
    assert all(sum([item["metric_output"] for item in batch]) == 1 for batch in batches)


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset(eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    hf_dataset = dataset.to_hf_dataset()

    assert dataset.get_sample_type() is type(eval_sample)
    assert len(hf_dataset) == 2
    assert len(dataset) == 2
    assert dataset[0] == eval_sample

    dataset_from_hf = EvaluationDataset.from_hf_dataset(hf_dataset)
    assert dataset_from_hf == dataset


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_save_load_csv(tmpdir, eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # save and load to csv
    csv_path = tmpdir / "csvfile.csv"
    dataset.to_csv(csv_path)


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_save_load_jsonl(tmpdir, eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # save and load to jsonl
    jsonl_path = tmpdir / "jsonlfile.jsonl"
    dataset.to_jsonl(jsonl_path)
    loaded_dataset = EvaluationDataset.from_jsonl(jsonl_path)
    assert loaded_dataset == dataset


@pytest.mark.parametrize("eval_sample", samples)
def test_evaluation_dataset_load_from_hf(eval_sample):
    dataset = EvaluationDataset(samples=[eval_sample, eval_sample])

    # convert to and load from hf dataset
    hf_dataset = dataset.to_hf_dataset()
    loaded_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
    assert loaded_dataset == dataset


def test_single_turn_sample_metadata_roundtrip_hf_and_jsonl(tmpdir):
    sample = SingleTurnSample(
        user_input="Q",
        response="A",
        reference_contexts=["ctx"],
        persona_name="Researcher",
        query_style="FORMAL",
        query_length="SHORT",
    )
    dataset = EvaluationDataset(samples=[sample])

    # HF round-trip
    hf = dataset.to_hf_dataset()
    loaded_hf = EvaluationDataset.from_hf_dataset(hf)
    assert loaded_hf.samples[0].persona_name == "Researcher"
    assert loaded_hf.samples[0].query_style == "FORMAL"
    assert loaded_hf.samples[0].query_length == "SHORT"

    # JSONL round-trip
    jsonl_path = tmpdir / "ds.jsonl"
    dataset.to_jsonl(jsonl_path)
    loaded_jsonl = EvaluationDataset.from_jsonl(jsonl_path)
    assert loaded_jsonl.samples[0].persona_name == "Researcher"
    assert loaded_jsonl.samples[0].query_style == "FORMAL"
    assert loaded_jsonl.samples[0].query_length == "SHORT"


@pytest.mark.parametrize("eval_sample", samples)
def test_single_type_evaluation_dataset(eval_sample):
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    multi_turn_sample = MultiTurnSample(
        user_input=[{"content": "What is X"}],
        response="Y",  # type: ignore (this type error is what we want to test)
    )

    with pytest.raises(ValueError) as exc_info:
        EvaluationDataset(samples=[single_turn_sample, multi_turn_sample])

    error_message = str(exc_info.value)

    assert (
        "Sample at index 1 is of type <class 'ragas.dataset_schema.MultiTurnSample'>"
        in error_message
    )
    assert "expected <class 'ragas.dataset_schema.SingleTurnSample'>" in error_message


def test_base_eval_sample():
    from ragas.dataset_schema import BaseSample

    class FakeSample(BaseSample):
        user_input: str
        response: str
        reference: t.Optional[str] = None

    fake_sample = FakeSample(user_input="What is X", response="Y")
    assert fake_sample.to_dict() == {"user_input": "What is X", "response": "Y"}
    assert fake_sample.get_features() == ["user_input", "response"]


def test_evaluation_dataset_iter():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")

    dataset = EvaluationDataset(samples=[single_turn_sample, single_turn_sample])

    for sample in dataset:
        assert sample == single_turn_sample


def test_evaluation_dataset_type():
    single_turn_sample = SingleTurnSample(user_input="What is X", response="Y")
    multi_turn_sample = MultiTurnSample(
        user_input=[{"content": "What is X"}],
        response="Y",  # type: ignore (this type error is what we want to test)
    )

    dataset = EvaluationDataset(samples=[single_turn_sample])
    assert dataset.get_sample_type() == SingleTurnSample

    dataset = EvaluationDataset(samples=[multi_turn_sample])
    assert dataset.get_sample_type() == MultiTurnSample


def test_multiturn_sample_validate_user_input_invalid_type():
    """Test that MultiTurnSample validation correctly rejects invalid message types."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MultiTurnSample(
            user_input=[
                HumanMessage(content="Hello"),
                "invalid_string",  # This should be rejected by Pydantic
            ]
        )


def test_multiturn_sample_validate_user_input_valid_types():
    """Test that MultiTurnSample validation accepts valid message types."""
    from ragas.messages import AIMessage

    sample = MultiTurnSample(
        user_input=[
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
    )
    assert len(sample.user_input) == 2
    assert isinstance(sample.user_input[0], HumanMessage)
    assert isinstance(sample.user_input[1], AIMessage)


def test_empty_evaluation_dataset_is_inspectable():
    """An empty batch must not crash ordinary dataset inspection/serialization."""
    dataset = EvaluationDataset.from_list([])
    assert dataset.features() == []
    assert dataset.to_list() == []
    assert repr(dataset) == "EvaluationDataset(features=[], len=0)"
    assert dataset.is_multi_turn() is False
    assert dataset.to_pandas().empty
    assert len(dataset.to_hf_dataset()) == 0

    with pytest.raises(ValueError, match="empty dataset"):
        dataset.get_sample_type()


def test_empty_evaluation_result_is_inspectable():
    """Callers can inspect a result with no scored rows."""
    from ragas.dataset_schema import EvaluationResult

    dataset = EvaluationDataset.from_list([])
    result = EvaluationResult(scores=[], dataset=dataset)
    assert repr(result) == "{}"
    assert result.to_pandas().empty


def test_sparse_evaluation_result_preserves_metric_union_and_row_alignment():
    """Missing row/metric combinations must be NaN, not lost or KeyError."""
    import math

    from ragas.dataset_schema import EvaluationResult

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(user_input="first"),
            SingleTurnSample(user_input="second"),
            SingleTurnSample(user_input="third"),
        ]
    )
    sparse_scores = [
        {},
        {"faithfulness": 0.25, "relevancy": 0.5},
        {"faithfulness": 0.75},
    ]
    result = EvaluationResult(scores=sparse_scores, dataset=dataset)

    assert list(result._scores_dict) == ["faithfulness", "relevancy"]
    assert math.isnan(result["faithfulness"][0])
    assert result["faithfulness"][1:] == [0.25, 0.75]
    assert math.isnan(result["relevancy"][0])
    assert result["relevancy"][1] == 0.5
    assert math.isnan(result["relevancy"][2])
    assert result._repr_dict == {"faithfulness": 0.5, "relevancy": 0.5}
    assert "faithfulness" in repr(result)
    rows = result.to_pandas()
    assert rows["user_input"].tolist() == ["first", "second", "third"]
    assert math.isnan(rows["faithfulness"].iloc[0])
    assert rows["faithfulness"].iloc[1:].tolist() == [0.25, 0.75]
    assert math.isnan(rows["relevancy"].iloc[2])
    assert sparse_scores == [
        {},
        {"faithfulness": 0.25, "relevancy": 0.5},
        {"faithfulness": 0.75},
    ]


def test_sparse_evaluation_result_keeps_all_metrics_and_all_nan_means():
    """Order is first-seen across all score dictionaries, including late metrics."""
    import math

    from ragas.dataset_schema import EvaluationResult

    dataset = EvaluationDataset(
        samples=[SingleTurnSample(user_input=str(i)) for i in range(4)]
    )
    result = EvaluationResult(
        scores=[
            {"accuracy": 1.0},
            {},
            {"safety": float("nan"), "accuracy": 0.0},
            {"uncertainty": float("nan"), "safety": 0.75},
        ],
        dataset=dataset,
        binary_columns=["safety"],
    )
    assert list(result._scores_dict) == ["accuracy", "safety", "uncertainty"]
    assert result["accuracy"][0] == 1.0
    assert math.isnan(result["accuracy"][1])
    assert result["accuracy"][2] == 0.0
    assert math.isnan(result["accuracy"][3])
    assert math.isnan(result["safety"][0])
    assert result["safety"][3] == 0.75
    assert math.isnan(result._repr_dict["uncertainty"])
    assert result._repr_dict["accuracy"] == 0.5
    assert result._repr_dict["safety"] == 0.75
    assert list(result.to_pandas()["uncertainty"].isna()) == [True, True, True, True]


def test_evaluation_result_all_empty_score_rows_remains_inspectable():
    """A nonempty dataset with no metric names should have no metric columns."""
    from ragas.dataset_schema import EvaluationResult

    dataset = EvaluationDataset(
        samples=[SingleTurnSample(user_input=str(i)) for i in range(2)]
    )
    result = EvaluationResult(scores=[{}, {}], dataset=dataset)
    assert repr(result) == "{}"
    assert result._scores_dict == {}
    assert result.to_pandas()["user_input"].tolist() == ["0", "1"]


def test_sparse_evaluation_result_randomized_metric_masks():
    """Across changing row masks, pandas and aggregation agree on every score."""
    import math
    import random

    from ragas.dataset_schema import EvaluationResult

    rng = random.Random(7719)
    metric_names = ["faithfulness", "relevancy", "precision", "recall"]
    for _ in range(80):
        row_count = rng.randrange(1, 19)
        rows = [
            {name: rng.random() for name in metric_names if rng.random() < 0.4}
            for _ in range(row_count)
        ]
        dataset = EvaluationDataset(
            samples=[SingleTurnSample(user_input=str(i)) for i in range(row_count)]
        )
        result = EvaluationResult(scores=rows, dataset=dataset)
        frame = result.to_pandas()
        expected_order = list(dict.fromkeys(k for row in rows for k in row))
        assert list(result._scores_dict) == expected_order

        for metric in expected_order:
            series = result[metric]
            assert len(series) == row_count
            for i, row in enumerate(rows):
                if metric in row:
                    assert series[i] == row[metric]
                    assert frame[metric].iloc[i] == row[metric]
                else:
                    assert math.isnan(series[i])
                    assert math.isnan(frame[metric].iloc[i])
            known = [row[metric] for row in rows if metric in row]
            assert math.isclose(result._repr_dict[metric], sum(known) / len(known))

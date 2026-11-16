import numpy as np
import pytest

from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import SemanticSimilarity


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reference_vector", "response_vector", "threshold", "expected"),
    [
        ([0.0, 0.0], [1.0, 0.0], None, 0.0),
        ([1.0, 0.0], [0.0, 0.0], None, 0.0),
        ([0.0, 0.0], [0.0, 0.0], None, 0.0),
        ([0.0, 0.0], [1.0, 0.0], 0.8, 0.0),
        ([1.0, 0.0], [1.0, 0.0], None, 1.0),
        ([1.0, 0.0], [0.6, 0.8], None, 0.6),
        ([1.0, 0.0], [0.6, 0.8], 0.8, 0.0),
    ],
)
async def test_semantic_similarity_zero_norm_embeddings(
    reference_vector, response_vector, threshold, expected
):
    class Embeddings(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return reference_vector if text == "reference" else response_vector

        async def aembed_text(self, text, **kwargs):
            return self.embed_text(text)

    metric = SemanticSimilarity(embeddings=Embeddings(), threshold=threshold)
    with np.errstate(divide="raise", invalid="raise"):
        result = await metric.ascore(reference="reference", response="response")
    assert result.value == pytest.approx(expected)

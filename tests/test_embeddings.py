from modules.embeddings import create_embedding


class FakeEmbeddingResponse:
    class Embedding:
        values = [0.1, 0.2, 0.3]

    embeddings = [Embedding()]


class FakeModels:
    def embed_content(self, model, contents):
        assert model == "gemini-embedding-001"
        assert contents == "test policy clause"

        return FakeEmbeddingResponse()


class FakeClient:
    models = FakeModels()


def test_create_embedding():
    client = FakeClient()

    vector = create_embedding(
        "test policy clause",
        client=client,
    )

    assert vector == [0.1, 0.2, 0.3]
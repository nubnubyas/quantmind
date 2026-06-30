"""Tests for EmbeddingClient."""

import pytest


class TestEmbeddingClient:
    def test_default_dim_is_768(self):
        """默认模型应为 BGE-base (768-dim)。"""
        from src.config.llm_client import EmbeddingClient

        client = EmbeddingClient()
        assert client.dim == 768, f"Expected 768-dim, got {client.dim}"
        assert "bge-base" in client.model.lower(), (
            f"Expected bge-base model, got {client.model}"
        )

    def test_embed_returns_correct_dim(self):
        """embed() 返回的向量维度应与 client.dim 一致。"""
        from src.config.llm_client import EmbeddingClient

        client = EmbeddingClient()
        vecs = client.embed(["test sentence", "another sentence"])
        assert len(vecs) == 2
        assert len(vecs[0]) == client.dim
        assert len(vecs[1]) == client.dim

    def test_embed_returns_floats(self):
        """返回的向量应为 float 列表。"""
        from src.config.llm_client import EmbeddingClient

        client = EmbeddingClient()
        vecs = client.embed(["test"])
        assert all(isinstance(v, float) for v in vecs[0])

    def test_backward_compatible_with_env_override(self, monkeypatch):
        """通过 EMBEDDING_MODEL 环境变量可以切回旧模型。"""
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        monkeypatch.delenv("EMBEDDING_DIM", raising=False)
        from src.config.llm_client import EmbeddingClient

        client = EmbeddingClient()
        assert client.dim == 384
        assert "bge-small" in client.model.lower()

    def test_embed_single_text(self):
        """单文本的 embed 调用正常工作。"""
        from src.config.llm_client import EmbeddingClient

        client = EmbeddingClient()
        vecs = client.embed(["hello world"])
        assert len(vecs) == 1
        assert isinstance(vecs[0], list)

    def test_embed_empty_list(self):
        """空列表应返回空列表。"""
        from src.config.llm_client import EmbeddingClient

        client = EmbeddingClient()
        vecs = client.embed([])
        assert vecs == []

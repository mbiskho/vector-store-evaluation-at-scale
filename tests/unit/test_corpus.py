import numpy as np
import pytest

from vsebench.corpus.generator import CorpusConfig, _extract_text


class TestCorpusGenerator:
    def test_extract_text_from_text_field(self):
        row = {"text": "This is a test document about vector databases.", "id": 1}
        result = _extract_text(row)
        assert result == "This is a test document about vector databases."

    def test_extract_text_from_content_field(self):
        row = {"content": "Another document about embeddings.", "id": 2}
        result = _extract_text(row)
        assert result == "Another document about embeddings."

    def test_extract_text_skips_short(self):
        row = {"text": "Hi", "id": 3}
        result = _extract_text(row)
        assert result == ""

    def test_extract_text_combines_title_and_text(self):
        row = {"title": "Introduction", "text": "This is the body text for the document."}
        result = _extract_text(row)
        assert result == "Introduction\n\nThis is the body text for the document."

    def test_extract_text_returns_empty_for_empty_row(self):
        result = _extract_text({})
        assert result == ""

    def test_corpus_config_defaults(self):
        config = CorpusConfig()
        assert config.dataset_name == "wikipedia"
        assert config.dimension == 384
        assert config.target_vectors == 10_000_000

    def test_corpus_config_custom(self):
        config = CorpusConfig(target_vectors=1000, query_count=100)
        assert config.target_vectors == 1000
        assert config.query_count == 100

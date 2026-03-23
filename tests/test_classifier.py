"""Tests for TensorFlow query classifier."""

import pytest


class TestQueryClassifier:

    def test_simple_query(self):
        from src.query_classifier.classifier import classify_query
        result = classify_query("diabetes trials")
        assert result["classification"] in ["simple", "complex"]
        assert 0 <= result["confidence"] <= 1

    def test_complex_query(self):
        from src.query_classifier.classifier import classify_query
        result = classify_query(
            "58 year old male with type 2 diabetes and CKD stage 3. "
            "Taking metformin and lisinopril. Lives in Maryland."
        )
        assert result["classification"] == "complex"

    def test_short_query_is_simple(self):
        from src.query_classifier.classifier import classify_query
        result = classify_query("cancer trials")
        assert result["classification"] == "simple"

    def test_returns_valid_structure(self):
        from src.query_classifier.classifier import classify_query
        result = classify_query("test query")
        assert "classification" in result
        assert "confidence" in result
        assert "model" in result
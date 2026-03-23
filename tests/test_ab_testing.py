"""Tests for A/B testing framework."""

import pytest
from src.evaluation.ab_test import measure_relevance, measure_diversity


class TestABTesting:

    def test_relevance_perfect_match(self):
        """Perfect results should score 1.0."""
        results = {
            "trials": [
                {"conditions": ["Type 2 Diabetes"], "brief_title": "Diabetes Study"},
                {"conditions": ["Diabetes Mellitus"], "brief_title": "DM Trial"},
            ]
        }
        score = measure_relevance(results, ["diabetes"])
        assert score == 1.0

    def test_relevance_no_match(self):
        """Irrelevant results should score 0.0."""
        results = {
            "trials": [
                {"conditions": ["Asthma"], "brief_title": "Asthma Study"},
            ]
        }
        score = measure_relevance(results, ["diabetes"])
        assert score == 0.0

    def test_relevance_empty_results(self):
        """Empty results should score 0.0."""
        score = measure_relevance({"trials": []}, ["diabetes"])
        assert score == 0.0

    def test_diversity_counts_unique(self):
        """Diversity should count unique conditions."""
        results = {
            "trials": [
                {"conditions": ["Diabetes", "Hypertension"]},
                {"conditions": ["Diabetes", "CKD"]},
            ]
        }
        diversity = measure_diversity(results)
        assert diversity == 3  # diabetes, hypertension, ckd

    def test_diversity_empty(self):
        """Empty results should have 0 diversity."""
        assert measure_diversity({"trials": []}) == 0
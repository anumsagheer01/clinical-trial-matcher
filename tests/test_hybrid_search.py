"""Tests for hybrid search functionality."""

import pytest
from src.search.hybrid_search import hybrid_search


class TestHybridSearch:

    def test_semantic_search_finds_synonyms(self):
        """'kidney failure' should match trials about 'renal' conditions."""
        results = hybrid_search(query_text="kidney failure", page_size=10)
        assert results["total"] > 0

        all_text = ""
        for trial in results["trials"]:
            all_text += " ".join(trial["conditions"]).lower()
            all_text += " " + trial.get("brief_title", "").lower()

        # Should find renal-related trials even though I searched "kidney"
        has_relevant = ("renal" in all_text or "kidney" in all_text)
        assert has_relevant, "Expected kidney/renal-related results"

    def test_hybrid_beats_empty(self):
        """Hybrid search should return more than zero results for medical terms."""
        results = hybrid_search(query_text="heart disease", page_size=5)
        assert results["total"] > 0
        assert len(results["trials"]) > 0

    def test_filters_work_with_hybrid(self):
        """Filters should work alongside hybrid search."""
        results = hybrid_search(
            query_text="cancer",
            sex="Female",
            page_size=10,
        )
        for trial in results["trials"]:
            trial_sex = trial.get("eligibility", {}).get("sex", "All")
            assert trial_sex in ["All", "Female"]

    def test_empty_query_returns_results(self):
        """No query should still return results (match_all)."""
        results = hybrid_search(page_size=5)
        assert results["total"] > 0


class TestAPI:

    def test_health_endpoint(self):
        """Health check should work."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_search_endpoint(self):
        """Search endpoint should return results."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.post("/search", json={
            "query": "diabetes",
            "max_results": 3,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0

    def test_trial_endpoint_invalid_id(self):
        """Invalid trial ID should return 404."""
        from fastapi.testclient import TestClient
        from src.api.main import app

        client = TestClient(app)
        response = client.get("/trial/NCT_FAKE_99999")
        assert response.status_code == 404
"""
Tests for clinical trial search.

- CI/CD pipeline will run these automatically later
- Use pytest

"""

import pytest
from src.search.trial_search import search_trials


class TestTrialSearch:
    """Test suite for trial search functionality."""
    
    def test_basic_text_search_returns_results(self):
        """A search for 'diabetes' should return at least some results."""
        results = search_trials(query_text="diabetes", page_size=5)
        assert results["total"] > 0, "Expected at least 1 result for 'diabetes'"
        assert len(results["trials"]) > 0
        assert len(results["trials"]) <= 5
    
    def test_search_returns_relevant_conditions(self):
        """Results for 'diabetes' should actually be about diabetes."""
        results = search_trials(query_text="type 2 diabetes", page_size=5)
        # At least one result should have 'diabetes' in its conditions
        all_conditions = []
        for trial in results["trials"]:
            all_conditions.extend([c.lower() for c in trial["conditions"]])
        
        has_diabetes = any("diabetes" in c for c in all_conditions)
        assert has_diabetes, "Expected at least one result to have 'diabetes' in conditions"
    
    def test_age_filter_works(self):
        """Searching with age=10 should not return adult-only trials."""
        results = search_trials(
            query_text="cancer",
            min_age=10,
            max_age=10,
            page_size=20,
        )
        
        for trial in results["trials"]:
            min_age = trial.get("eligibility", {}).get("min_age")
            if min_age is not None:
                assert min_age <= 10, (
                    f"Trial {trial['nct_id']} has min_age {min_age} "
                    f"but we searched for age 10"
                )
    
    def test_sex_filter_excludes_wrong_sex(self):
        """Filtering for Male should not return Female-only trials."""
        results = search_trials(
            query_text="cancer",
            sex="Male",
            page_size=20,
        )
        
        for trial in results["trials"]:
            trial_sex = trial.get("eligibility", {}).get("sex", "All")
            assert trial_sex in ["All", "Male"], (
                f"Trial {trial['nct_id']} is for {trial_sex} only, "
                f"but we filtered for Male"
            )
    
    def test_pagination_works(self):
        """Page 1 and page 2 should return different results."""
        page1 = search_trials(query_text="cancer", page_size=5, page=1)
        page2 = search_trials(query_text="cancer", page_size=5, page=2)
        
        page1_ids = {t["nct_id"] for t in page1["trials"]}
        page2_ids = {t["nct_id"] for t in page2["trials"]}
        
        assert page1_ids != page2_ids, "Page 1 and page 2 returned same results"
    
    def test_empty_query_returns_all(self):
        """No filters should return all trials."""
        results = search_trials(page_size=1)
        assert results["total"] > 1000, "Expected many results for empty query"
    
    def test_nonsense_query_returns_few_or_no_results(self):
        """A nonsense query should return few/no results."""
        results = search_trials(query_text="xyzzy12345 qqqqblarg", page_size=5)
        assert results["total"] < 10, "Nonsense query returned too many results"
    
    def test_response_structure(self):
        """Response should have the expected fields."""
        results = search_trials(query_text="diabetes", page_size=1)
        
        assert "total" in results
        assert "trials" in results
        assert "took_ms" in results
        assert "page" in results
        assert "page_size" in results
        
        if results["trials"]:
            trial = results["trials"][0]
            assert "nct_id" in trial
            assert "brief_title" in trial
            assert "conditions" in trial
            assert "eligibility" in trial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
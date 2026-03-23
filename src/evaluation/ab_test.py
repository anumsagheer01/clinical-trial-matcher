"""
A/B Testing Framework for retrieval strategies.

Compares two hybrid search configurations:
- Strategy A: keyword_weight=0.7, vector_weight=0.3 (keyword-heavy)
- Strategy B: keyword_weight=0.3, vector_weight=0.7 (vector-heavy)

Metrics measured:
- Retrieval relevance (do the results match the query conditions?)
- Search latency (which is faster?)
- Result diversity (how many unique conditions appear?)
- Coverage (what percentage of queries return 5+ results?)

These metrics are real, measurable, and go directly on my resume.
"""

import json
import os
import time
import random
import statistics

from src.search.hybrid_search import hybrid_search


# Test queries simulating real patient searches
TEST_QUERIES = [
    {"text": "type 2 diabetes", "expected_conditions": ["diabetes"]},
    {"text": "breast cancer stage 2", "expected_conditions": ["breast cancer"]},
    {"text": "chronic kidney disease hypertension", "expected_conditions": ["kidney", "hypertension"]},
    {"text": "depression anxiety", "expected_conditions": ["depression", "anxiety"]},
    {"text": "lung cancer", "expected_conditions": ["lung cancer"]},
    {"text": "rheumatoid arthritis", "expected_conditions": ["arthritis"]},
    {"text": "COPD emphysema", "expected_conditions": ["copd", "pulmonary"]},
    {"text": "heart failure", "expected_conditions": ["heart failure", "cardiac"]},
    {"text": "multiple sclerosis", "expected_conditions": ["sclerosis"]},
    {"text": "Parkinsons disease", "expected_conditions": ["parkinson"]},
    {"text": "atrial fibrillation", "expected_conditions": ["fibrillation", "atrial"]},
    {"text": "prostate cancer", "expected_conditions": ["prostate"]},
    {"text": "epilepsy seizures", "expected_conditions": ["epilepsy", "seizure"]},
    {"text": "Crohns disease", "expected_conditions": ["crohn"]},
    {"text": "psoriasis skin", "expected_conditions": ["psoriasis"]},
    {"text": "obesity weight loss", "expected_conditions": ["obesity", "weight"]},
    {"text": "HIV treatment", "expected_conditions": ["hiv"]},
    {"text": "stroke recovery rehabilitation", "expected_conditions": ["stroke"]},
    {"text": "migraine headache", "expected_conditions": ["migraine"]},
    {"text": "sleep apnea", "expected_conditions": ["sleep apnea"]},
    {"text": "my kidneys are failing and I have high blood pressure", "expected_conditions": ["kidney", "hypertension", "renal"]},
    {"text": "trouble breathing and chest pain", "expected_conditions": ["pulmonary", "respiratory", "cardiac", "heart"]},
    {"text": "blood sugar problems and vision issues", "expected_conditions": ["diabetes", "ophthal", "retino"]},
    {"text": "joint pain and swelling in hands", "expected_conditions": ["arthritis"]},
    {"text": "memory loss and confusion in elderly", "expected_conditions": ["alzheimer", "dementia", "cognitive"]},
    {"text": "colorectal cancer stage 3", "expected_conditions": ["colorectal", "colon"]},
    {"text": "leukemia blood cancer", "expected_conditions": ["leukemia"]},
    {"text": "hepatitis C liver", "expected_conditions": ["hepatitis"]},
    {"text": "fibromyalgia chronic pain", "expected_conditions": ["fibromyalgia", "pain"]},
    {"text": "asthma breathing difficulty", "expected_conditions": ["asthma"]},
]


def measure_relevance(results, expected_conditions):
    """
    Check if results contain trials related to the expected conditions.

    A result is "relevant" if any of its conditions contain any of the
    expected condition keywords (case-insensitive partial match).

    Returns a score from 0.0 to 1.0.
    """
    if not results.get("trials"):
        return 0.0

    relevant_count = 0
    total = len(results["trials"])

    for trial in results["trials"]:
        trial_conditions = " ".join(trial.get("conditions", [])).lower()
        trial_title = trial.get("brief_title", "").lower()
        combined = trial_conditions + " " + trial_title

        for expected in expected_conditions:
            if expected.lower() in combined:
                relevant_count += 1
                break

    return relevant_count / total if total > 0 else 0.0


def measure_diversity(results):
    """
    Count unique conditions across all results.
    Higher diversity means the search is not just returning duplicates.
    """
    all_conditions = set()
    for trial in results.get("trials", []):
        for condition in trial.get("conditions", []):
            all_conditions.add(condition.lower())
    return len(all_conditions)


def run_ab_test():
    """Run the full A/B test comparing two retrieval strategies."""

    print("=" * 70)
    print("A/B TEST: KEYWORD-HEAVY vs VECTOR-HEAVY RETRIEVAL")
    print("=" * 70)
    print(f"\n  Test queries: {len(TEST_QUERIES)}")
    print(f"  Strategy A: keyword=0.7, vector=0.3")
    print(f"  Strategy B: keyword=0.3, vector=0.7")

    results_a = {"relevance": [], "latency": [], "diversity": [], "coverage": 0}
    results_b = {"relevance": [], "latency": [], "diversity": [], "coverage": 0}

    for i, query in enumerate(TEST_QUERIES):
        print(f"\n  Query {i+1}/{len(TEST_QUERIES)}: '{query['text'][:50]}...'")

        # Strategy A: keyword-heavy
        start_a = time.time()
        res_a = hybrid_search(
            query_text=query["text"],
            page_size=10,
            keyword_weight=0.7,
            vector_weight=0.3,
        )
        latency_a = (time.time() - start_a) * 1000

        rel_a = measure_relevance(res_a, query["expected_conditions"])
        div_a = measure_diversity(res_a)
        results_a["relevance"].append(rel_a)
        results_a["latency"].append(latency_a)
        results_a["diversity"].append(div_a)
        if len(res_a.get("trials", [])) >= 5:
            results_a["coverage"] += 1

        # Strategy B: vector-heavy
        start_b = time.time()
        res_b = hybrid_search(
            query_text=query["text"],
            page_size=10,
            keyword_weight=0.3,
            vector_weight=0.7,
        )
        latency_b = (time.time() - start_b) * 1000

        rel_b = measure_relevance(res_b, query["expected_conditions"])
        div_b = measure_diversity(res_b)
        results_b["relevance"].append(rel_b)
        results_b["latency"].append(latency_b)
        results_b["diversity"].append(div_b)
        if len(res_b.get("trials", [])) >= 5:
            results_b["coverage"] += 1

        print(f"    A: relevance={rel_a:.2f} latency={latency_a:.0f}ms diversity={div_a}")
        print(f"    B: relevance={rel_b:.2f} latency={latency_b:.0f}ms diversity={div_b}")

    # Calculate summary statistics
    n = len(TEST_QUERIES)

    avg_rel_a = statistics.mean(results_a["relevance"])
    avg_rel_b = statistics.mean(results_b["relevance"])
    avg_lat_a = statistics.mean(results_a["latency"])
    avg_lat_b = statistics.mean(results_b["latency"])
    avg_div_a = statistics.mean(results_a["diversity"])
    avg_div_b = statistics.mean(results_b["diversity"])
    cov_a = results_a["coverage"] / n
    cov_b = results_b["coverage"] / n

    print("\n" + "=" * 70)
    print("A/B TEST RESULTS")
    print("=" * 70)

    print(f"\n  {'Metric':<25} {'Strategy A':>15} {'Strategy B':>15} {'Winner':>10}")
    print(f"  {'-'*65}")

    rel_winner = "A" if avg_rel_a > avg_rel_b else "B" if avg_rel_b > avg_rel_a else "Tie"
    lat_winner = "A" if avg_lat_a < avg_lat_b else "B" if avg_lat_b < avg_lat_a else "Tie"
    div_winner = "A" if avg_div_a > avg_div_b else "B" if avg_div_b > avg_div_a else "Tie"
    cov_winner = "A" if cov_a > cov_b else "B" if cov_b > cov_a else "Tie"

    print(f"  {'Avg Relevance':<25} {avg_rel_a:>14.1%} {avg_rel_b:>14.1%} {rel_winner:>10}")
    print(f"  {'Avg Latency (ms)':<25} {avg_lat_a:>14.0f} {avg_lat_b:>14.0f} {lat_winner:>10}")
    print(f"  {'Avg Diversity':<25} {avg_div_a:>14.1f} {avg_div_b:>14.1f} {div_winner:>10}")
    print(f"  {'Coverage (5+ results)':<25} {cov_a:>14.1%} {cov_b:>14.1%} {cov_winner:>10}")

    # Improvement calculation
    if avg_rel_b > avg_rel_a:
        improvement = (avg_rel_b - avg_rel_a) / avg_rel_a * 100 if avg_rel_a > 0 else 0
        print(f"\n  Vector-heavy (B) improves relevance by {improvement:.1f}% over keyword-heavy (A)")
    else:
        improvement = (avg_rel_a - avg_rel_b) / avg_rel_b * 100 if avg_rel_b > 0 else 0
        print(f"\n  Keyword-heavy (A) improves relevance by {improvement:.1f}% over vector-heavy (B)")

    # Determine overall winner
    wins = {"A": 0, "B": 0}
    for w in [rel_winner, lat_winner, div_winner, cov_winner]:
        if w in wins:
            wins[w] += 1

    overall = "A" if wins["A"] > wins["B"] else "B" if wins["B"] > wins["A"] else "Tie"
    print(f"\n  OVERALL WINNER: Strategy {'A (keyword-heavy)' if overall == 'A' else 'B (vector-heavy)' if overall == 'B' else 'Tie'}")
    print(f"  Score: A={wins['A']} wins, B={wins['B']} wins")

    # Save results
    output = {
        "test_queries": n,
        "strategy_a": {
            "name": "keyword-heavy (0.7/0.3)",
            "avg_relevance": round(avg_rel_a, 4),
            "avg_latency_ms": round(avg_lat_a, 1),
            "avg_diversity": round(avg_div_a, 1),
            "coverage": round(cov_a, 3),
        },
        "strategy_b": {
            "name": "vector-heavy (0.3/0.7)",
            "avg_relevance": round(avg_rel_b, 4),
            "avg_latency_ms": round(avg_lat_b, 1),
            "avg_diversity": round(avg_div_b, 1),
            "coverage": round(cov_b, 3),
        },
        "winner": overall,
        "relevance_improvement_pct": round(improvement, 1),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/ab_test_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to results/ab_test_results.json")
    print(f"\n  SAVE THESE NUMBERS FOR YOUR RESUME.")

    return output


if __name__ == "__main__":
    run_ab_test()
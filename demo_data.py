"""
Generate a small demo dataset for the deployed Streamlit app.
This lets the app run without OpenSearch for live deployment.
"""

import json
import os


def create_demo_data():
    """Create a small searchable dataset from processed trials."""
    source = os.path.join("data", "processed", "trials_cleaned.json")

    if not os.path.exists(source):
        print("No processed data found. Run the data pipeline first.")
        return

    with open(source, "r", encoding="utf-8") as f:
        all_trials = json.load(f)

    # Pick 500 diverse trials for the demo
    # Prioritize trials with conditions, locations, and eligibility info
    good_trials = []
    for trial in all_trials:
        has_conditions = len(trial.get("conditions", [])) > 0
        has_locations = len(trial.get("locations", [])) > 0
        has_eligibility = trial.get("eligibility", {}).get("criteria_text", "")
        if has_conditions and has_locations and has_eligibility:
            # Remove the vector field (too large for demo file)
            trial.pop("search_vector", None)
            good_trials.append(trial)

        if len(good_trials) >= 500:
            break

    output_path = "demo_trials.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(good_trials, f, ensure_ascii=False)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Created {output_path} with {len(good_trials)} trials ({size_mb:.1f} MB)")


if __name__ == "__main__":
    create_demo_data()
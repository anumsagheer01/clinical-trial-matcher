"""
Download clinical trial data from ClinicalTrials.gov API.

This script downloads trials in batches using their public API.
Goal is to focus on trials that are currently recruiting or not yet recruiting,
because those are the ones patients can actually join.

API docs: https://clinicaltrials.gov/data-api/api
"""

import json
import os
import time
import requests
from tqdm import tqdm


# === CONFIGURATION ===

# Base URL for the ClinicalTrials.gov API (version 2)
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

# Keeping only the trials patients can potentially join
# "RECRUITING" = actively looking for participants right now
# "NOT_YET_RECRUITING" = approved but hasn't started enrolling yet
STATUSES = ["RECRUITING", "NOT_YET_RECRUITING"]

# How many trials to download per API call (max is 1000)
PAGE_SIZE = 1000

# Where to save the raw downloaded data
RAW_DATA_DIR = os.path.join("data", "raw")

# Wait this many seconds between requests
DELAY_BETWEEN_REQUESTS = 1.0


def download_all_trials():
    """
    Download all recruiting/not-yet-recruiting trials from ClinicalTrials.gov.
    
    How it works:
    1. Ask the API for trials, 1000 at a time
    2. The API gives "nextPageToken" to get the next batch
    3. Keep going until there are no more pages
    4. Save everything to a single JSON file
    
    Returns:
        list: All downloaded trial studies
    """
    
    # Make sure save directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    all_studies = []
    page_token = None  # First request has no token
    page_number = 1
    
    print("=" * 60)
    print("DOWNLOADING CLINICAL TRIALS FROM ClinicalTrials.gov")
    print("=" * 60)
    print(f"Looking for trials with status: {', '.join(STATUSES)}")
    print()
    
    while True:
        # Build the API request parameters
        params = {
            "format": "json",
            "pageSize": PAGE_SIZE,
            "filter.overallStatus": ",".join(STATUSES),
            # Request specific fields only 
            "fields": ",".join([
                "NCTId",
                "BriefTitle",
                "OfficialTitle",
                "BriefSummary",
                "DetailedDescription",
                "OverallStatus",
                "Phase",
                "StudyType",
                "EnrollmentInfo",
                "Condition",
                "Keyword",
                "InterventionName",
                "InterventionType",
                "InterventionDescription",
                "EligibilityModule",
                "LocationFacility",
                "LocationCity",
                "LocationState",
                "LocationCountry",
                "LocationStatus",
                "LeadSponsorName",
                "StartDate",
                "CompletionDate",
                "LastUpdatePostDate",
            ]),
        }
        
        # If there is a page token from the previous response, include it
        if page_token:
            params["pageToken"] = page_token
        
        # Make the API call
        print(f"  Downloading page {page_number}...", end=" ")
        
        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()  # Raises an error if HTTP status is 4xx or 5xx
        except requests.exceptions.RequestException as e:
            print(f"\n  ERROR: Failed to download page {page_number}: {e}")
            print("  Retrying in 5 seconds")
            time.sleep(5)
            continue  # Try the same page again
        
        data = response.json()
        
        # Extract the studies from this page
        studies = data.get("studies", [])
        all_studies.extend(studies)
        
        print(f"got {len(studies)} trials (total so far: {len(all_studies)})")
        
        # Check if there's another page
        page_token = data.get("nextPageToken")
        
        if not page_token:
            # If no more pages
            print(f"\n  Download complete! Total trials: {len(all_studies)}")
            break
        
        page_number += 1
        
        # Wait time 
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Save the raw data
    output_path = os.path.join(RAW_DATA_DIR, "raw_trials.json")
    print(f"\n  Saving raw data to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_studies, f, indent=2, ensure_ascii=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved! File size: {file_size_mb:.1f} MB")
    
    return all_studies


if __name__ == "__main__":
    
    download_all_trials()
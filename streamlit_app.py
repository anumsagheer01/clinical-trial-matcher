"""
Clinical Trial Matcher - Patient Frontend
Clean, accessible interface for finding clinical trials.
"""

import streamlit as st
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEMO_MODE = False
try:
    from src.search.hybrid_search import hybrid_search
    from src.mcp_servers.entity_extraction_server import extract_patient_entities
    from src.query_classifier.classifier import classify_query
except ImportError:
    DEMO_MODE = True

from dotenv import load_dotenv

load_dotenv()


def demo_search(query_text, page_size=10, **kwargs):
    """Fallback search using local JSON when OpenSearch is not available."""
    try:
        with open("demo_trials.json", "r", encoding="utf-8") as f:
            trials = json.load(f)
    except FileNotFoundError:
        return {"total": 0, "trials": [], "took_ms": 0}

    query_lower = query_text.lower()
    scored = []

    for trial in trials:
        score = 0
        searchable = (
            " ".join(trial.get("conditions", [])).lower() + " " +
            trial.get("brief_title", "").lower() + " " +
            trial.get("brief_summary", "").lower()
        )

        for word in query_lower.split():
            if word in searchable:
                score += 1

        if score > 0:
            trial["_score"] = score
            scored.append(trial)

    scored.sort(key=lambda x: x["_score"], reverse=True)

    return {
        "total": len(scored),
        "trials": scored[:page_size],
        "took_ms": 0,
    }


# Page config
st.set_page_config(
    page_title="Clinical Trial Matcher",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for clean, accessible design
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 1.15rem;
        color: #64748b;
        font-weight: 400;
    }

    .trial-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s ease;
    }
    .trial-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .trial-card h3 {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    .trial-card .nct-id {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        font-family: monospace;
    }
    .trial-card .conditions-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.6rem 0;
    }
    .condition-tag {
        background: #eff6ff;
        color: #1e40af;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .trial-card .summary {
        font-size: 1rem;
        color: #475569;
        line-height: 1.6;
        margin: 0.5rem 0;
    }

    .elig-row {
        display: flex;
        gap: 1.5rem;
        margin-top: 0.6rem;
        flex-wrap: wrap;
    }
    .elig-item {
        font-size: 0.9rem;
        color: #475569;
    }
    .elig-item strong {
        color: #1a365d;
    }

    .stats-bar {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 1.5rem;
        font-weight: 800;
        color: #1a365d;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .ct-link {
        display: inline-block;
        margin-top: 0.5rem;
        padding: 0.4rem 1rem;
        background: #1a365d;
        color: white !important;
        border-radius: 8px;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .ct-link:hover {
        background: #2d4a7c;
    }

    .stTextArea textarea {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        border-radius: 12px !important;
        border: 2px solid #cbd5e1 !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #1a365d !important;
        box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.12) !important;
    }

    .stButton > button {
        background: #1a365d !important;
        color: white !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding: 0.7rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        transition: background 0.2s !important;
    }
    .stButton > button:hover {
        background: #2d4a7c !important;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a365d;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Clinical Trial Matcher</h1>
        <p>Describe your medical situation below. We will search 92,000+ clinical trials to find ones you may qualify for.</p>
    </div>
    """, unsafe_allow_html=True)

    # Main input
    description = st.text_area(
        "Describe your medical situation",
        placeholder="Example: I am a 58 year old male with type 2 diabetes and chronic kidney disease stage 3. I currently take metformin and lisinopril. I live in Maryland.",
        height=160,
        help="Include your age, sex, medical conditions, any medications you take, and your location.",
    )

    # Optional filters
    st.markdown('<p style="font-size: 1.1rem; font-weight: 600; color: #1a365d; margin-top: 1rem;">Optional Filters</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
        age = age if age > 0 else None

    with col2:
        sex = st.selectbox("Sex", ["Any", "Male", "Female"])
        sex = sex if sex != "Any" else None

    with col3:
        state = st.selectbox("State", [
            "Any", "Alabama", "Alaska", "Arizona", "Arkansas", "California",
            "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
            "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
            "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
            "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
            "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
            "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
            "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
            "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
            "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
        ])
        state = state if state != "Any" else None

    with col4:
        phase = st.selectbox("Trial Phase", ["Any Phase", "Phase 1", "Phase 2", "Phase 3", "Phase 4"])
        phase_map = {"Phase 1": "PHASE1", "Phase 2": "PHASE2", "Phase 3": "PHASE3", "Phase 4": "PHASE4"}
        phase = phase_map.get(phase)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Find Matching Trials", type="primary"):
        if not description.strip():
            st.warning("Please describe your medical situation above.")
            return

        # Run search based on mode
        if DEMO_MODE:
            entities = {"age": age, "sex": sex, "conditions": [], "medications": [], "location": state}
            query_class = {"classification": "simple", "confidence": 1.0}
            results = demo_search(description, page_size=10)
        else:
            with st.spinner("Analyzing your query..."):
                query_class = classify_query(description)

            with st.spinner("Understanding your medical information..."):
                entities_json = extract_patient_entities(description)
                entities = json.loads(entities_json)

            search_age = age or entities.get("age")
            search_sex = sex or entities.get("sex")
            search_state = state or entities.get("location")

            with st.spinner("Searching clinical trials..."):
                try:
                    results = hybrid_search(
                        query_text=description,
                        min_age=search_age,
                        max_age=search_age,
                        sex=search_sex,
                        state=search_state,
                        phase=phase,
                        status="RECRUITING",
                        page_size=10,
                    )
                except Exception:
                    results = demo_search(description, page_size=10)

        # Show what was understood
        st.markdown('<div class="section-header">What I Understood</div>', unsafe_allow_html=True)

        ecol1, ecol2, ecol3 = st.columns(3)
        with ecol1:
            st.markdown(f"**Age:** {entities.get('age') or age or 'Not specified'}")
            st.markdown(f"**Sex:** {entities.get('sex') or sex or 'Not specified'}")
        with ecol2:
            conditions = entities.get("conditions", [])
            if conditions:
                st.markdown(f"**Conditions:** {', '.join(conditions)}")
            else:
                st.markdown("**Conditions:** Searching by description")
        with ecol3:
            meds = entities.get("medications", [])
            if meds:
                st.markdown(f"**Medications:** {', '.join(meds)}")
            st.markdown(f"**Location:** {entities.get('location') or state or 'Any'}")

        # Stats bar
        total = results.get("total", 0)
        trials = results.get("trials", [])
        search_ms = results.get("took_ms", 0)

        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-number">{total:,}</div>
                <div class="stat-label">Total Matches</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(trials)}</div>
                <div class="stat-label">Showing</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{search_ms}ms</div>
                <div class="stat-label">Search Time</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{query_class['classification'].title()}</div>
                <div class="stat-label">Query Type</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not trials:
            st.info("No matching trials found. Try broadening your search by removing some filters.")
            return

        st.markdown('<div class="section-header">Matching Clinical Trials</div>', unsafe_allow_html=True)

        for i, trial in enumerate(trials):
            nct_id = trial.get("nct_id", "")
            title = trial.get("brief_title", "No title")
            conditions_list = trial.get("conditions", [])
            summary = trial.get("brief_summary", "")[:300]
            elig = trial.get("eligibility", {})
            locations = trial.get("locations", [])
            phases = trial.get("phases", [])

            min_a = elig.get("min_age")
            max_a = elig.get("max_age")
            age_text = "Any age"
            if min_a and max_a:
                age_text = f"{min_a} to {max_a} years"
            elif min_a:
                age_text = f"{min_a}+ years"
            elif max_a:
                age_text = f"Up to {max_a} years"

            loc_count = len(locations)
            loc_text = "No locations listed"
            if loc_count > 0:
                first = locations[0]
                city = first.get("city", "")
                loc_state = first.get("state", "")
                loc_text = f"{city}, {loc_state}" if city else loc_state
                if loc_count > 1:
                    loc_text += f" and {loc_count - 1} other sites"

            phase_text = ", ".join(p.replace("PHASE", "Phase ") for p in phases) if phases else "Not specified"

            condition_tags = ""
            for c in conditions_list[:5]:
                condition_tags += f'<span class="condition-tag">{c}</span>'

            ct_link = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "#"

            st.markdown(f"""
            <div class="trial-card">
                <span class="nct-id">{nct_id}</span>
                <h3>{title}</h3>
                <div class="conditions-list">{condition_tags}</div>
                <p class="summary">{summary}{'...' if len(summary) >= 300 else ''}</p>
                <div class="elig-row">
                    <div class="elig-item"><strong>Age:</strong> {age_text}</div>
                    <div class="elig-item"><strong>Sex:</strong> {elig.get('sex', 'All')}</div>
                    <div class="elig-item"><strong>Phase:</strong> {phase_text}</div>
                    <div class="elig-item"><strong>Sites:</strong> {loc_text}</div>
                </div>
                <a href="{ct_link}" target="_blank" class="ct-link">View on ClinicalTrials.gov</a>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem 0; border-top: 1px solid #e2e8f0;">
        Clinical Trial Matcher searches ClinicalTrials.gov data. This tool is for informational purposes only.
        Always consult your doctor before enrolling in a clinical trial.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

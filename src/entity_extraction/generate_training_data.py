"""
Generate synthetic training data for medical entity extraction.

Input:  "58 year old male with type 2 diabetes and CKD stage 3, taking metformin"
Output: {
    "age": 58,
    "sex": "Male",
    "conditions": ["type 2 diabetes", "chronic kidney disease stage 3"],
    "medications": ["metformin"],
    "location": null
}

Generating 2,000 examples because:
- Too few (<500) and the model will not learn enough patterns
- Too many (>10,000) and training takes forever on a CPU
- 2,000 is the sweet spot for a fine-tuning task with LoRA
"""

import json
import os
import random

random.seed(42)

OUTPUT_DIR = os.path.join("data", "processed", "entity_training")

# === MEDICAL KNOWLEDGE BASE ===

CONDITIONS = [
    ("type 2 diabetes", ["type 2 diabetes", "t2dm", "type II diabetes", "diabetes mellitus type 2"]),
    ("type 1 diabetes", ["type 1 diabetes", "t1dm", "type I diabetes", "juvenile diabetes"]),
    ("hypertension", ["hypertension", "high blood pressure", "htn", "elevated blood pressure"]),
    ("chronic kidney disease", ["chronic kidney disease", "ckd", "kidney disease", "renal disease"]),
    ("heart failure", ["heart failure", "congestive heart failure", "chf", "cardiac failure"]),
    ("coronary artery disease", ["coronary artery disease", "cad", "heart disease"]),
    ("breast cancer", ["breast cancer", "breast carcinoma", "breast tumor"]),
    ("lung cancer", ["lung cancer", "non-small cell lung cancer", "nsclc", "lung carcinoma"]),
    ("colorectal cancer", ["colorectal cancer", "colon cancer", "rectal cancer"]),
    ("prostate cancer", ["prostate cancer", "prostate carcinoma"]),
    ("depression", ["depression", "major depressive disorder", "mdd", "clinical depression"]),
    ("anxiety", ["anxiety", "generalized anxiety disorder", "gad", "anxiety disorder"]),
    ("ptsd", ["ptsd", "post-traumatic stress disorder", "post traumatic stress disorder"]),
    ("copd", ["copd", "chronic obstructive pulmonary disease", "emphysema"]),
    ("asthma", ["asthma", "bronchial asthma", "reactive airway disease"]),
    ("rheumatoid arthritis", ["rheumatoid arthritis", "ra", "inflammatory arthritis"]),
    ("osteoarthritis", ["osteoarthritis", "oa", "degenerative joint disease"]),
    ("multiple sclerosis", ["multiple sclerosis", "ms", "relapsing-remitting ms"]),
    ("parkinsons disease", ["parkinsons disease", "parkinson disease", "pd"]),
    ("alzheimers disease", ["alzheimers disease", "alzheimer disease", "ad", "dementia"]),
    ("atrial fibrillation", ["atrial fibrillation", "afib", "a-fib"]),
    ("obesity", ["obesity", "morbid obesity", "severe obesity", "bmi over 30"]),
    ("sleep apnea", ["sleep apnea", "obstructive sleep apnea", "osa"]),
    ("migraine", ["migraine", "chronic migraine", "migraine headaches"]),
    ("epilepsy", ["epilepsy", "seizure disorder", "convulsive disorder"]),
    ("crohns disease", ["crohns disease", "crohn disease", "cd"]),
    ("ulcerative colitis", ["ulcerative colitis", "uc", "colitis"]),
    ("psoriasis", ["psoriasis", "plaque psoriasis", "chronic psoriasis"]),
    ("lupus", ["lupus", "systemic lupus erythematosus", "sle"]),
    ("hepatitis c", ["hepatitis c", "hep c", "hcv", "chronic hepatitis c"]),
    ("hiv", ["hiv", "human immunodeficiency virus", "hiv positive"]),
    ("chronic pain", ["chronic pain", "persistent pain", "long-term pain"]),
    ("fibromyalgia", ["fibromyalgia", "fibromyalgia syndrome", "fms"]),
    ("anemia", ["anemia", "iron deficiency anemia", "low hemoglobin"]),
    ("hypothyroidism", ["hypothyroidism", "underactive thyroid", "low thyroid"]),
    ("gout", ["gout", "gouty arthritis", "hyperuricemia"]),
    ("peripheral neuropathy", ["peripheral neuropathy", "neuropathy", "nerve damage"]),
    ("stroke", ["stroke", "cerebrovascular accident", "cva", "brain attack"]),
    ("leukemia", ["leukemia", "acute lymphoblastic leukemia", "chronic lymphocytic leukemia"]),
]

MEDICATIONS = [
    ("metformin", ["metformin", "glucophage", "metformin hcl"]),
    ("lisinopril", ["lisinopril", "prinivil", "zestril"]),
    ("amlodipine", ["amlodipine", "norvasc"]),
    ("atorvastatin", ["atorvastatin", "lipitor"]),
    ("omeprazole", ["omeprazole", "prilosec"]),
    ("levothyroxine", ["levothyroxine", "synthroid", "levoxyl"]),
    ("losartan", ["losartan", "cozaar"]),
    ("gabapentin", ["gabapentin", "neurontin"]),
    ("sertraline", ["sertraline", "zoloft"]),
    ("fluoxetine", ["fluoxetine", "prozac"]),
    ("prednisone", ["prednisone", "deltasone"]),
    ("insulin", ["insulin", "insulin glargine", "lantus", "humalog"]),
    ("warfarin", ["warfarin", "coumadin"]),
    ("aspirin", ["aspirin", "baby aspirin", "low-dose aspirin"]),
    ("ibuprofen", ["ibuprofen", "advil", "motrin"]),
    ("albuterol", ["albuterol", "ventolin", "proair"]),
    ("montelukast", ["montelukast", "singulair"]),
    ("hydrochlorothiazide", ["hydrochlorothiazide", "hctz"]),
    ("furosemide", ["furosemide", "lasix"]),
    ("carvedilol", ["carvedilol", "coreg"]),
    ("tamoxifen", ["tamoxifen", "nolvadex"]),
    ("methotrexate", ["methotrexate", "trexall"]),
    ("hydroxychloroquine", ["hydroxychloroquine", "plaquenil"]),
    ("semaglutide", ["semaglutide", "ozempic", "wegovy"]),
    ("empagliflozin", ["empagliflozin", "jardiance"]),
    ("sitagliptin", ["sitagliptin", "januvia"]),
    ("apixaban", ["apixaban", "eliquis"]),
    ("duloxetine", ["duloxetine", "cymbalta"]),
]

US_LOCATIONS = [
    ("Maryland", ["Maryland", "MD", "Baltimore", "College Park", "Bethesda"]),
    ("California", ["California", "CA", "Los Angeles", "San Francisco", "San Diego"]),
    ("New York", ["New York", "NY", "NYC", "Manhattan", "Brooklyn"]),
    ("Texas", ["Texas", "TX", "Houston", "Dallas", "Austin"]),
    ("Florida", ["Florida", "FL", "Miami", "Tampa", "Orlando"]),
    ("Illinois", ["Illinois", "IL", "Chicago"]),
    ("Pennsylvania", ["Pennsylvania", "PA", "Philadelphia", "Pittsburgh"]),
    ("Massachusetts", ["Massachusetts", "MA", "Boston", "Cambridge"]),
    ("Ohio", ["Ohio", "OH", "Cleveland", "Columbus", "Cincinnati"]),
    ("Georgia", ["Georgia", "GA", "Atlanta"]),
    ("North Carolina", ["North Carolina", "NC", "Charlotte", "Raleigh"]),
    ("Michigan", ["Michigan", "MI", "Detroit", "Ann Arbor"]),
    ("Virginia", ["Virginia", "VA", "Richmond", "Arlington"]),
    ("Washington", ["Washington", "WA", "Seattle", "Tacoma"]),
    ("Arizona", ["Arizona", "AZ", "Phoenix", "Tucson"]),
]

TEMPLATES = [
    "{age} year old {sex} with {conditions}. Currently taking {medications}. Located in {location}.",
    "I am a {age}-year-old {sex} diagnosed with {conditions}. My current medications include {medications}. I live in {location}.",
    "{sex}, age {age}, presenting with {conditions}. On {medications}. Based in {location}.",
    "Patient is a {age} yo {sex_abbrev} with history of {conditions}. Meds: {medications}. Lives in {location}.",
    "{age} y/o {sex} patient with {conditions}, on {medications}.",
    "I have {conditions}. I am {age} years old, {sex}. I take {medications} daily. I am in {location}.",
    "{sex} patient, {age}, diagnosed with {conditions}. Current medications: {medications}. Location: {location}.",
    "Hi, I am a {age} year old {sex}. I have been diagnosed with {conditions}. I currently take {medications}. I live near {location}.",
    "{conditions} - {age} year old {sex}. Medications: {medications}.",
    "I was diagnosed with {conditions} last year. I am {age}, {sex}. Taking {medications}. I am in {location}.",
    "{age} {sex} with {conditions}. Meds include {medications}.",
    "Patient: {age} year old {sex}. Conditions: {conditions}. Medications: {medications}. Area: {location}.",
    "I am looking for clinical trials. I have {conditions}. Age {age}, {sex}. On {medications}. Near {location}.",
    "{sex}, {age} years old. Suffering from {conditions} for several years. Taking {medications}.",
    "Diagnosed with {conditions} at age {diag_age}. Now {age}, {sex}. Currently on {medications}. Living in {location}.",
]

TEMPLATES_NO_LOCATION = [
    "{age} year old {sex} with {conditions}. Currently taking {medications}.",
    "I have {conditions}. Age {age}, {sex}. On {medications}.",
    "{sex}, {age}, with {conditions}. Medications: {medications}.",
    "Patient is {age} yo {sex_abbrev} with {conditions}, on {medications}.",
]

TEMPLATES_NO_MEDS = [
    "{age} year old {sex} with {conditions}. Located in {location}.",
    "I am a {age} year old {sex} with {conditions}. I live in {location}.",
    "{age} {sex} diagnosed with {conditions}. Based in {location}.",
]

TEMPLATES_MINIMAL = [
    "I have {conditions}.",
    "{conditions} patient looking for trials.",
    "Diagnosed with {conditions}. Age {age}.",
    "{age} year old with {conditions}.",
]


def pick_random(item_list, count):
    """Pick random items from a list of (canonical, variants) tuples."""
    chosen = random.sample(item_list, min(count, len(item_list)))
    results = []
    for canonical, variants in chosen:
        display = random.choice(variants)
        results.append((canonical, display))
    return results


def generate_one_example():
    """Generate a single training example."""
    age = random.randint(18, 85)
    sex_canonical = random.choice(["Male", "Female"])
    sex_display = random.choice({
        "Male": ["male", "Male", "man", "M", "gentleman"],
        "Female": ["female", "Female", "woman", "F", "lady"],
    }[sex_canonical])
    sex_abbrev = "M" if sex_canonical == "Male" else "F"

    num_conditions = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
    conditions = pick_random(CONDITIONS, num_conditions)
    conditions_canonical = [c[0] for c in conditions]
    conditions_display = [c[1] for c in conditions]

    final_conditions_canonical = []
    final_conditions_display = []
    for canon, display in zip(conditions_canonical, conditions_display):
        if canon == "chronic kidney disease" and random.random() > 0.5:
            stage = random.randint(1, 5)
            canon = f"chronic kidney disease stage {stage}"
            display = f"{display} stage {stage}"
        if "cancer" in canon and random.random() > 0.5:
            stage_num = random.choice([1, 2, 3, 4])
            canon = f"{canon} stage {stage_num}"
            display = f"{display} stage {stage_num}"
        final_conditions_canonical.append(canon)
        final_conditions_display.append(display)

    conditions_text = " and ".join(final_conditions_display)
    if len(final_conditions_display) > 2:
        conditions_text = ", ".join(final_conditions_display[:-1]) + ", and " + final_conditions_display[-1]

    num_meds = random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.3, 0.3, 0.2, 0.1])[0]
    medications = pick_random(MEDICATIONS, num_meds)
    meds_canonical = [m[0] for m in medications]
    meds_display = [m[1] for m in medications]
    meds_text = " and ".join(meds_display)
    if len(meds_display) > 2:
        meds_text = ", ".join(meds_display[:-1]) + ", and " + meds_display[-1]

    include_location = random.random() > 0.3
    location_canonical = None
    location_display = ""
    if include_location:
        loc = random.choice(US_LOCATIONS)
        location_canonical = loc[0]
        location_display = random.choice(loc[1])

    diag_age = age - random.randint(1, min(20, age - 18)) if age > 20 else age

    if num_meds == 0 and include_location:
        template = random.choice(TEMPLATES_NO_MEDS)
    elif num_meds == 0 and not include_location:
        template = random.choice(TEMPLATES_MINIMAL)
    elif not include_location:
        template = random.choice(TEMPLATES_NO_LOCATION)
    else:
        template = random.choice(TEMPLATES)

    text = template.format(
        age=age, sex=sex_display, sex_abbrev=sex_abbrev,
        conditions=conditions_text, medications=meds_text,
        location=location_display, diag_age=diag_age,
    )

    label = {
        "age": age,
        "sex": sex_canonical,
        "conditions": final_conditions_canonical,
        "medications": meds_canonical,
        "location": location_canonical,
    }

    return {"text": text, "entities": label}


def generate_dataset(num_examples=5000):
    """Generate the full training dataset."""
    print("=" * 60)
    print("GENERATING SYNTHETIC TRAINING DATA")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    examples = []
    for i in range(num_examples):
        examples.append(generate_one_example())

    random.shuffle(examples)
    train_size = int(0.8 * len(examples))
    val_size = int(0.1 * len(examples))

    train_data = examples[:train_size]
    val_data = examples[train_size:train_size + val_size]
    test_data = examples[train_size + val_size:]

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(data)} examples to {path}")

    print("\n  Sample examples:")
    for ex in examples[:3]:
        print(f"\n  Input: {ex['text']}")
        print(f"  Output: {json.dumps(ex['entities'])}")

    print(f"\n  Total: {len(examples)} examples")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    return train_data, val_data, test_data


if __name__ == "__main__":
    generate_dataset()
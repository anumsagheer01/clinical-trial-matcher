"""
TensorFlow query complexity classifier.

Classifies patient queries as "simple" (search only) or "complex" (needs full AI pipeline).
Simple queries skip Claude entirely, saving time and money.
"""

import os
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers, models

MODEL_DIR = os.path.join("models", "query_classifier")


def create_features(text):
    """
    Hand-crafted features that capture query complexity.
    No embeddings needed, keeps it lightweight.
    """
    words = text.lower().split()
    return [
        len(words),
        sum(1 for w in words if w.isdigit()),
        text.lower().count("and"),
        text.lower().count(","),
        1 if any(w in text.lower() for w in ["taking", "on", "medication", "meds", "prescribed"]) else 0,
        1 if any(w in text.lower() for w in ["stage", "grade", "type", "class"]) else 0,
        1 if any(w in text.lower() for w in ["male", "female", "man", "woman"]) else 0,
        1 if any(c.isdigit() for c in text[:20]) else 0,
        len(text),
        text.count("."),
    ]


def generate_classifier_data():
    """Generate training data for the classifier."""
    simple_examples = [
        "diabetes trials", "cancer clinical trials", "depression studies",
        "heart disease trials in New York", "asthma research studies",
        "trials for arthritis", "lung cancer phase 3", "recruiting diabetes studies",
        "breast cancer trials California", "COPD clinical trials",
        "migraine studies", "epilepsy research", "HIV trials",
        "obesity studies near me", "psoriasis clinical trials",
        "trials for anxiety", "Alzheimer research studies",
        "leukemia trials phase 2", "hypertension studies", "stroke clinical trials",
    ]

    complex_examples = [
        "58 year old male with type 2 diabetes and chronic kidney disease stage 3. Taking metformin and lisinopril.",
        "I am a 45 year old woman diagnosed with breast cancer stage 2 and depression. On tamoxifen and sertraline. In Maryland.",
        "72 yo M with COPD, hypertension, and atrial fibrillation. Meds: albuterol, losartan, apixaban. Ohio.",
        "34 female with lupus and rheumatoid arthritis, currently on methotrexate and hydroxychloroquine.",
        "Patient is 62, male, type 1 diabetes with peripheral neuropathy. Taking insulin glargine and gabapentin.",
        "I have stage 4 lung cancer, age 55, male. On carboplatin. Also have hypertension and take amlodipine.",
        "67 year old female with congestive heart failure and CKD stage 4. On furosemide, carvedilol, and lisinopril.",
        "23 year old woman with Crohns disease and anemia. Taking adalimumab and iron supplements. Boston area.",
        "I am 80 years old, male, with Parkinsons disease and depression. Taking levodopa and fluoxetine.",
        "41 yo F with multiple sclerosis, relapsing type. On ocrelizumab. Also has hypothyroidism, takes levothyroxine.",
        "55 male, prostate cancer stage 3, hypertension, diabetes. Meds: enzalutamide, metformin, losartan. Texas.",
        "28 year old female, epilepsy and migraine. On lamotrigine and topiramate. Looking for trials in California.",
        "My father is 70, has COPD and heart failure. He takes albuterol, furosemide, and warfarin. We are in Florida.",
        "Patient: 49 yo M with hepatitis C and cirrhosis. On sofosbuvir. Also has type 2 diabetes, takes sitagliptin.",
        "I am a 38 year old woman with fibromyalgia, anxiety, and IBS. Taking duloxetine and dicyclomine. Pennsylvania.",
        "63 male with atrial fibrillation and stroke history. On apixaban and atorvastatin. Looking for rehab trials.",
        "44 year old female, breast cancer stage 1 HER2 positive. Post surgery, on trastuzumab. No other conditions.",
        "I am 56, male, with obesity BMI 42, sleep apnea, and prediabetes. On CPAP. Want weight loss trial.",
        "71 yo F with osteoarthritis in both knees, hypertension. Taking ibuprofen and hydrochlorothiazide. Michigan.",
        "29 year old male with ulcerative colitis, moderate severity. On mesalamine. Also has anxiety, takes sertraline.",
    ]

    all_simple = simple_examples * 5
    all_complex = complex_examples * 5

    X, y = [], []
    for text in all_simple:
        X.append(create_features(text))
        y.append(0)
    for text in all_complex:
        X.append(create_features(text))
        y.append(1)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_classifier():
    """Train the TensorFlow query classifier."""
    print("=" * 60)
    print("TRAINING TENSORFLOW QUERY CLASSIFIER")
    print("=" * 60)

    X, y = generate_classifier_data()
    print(f"  Data: {len(X)} examples ({int(sum(y))} complex, {int(len(y) - sum(y))} simple)")

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std

    model = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("  Training...")
    history = model.fit(X_norm, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0)

    final_acc = history.history["val_accuracy"][-1]
    print(f"  Validation accuracy: {final_acc:.1%}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "query_classifier.keras"))
    with open(os.path.join(MODEL_DIR, "norm_stats.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    print(f"  Model saved to {MODEL_DIR}")
    return model, mean, std


_model = None
_mean = None
_std = None


def classify_query(text):
    """Classify a query as simple or complex."""
    global _model, _mean, _std

    if _model is None:
        model_path = os.path.join(MODEL_DIR, "query_classifier.keras")
        if not os.path.exists(model_path):
            return {"classification": "complex", "confidence": 0.5, "model": "default"}
        _model = tf.keras.models.load_model(model_path)
        with open(os.path.join(MODEL_DIR, "norm_stats.json")) as f:
            stats = json.load(f)
            _mean = np.array(stats["mean"])
            _std = np.array(stats["std"])

    features = np.array([create_features(text)], dtype=np.float32)
    features_norm = (features - _mean) / _std
    prediction = _model.predict(features_norm, verbose=0)[0][0]

    return {
        "classification": "complex" if prediction > 0.5 else "simple",
        "confidence": round(float(prediction if prediction > 0.5 else 1 - prediction), 3),
        "model": "tensorflow_query_classifier",
    }


if __name__ == "__main__":
    train_classifier()
    print("\n  Testing:")
    tests = [
        "diabetes trials",
        "58 year old male with type 2 diabetes and CKD stage 3. Taking metformin.",
        "lung cancer phase 3",
        "45 year old woman with breast cancer stage 2 and depression. On tamoxifen.",
    ]
    for t in tests:
        r = classify_query(t)
        print(f"    '{t[:50]}...' -> {r['classification']} ({r['confidence']:.0%})")
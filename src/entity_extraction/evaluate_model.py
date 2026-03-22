"""
Evaluate the fine-tuned entity extraction model.
Measures: exact match accuracy, field-level F1 scores, inference latency.
These metrics go directly on my resume.
"""

import json
import os
import time

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

MODEL_NAME = "google/flan-t5-small"
ADAPTER_PATH = os.path.join("models", "entity_extractor", "lora_adapter")
TEST_DATA = os.path.join("data", "processed", "entity_training", "test.json")


def load_model():
    """Load the base model and apply the LoRA adapter."""
    print("  Loading base model...")
    tokenizer = T5Tokenizer.from_pretrained(ADAPTER_PATH)
    base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def extract_entities(model, tokenizer, text):
    """Run the model on a single patient description."""
    input_text = f"Extract medical entities from this patient description: {text}"
    inputs = tokenizer(input_text, max_length=256, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256, num_beams=2, early_stopping=True,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        try:
            return json.loads("{" + decoded + "}")
        except json.JSONDecodeError:
            return None


def compute_f1(predicted_set, true_set):
    """Compute F1 score between two sets."""
    if not predicted_set and not true_set:
        return 1.0
    if not predicted_set or not true_set:
        return 0.0
    true_positives = len(predicted_set.intersection(true_set))
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(true_set)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate():
    """Run full evaluation on test set."""
    print("=" * 60)
    print("EVALUATING ENTITY EXTRACTION MODEL")
    print("=" * 60)

    model, tokenizer = load_model()

    with open(TEST_DATA, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"\n  Test set size: {len(test_data)}")

    total = len(test_data)
    valid_json = 0
    exact_match = 0
    age_correct = 0
    sex_correct = 0
    conditions_f1_sum = 0
    medications_f1_sum = 0
    location_correct = 0
    latencies = []

    print("\n  Running inference on test set...")

    for i, example in enumerate(test_data):
        expected = example["entities"]

        start = time.time()
        predicted = extract_entities(model, tokenizer, example["text"])
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        if predicted is None:
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{total}")
            continue

        valid_json += 1

        if predicted.get("age") == expected["age"]:
            age_correct += 1
        if predicted.get("sex", "").lower() == expected["sex"].lower():
            sex_correct += 1
        if predicted.get("location") == expected["location"]:
            location_correct += 1

        pred_conditions = set(c.lower() for c in predicted.get("conditions", []))
        true_conditions = set(c.lower() for c in expected["conditions"])
        conditions_f1_sum += compute_f1(pred_conditions, true_conditions)

        pred_meds = set(m.lower() for m in predicted.get("medications", []))
        true_meds = set(m.lower() for m in expected["medications"])
        medications_f1_sum += compute_f1(pred_meds, true_meds)

        if (
            predicted.get("age") == expected["age"]
            and predicted.get("sex", "").lower() == expected["sex"].lower()
            and pred_conditions == true_conditions
            and pred_meds == true_meds
            and predicted.get("location") == expected["location"]
        ):
            exact_match += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total}")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  Valid JSON output: {valid_json}/{total} ({100 * valid_json / total:.1f}%)")
    print(f"  Exact match accuracy: {exact_match}/{total} ({100 * exact_match / total:.1f}%)")
    print(f"\n  Field-level accuracy:")
    print(f"    Age:            {100 * age_correct / total:.1f}%")
    print(f"    Sex:            {100 * sex_correct / total:.1f}%")
    print(f"    Location:       {100 * location_correct / total:.1f}%")
    print(f"    Conditions F1:  {100 * conditions_f1_sum / total:.1f}%")
    print(f"    Medications F1: {100 * medications_f1_sum / total:.1f}%")

    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]

    print(f"\n  Inference latency:")
    print(f"    Average: {avg_latency:.0f}ms")
    print(f"    P50:     {p50:.0f}ms")
    print(f"    P95:     {p95:.0f}ms")

    print(f"\n  Cost comparison:")
    print(f"    This model: ~$0.0001 per query (self-hosted)")
    print(f"    Claude API: ~$0.003 per query")
    print(f"    Savings: ~30x cheaper")

    results = {
        "total_examples": total,
        "valid_json_rate": round(valid_json / total, 3),
        "exact_match_accuracy": round(exact_match / total, 3),
        "age_accuracy": round(age_correct / total, 3),
        "sex_accuracy": round(sex_correct / total, 3),
        "location_accuracy": round(location_correct / total, 3),
        "conditions_f1": round(conditions_f1_sum / total, 3),
        "medications_f1": round(medications_f1_sum / total, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
    }

    results_path = os.path.join("models", "entity_extractor", "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    return results


if __name__ == "__main__":
    evaluate()
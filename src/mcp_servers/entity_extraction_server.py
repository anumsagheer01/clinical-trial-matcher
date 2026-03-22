"""
MCP Server: Medical Entity Extraction

Exposes the fine-tuned model as an MCP tool.
Takes raw patient text, returns structured entities in ~100ms.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcp.server.fastmcp import FastMCP
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

mcp = FastMCP("MedicalEntityExtraction")

MODEL_NAME = "google/flan-t5-small"
ADAPTER_PATH = os.path.join("models", "entity_extractor", "lora_adapter")

_model = None
_tokenizer = None


def get_model():
    """Load the model once and cache it."""
    global _model, _tokenizer
    if _model is None:
        print("  Loading entity extraction model...")
        _tokenizer = T5Tokenizer.from_pretrained(ADAPTER_PATH)
        base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        _model.eval()
        print("  Model loaded and ready.")
    return _model, _tokenizer


@mcp.tool()
def extract_patient_entities(patient_text: str) -> str:
    """
    Extract structured medical entities from a patient description.

    Takes free-text patient input and returns structured data including
    age, sex, medical conditions, current medications, and location.

    Much faster and cheaper than using an LLM for extraction.
    Average latency: ~100ms. Cost: ~$0.0001 per query.

    Args:
        patient_text: Free-text patient description.
            Example: "58 year old male with type 2 diabetes and CKD,
                     taking metformin and lisinopril, lives in Maryland"

    Returns:
        JSON string with extracted entities.
    """
    model, tokenizer = get_model()

    input_text = f"Extract medical entities from this patient description: {patient_text}"
    inputs = tokenizer(input_text, max_length=256, padding=True, truncation=True, return_tensors="pt")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256, num_beams=2, early_stopping=True,
        )
    extraction_time = (time.time() - start_time) * 1000

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        entities = json.loads(decoded)
    except json.JSONDecodeError:
        # Model often outputs JSON content without outer braces
        # Try wrapping in braces before giving up
        try:
            entities = json.loads("{" + decoded + "}")
        except json.JSONDecodeError:
            entities = {
                "age": None, "sex": None, "conditions": [],
                "medications": [], "location": None,
                "parse_error": "Model output was not valid JSON",
                "raw_output": decoded[:200],
            }

    entities["extraction_time_ms"] = round(extraction_time, 1)
    entities["model"] = "flan-t5-small-lora"

    return json.dumps(entities, indent=2)


if __name__ == "__main__":
    print("Testing entity extraction MCP tool...\n")
    test_cases = [
        "58 year old male with type 2 diabetes and CKD stage 3. Taking metformin and lisinopril. Lives in Maryland.",
        "I am a 34 year old woman with breast cancer stage 2. Currently on tamoxifen. Based in San Francisco.",
        "72 yo M with COPD, hypertension, and atrial fibrillation. Meds: albuterol, losartan, apixaban.",
        "I have depression and anxiety. 25 female. On sertraline.",
    ]
    for text in test_cases:
        print(f"Input: {text}")
        result = extract_patient_entities(text)
        print(f"Output: {result}\n")
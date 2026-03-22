"""Tests for entity extraction model and MCP server."""

import json
import pytest


class TestEntityExtraction:

    def test_model_loads(self):
        from src.mcp_servers.entity_extraction_server import get_model
        model, tokenizer = get_model()
        assert model is not None
        assert tokenizer is not None

    def test_basic_extraction(self):
        from src.mcp_servers.entity_extraction_server import extract_patient_entities
        result = json.loads(extract_patient_entities(
            "45 year old female with type 2 diabetes. Taking metformin."
        ))
        assert "age" in result
        assert "conditions" in result
        assert "extraction_time_ms" in result

    def test_extraction_speed(self):
        from src.mcp_servers.entity_extraction_server import extract_patient_entities
        result = json.loads(extract_patient_entities("30 year old male with asthma."))
        assert result["extraction_time_ms"] < 3000

    def test_handles_empty_input(self):
        from src.mcp_servers.entity_extraction_server import extract_patient_entities
        result = json.loads(extract_patient_entities(""))
        assert result is not None

    def test_handles_complex_input(self):
        from src.mcp_servers.entity_extraction_server import extract_patient_entities
        result = json.loads(extract_patient_entities(
            "72 year old male with COPD, hypertension, and diabetes. "
            "On albuterol, lisinopril, and metformin. Lives in Texas."
        ))
        assert "conditions" in result


class TestExtractEndpoint:

    def test_extract_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/extract", json={
            "text": "50 year old female with breast cancer."
        })
        assert response.status_code == 200
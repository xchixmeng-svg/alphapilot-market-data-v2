"""
test_run_fresh_canary.py

Tests the fail-closed guards and the Ollama response-parsing logic in
run_fresh_canary.py WITHOUT any real network call -- urllib.request.urlopen
is monkeypatched to return a canned response, per the honest disclosure in
run_fresh_canary.py's own docstring that the real network path has never
been exercised against a live Ollama server in this environment.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_fresh_canary as rfc


def test_missing_prep_summary_raises_guard_failure():
    with tempfile.TemporaryDirectory() as d:
        with_error = False
        try:
            rfc.load_and_guard_prep_summary(Path(d))
        except rfc.GuardFailure:
            with_error = True
        assert with_error, "missing PREP_SUMMARY.json must raise GuardFailure, not warn-and-continue"


def test_malformed_prep_summary_json_raises_guard_failure():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "PREP_SUMMARY.json").write_text("{not valid json", encoding="utf-8")
        raised = False
        try:
            rfc.load_and_guard_prep_summary(Path(d))
        except rfc.GuardFailure:
            raised = True
        assert raised


def test_prep_summary_2025_opened_true_raises_guard_failure():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "PREP_SUMMARY.json").write_text(json.dumps({"2025_opened": True}), encoding="utf-8")
        raised = False
        try:
            rfc.load_and_guard_prep_summary(Path(d))
        except rfc.GuardFailure:
            raised = True
        assert raised


def test_prep_summary_2025_opened_missing_raises_guard_failure():
    """2025_opened absent entirely (not even explicitly False) must also
    be treated as fail-closed, not assumed safe."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "PREP_SUMMARY.json").write_text(json.dumps({"n_cases": 64}), encoding="utf-8")
        raised = False
        try:
            rfc.load_and_guard_prep_summary(Path(d))
        except rfc.GuardFailure:
            raised = True
        assert raised


def test_prep_summary_2025_opened_false_passes():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "PREP_SUMMARY.json").write_text(json.dumps({"2025_opened": False}), encoding="utf-8")
        summary = rfc.load_and_guard_prep_summary(Path(d))
        assert summary["2025_opened"] is False


def test_load_packets_rejects_duplicate_case_id():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cases_shard_0.jsonl"
        rec = {"case_id": 1, "decision_date": 20240101, "code": "0050", "evidence": {}}
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
            f.write(json.dumps(rec) + "\n")  # duplicate
        raised = False
        try:
            rfc.load_packets(Path(d))
        except rfc.GuardFailure:
            raised = True
        assert raised


def test_load_packets_rejects_2025_decision_date():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cases_shard_0.jsonl"
        rec = {"case_id": 1, "decision_date": 20250115, "code": "0050", "evidence": {}}
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        raised = False
        try:
            rfc.load_packets(Path(d))
        except rfc.GuardFailure:
            raised = True
        assert raised


def test_load_packets_no_shard_files_raises_guard_failure():
    with tempfile.TemporaryDirectory() as d:
        raised = False
        try:
            rfc.load_packets(Path(d))
        except rfc.GuardFailure:
            raised = True
        assert raised


def test_load_packets_accepts_valid_data():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cases_shard_0.jsonl"
        rec = {"case_id": 1, "decision_date": 20240101, "code": "0050", "evidence": {}}
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        packets = rfc.load_packets(Path(d))
        assert len(packets) == 1
        assert packets[0]["case_id"] == 1


def test_immutable_packet_set_requires_summary_and_loaded_count_of_64():
    packets = [{"case_id": i} for i in range(64)]
    rfc.validate_immutable_packet_set({"cases": 64}, packets)

    for summary, rows in (({"cases": 63}, packets), ({"cases": 64}, packets[:-1])):
        raised = False
        try:
            rfc.validate_immutable_packet_set(summary, rows)
        except rfc.GuardFailure:
            raised = True
        assert raised


def test_call_ollama_parses_valid_response():
    """Mocks urlopen to return Ollama's documented /api/chat response
    shape and verifies call_ollama correctly extracts and parses the
    model's JSON content."""
    fake_model_json = {"decision": "REJECT", "evidence_quality": "WEAK"}
    ollama_envelope = {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": json.dumps(fake_model_json)},
        "done": True,
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(ollama_envelope).encode("utf-8")

    with mock.patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
        result = rfc.call_ollama(
            "system prompt", {"case_id": 1},
            model="qwen2.5:7b", url="http://127.0.0.1:11434/api/chat", timeout_s=10.0,
        )
        assert result == fake_model_json
        assert mocked.called
        called_request = mocked.call_args[0][0]
        sent_body = json.loads(called_request.data.decode("utf-8"))
        assert sent_body["model"] == "qwen2.5:7b"
        assert sent_body["format"] == "json"
        assert sent_body["stream"] is False
        assert sent_body["messages"][0]["role"] == "system"
        assert sent_body["messages"][0]["content"] == "system prompt"


def test_call_ollama_sends_json_schema_for_structured_output():
    schema = {
        "type": "object",
        "required": ["case_id", "evidence_observations"],
        "properties": {
            "case_id": {"type": "integer"},
            "evidence_observations": {"type": "array"},
        },
    }
    envelope = {"message": {"content": json.dumps({"case_id": 1, "evidence_observations": []})}}

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(envelope).encode("utf-8")

    with mock.patch("urllib.request.urlopen", return_value=FakeResponse()) as mocked:
        rfc.call_ollama(
            "sys", {"case_id": 1}, model="qwen2.5:7b", url="http://x",
            timeout_s=10.0, response_schema=schema,
        )
        sent_body = json.loads(mocked.call_args[0][0].data.decode("utf-8"))
        assert sent_body["format"] == schema
        assert sent_body["options"]["temperature"] == 0


def test_call_ollama_raises_on_missing_message_content():
    ollama_envelope = {"model": "qwen2.5:7b", "message": {"role": "assistant"}, "done": True}  # no content

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(ollama_envelope).encode("utf-8")

    with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
        raised = False
        try:
            rfc.call_ollama("sys", {}, model="qwen2.5:7b", url="http://x", timeout_s=10.0)
        except RuntimeError:
            raised = True
        assert raised


def test_call_ollama_raises_on_non_json_model_content():
    """If the model's content is not valid JSON (e.g. it added prose
    despite format='json'), call_ollama must raise, not silently return
    garbage -- the orchestrator treats this as a normal failed attempt."""
    ollama_envelope = {"message": {"content": "Sorry, I cannot comply with format=json here."}}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(ollama_envelope).encode("utf-8")

    with mock.patch("urllib.request.urlopen", return_value=FakeResponse()):
        raised = False
        try:
            rfc.call_ollama("sys", {}, model="qwen2.5:7b", url="http://x", timeout_s=10.0)
        except json.JSONDecodeError:
            raised = True
        assert raised


def test_max_cases_is_hard_capped_regardless_of_argument():
    assert rfc.HARD_MAX_CASES == 20

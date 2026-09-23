import json

import pytest

from layaspeech.corpus import load_manifest


def test_load_manifest(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps({"id": "u1", "text": "こんにちは"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    items = load_manifest(path)
    assert len(items) == 1
    assert items[0].utterance_id == "u1"


def test_load_manifest_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    row = json.dumps({"id": "u1", "text": "こんにちは"}, ensure_ascii=False)
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)

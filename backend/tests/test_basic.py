
from app.datasets import inventory, inspect_amrita


def test_missing_path(tmp_path):
    assert inventory(tmp_path / "missing")["exists"] is False


def test_amrita_inventory(tmp_path):
    assert inspect_amrita(tmp_path)["video_count"] == 0

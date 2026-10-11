"""CLI smoke tests.

Author: 晨星
"""

from __future__ import annotations

import json

from flowforge.cli import main


def test_selftest_passes():
    assert main(["selftest"]) == 0


def test_benchmark_writes_json(tmp_path):
    out = tmp_path / "bench.json"
    rc = main(
        [
            "benchmark",
            "--datasets",
            "torus3",
            "--seeds",
            "1",
            "--epochs",
            "8",
            "--n-train",
            "200",
            "--n-val",
            "80",
            "--n-test",
            "150",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "rows" in data and len(data["rows"]) >= 5
    assert any(r["model"] == "FlowFuse" for r in data["rows"])


def test_fit_command_runs():
    rc = main(["fit", "--dataset", "torus3", "--model", "realnvp", "--epochs", "8"])
    assert rc == 0

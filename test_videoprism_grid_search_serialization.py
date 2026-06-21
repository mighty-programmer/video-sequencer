import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from videoprism_grid_search import _json_safe


def test_json_safe_converts_numpy_values():
    payload = {
        "score": np.float32(0.25),
        "values": np.array([np.float32(1.0), np.float32(2.0)]),
        "nested": [{"flag": np.bool_(True), "count": np.int64(3)}],
    }

    converted = _json_safe(payload)

    assert converted == {
        "score": 0.25,
        "values": [1.0, 2.0],
        "nested": [{"flag": True, "count": 3}],
    }
    json.dumps(converted)

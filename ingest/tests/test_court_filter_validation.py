from __future__ import annotations

import pytest

from ingest.search import build_filter


def test_judge_filter_refuses_an_input_that_normalizes_to_empty() -> None:
    with pytest.raises(ValueError, match="Georgian surname"):
        build_filter(judges="---")

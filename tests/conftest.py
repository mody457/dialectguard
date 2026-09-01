"""Shared test fixtures.

Loading MARBERTv2 takes several seconds, so the TestClient is built once per
session. Entering it as a context manager runs the app's lifespan, which is
what actually loads the model.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """A TestClient with the real model loaded through the app lifespan."""
    with TestClient(app) as test_client:
        yield test_client

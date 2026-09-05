"""Shared test fixtures.

Loading MARBERTv2 takes several seconds, so the TestClient is built once per
session. Entering it as a context manager runs the app's lifespan, which is
what actually loads the model.

The rate limit is raised out of the way for the suite as a whole. Every test
shares one TestClient, so it shares one bucket, and at the production ceiling
of 30 a minute the suite would start failing once it grew past thirty
requests. That failure would look like a broken endpoint rather than a tripped
limit, so the limit is lifted here and exercised deliberately in
test_rate_limit.py instead.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import rate_limit
from app.main import app

# High enough that no ordinary test can reach it, and still a real limit rather
# than a disabled limiter, so the middleware stays in the path under test.
TEST_RATE_LIMIT = "100000/minute"


@pytest.fixture(scope="session", autouse=True)
def relax_rate_limit() -> Iterator[None]:
    """Lift the per-client ceiling for every test that does not set its own."""
    original = rate_limit.RATE_LIMIT
    rate_limit.RATE_LIMIT = TEST_RATE_LIMIT
    yield
    rate_limit.RATE_LIMIT = original


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """A TestClient with the real model loaded through the app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def rate_limited(request: pytest.FixtureRequest) -> Iterator[None]:
    """Apply a specific limit to one test, then clear the counters.

    Parametrized with the limit string, for example:

        @pytest.mark.parametrize("rate_limited", ["2/minute"], indirect=True)

    The storage is reset on both sides. Buckets are keyed by client address and
    every test shares one, so counts left behind would leak into whatever ran
    next and fail it for reasons that have nothing to do with it.
    """
    original = rate_limit.RATE_LIMIT
    rate_limit.limiter.reset()
    rate_limit.RATE_LIMIT = request.param
    try:
        yield
    finally:
        rate_limit.RATE_LIMIT = original
        rate_limit.limiter.reset()

"""Per-client rate limiting for the classification routes.

Inference is the expensive thing this service does. A forward pass costs about
120ms on CPU and the container runs a single uvicorn worker on purpose, so a
few unthrottled callers can occupy the whole process and every other client
waits behind them. The limit exists to contain that, not to meter usage.

Three decisions here are deliberate.

Identity is the socket peer, not X-Forwarded-For. That header is caller
supplied, so honouring it unconditionally would let anyone reset their own
bucket by inventing an address, which is worse than no limit because it still
looks like one. TRUST_PROXY_HEADER opts into it for deployments that really do
sit behind a proxy that overwrites the header.

Rejections are rendered through app.errors.error_response rather than
slowapi's own handler, which emits a different body shape. A client that has
learned one error envelope should not meet a second one at the moment it is
being throttled.

Only Retry-After is set, not the X-RateLimit family. slowapi's header
injection also appends Retry-After to successful responses, where it means
nothing, and reports the reset as a float epoch. One header the service
controls beats three it does not.
"""

from __future__ import annotations

import time

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request
from starlette.responses import Response

from app.config import RATE_LIMIT, TRUST_PROXY_HEADER
from app.errors import error_response

RATE_LIMIT_ERROR_CODE = "RATE_LIMIT_EXCEEDED"
RATE_LIMIT_STATUS_CODE = 429

FORWARDED_FOR_HEADER = "x-forwarded-for"

# Bucket for requests that arrive with no peer address, which happens on ASGI
# transports that do not set one. Sharing a single bucket throttles those
# callers together rather than exempting them, which is the safer failure.
UNKNOWN_CLIENT = "unknown"

# Floor for Retry-After. A window with less than a second left would otherwise
# advertise 0 and invite an immediate retry that is certain to fail.
MIN_RETRY_AFTER_SECONDS = 1


def client_identifier(request: Request) -> str:
    """Return the bucket key for a caller.

    The parameter has to be named `request`. slowapi calls the key function
    with a keyword argument, so renaming it raises a TypeError that surfaces
    only once a limit is actually evaluated.
    """
    if TRUST_PROXY_HEADER:
        forwarded = request.headers.get(FORWARDED_FOR_HEADER, "")
        # The left-most entry is the original client. Everything after it is
        # the proxy chain, which is not what should be throttled.
        original = forwarded.split(",")[0].strip()
        if original:
            return original

    return request.client.host if request.client else UNKNOWN_CLIENT


def current_rate_limit() -> str:
    """Return the configured limit, re-read on every evaluation.

    slowapi accepts a callable here, which keeps the limit out of import-time
    state. Tests rely on that: they raise the ceiling or lower it by setting
    the module attribute, without rebuilding the app and reloading the model.
    """
    return RATE_LIMIT


limiter = Limiter(
    key_func=client_identifier,
    default_limits=[current_rate_limit],
    # Off because slowapi appends Retry-After to successful responses too. See
    # the module docstring.
    headers_enabled=False,
)


def retry_after_seconds(request: Request) -> int:
    """Seconds until the caller's window resets.

    Reads the window slowapi recorded while rejecting the request. If that is
    missing the floor is returned, so a caller always gets a usable number
    rather than a header that is absent exactly when it is needed.
    """
    current = getattr(request.state, "view_rate_limit", None)
    if current is None:
        return MIN_RETRY_AFTER_SECONDS

    limit_item, identifiers = current
    reset_at, _remaining = limiter.limiter.get_window_stats(limit_item, *identifiers)
    return max(MIN_RETRY_AFTER_SECONDS, int(reset_at - time.time()))


def handle_rate_limit_exceeded(request: Request, exc: RateLimitExceeded) -> Response:
    """Render a throttled request in the same envelope as every other error."""
    retry_after = retry_after_seconds(request)
    response = error_response(
        RATE_LIMIT_STATUS_CODE,
        RATE_LIMIT_ERROR_CODE,
        f"Rate limit of {exc.detail} exceeded. "
        f"Retry after {retry_after} seconds.",
    )
    response.headers["Retry-After"] = str(retry_after)
    return response

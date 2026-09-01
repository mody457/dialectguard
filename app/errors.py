"""Uniform API error handling.

Every failure leaving the API has the same body shape, so clients can branch
on a stable machine-readable code instead of parsing prose or, worse, a stack
trace. Unhandled exceptions are caught and flattened here as well.
"""

from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class APIError(Exception):
    """An error that is safe to report back to the caller verbatim."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Build the canonical error body for a given status code."""
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, status_code=status_code)
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so no endpoint has to format errors by hand."""

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        # Rejections are logged without the offending text, so that a spike in
        # a single error code is visible without storing user content.
        logger.warning(
            "request_rejected",
            extra={
                "context": {
                    "event": "request_rejected",
                    "path": request.url.path,
                    "error_code": exc.code,
                    "status_code": exc.status_code,
                }
            },
        )
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Pydantic's own 422 body has a different shape from ours, so it is
        # reduced to the first problem and re-emitted in the canonical form.
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
        detail = first.get("msg", "Request body failed validation.")
        message = f"{location}: {detail}" if location else detail
        return error_response(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "INVALID_REQUEST_BODY", message
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        _: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        phrase = HTTPStatus(exc.status_code).phrase.upper().replace(" ", "_")
        return error_response(exc.status_code, phrase, str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        # The traceback goes to the logs, never to the client.
        logger.exception("unhandled_exception", extra={"context": {"error": str(exc)}})
        return error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            "An internal error occurred while handling the request.",
        )

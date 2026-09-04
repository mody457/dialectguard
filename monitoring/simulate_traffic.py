"""Send simulated traffic at a running DialectGuard instance and record it.

The point is not to measure accuracy. It is to capture what the service does
when the input is not what it was trained on, including the requests it
refuses, so that generate_report can compare a shifted category against the
in-distribution reference.

Requests go over HTTP against a running instance rather than calling the model
directly. The validation gate is part of what is being measured here, and it
exists only on the HTTP path: English never reaches the tokenizer, so the only
way to observe that is to watch the API reject it.

Raw text is kept in the output, which the service itself does not do by
default. The privacy rule that applies to production logs does not apply here,
because every row is public corpus text or a generated sentence, and Evidently
needs the column to run text descriptors over it.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import pandas as pd

from app.config import API_PREFIX, PROJECT_ROOT
from monitoring.samples import (
    ALL_CATEGORIES,
    RANDOM_SEED,
    SAMPLES_PER_CATEGORY,
    SampleSourceUnavailable,
    load_category,
    split_for,
)

DEFAULT_API_URL = "http://127.0.0.1:8000"
PREDICT_PATH = f"{API_PREFIX}/predict"
HEALTH_PATH = "/health"

REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_OUTPUT = PROJECT_ROOT / "monitoring" / "reports" / "traffic.parquet"

# Recorded in place of an HTTP status when the request never got one, so a
# connection dropped mid-run shows up as a row rather than ending the run.
TRANSPORT_ERROR_STATUS = 0
TRANSPORT_ERROR_CODE = "TRANSPORT_ERROR"


@dataclass(frozen=True)
class TrafficRecord:
    """One request and its outcome.

    dialect and confidence are None for anything the service refused, which is
    the expected shape for the whole English category rather than an anomaly.
    """

    category: str
    split: str
    text: str
    text_length: int
    status_code: int
    dialect: str | None
    confidence: float | None
    error_code: str | None
    latency_ms: float


def check_health(client: httpx.Client, api_url: str) -> None:
    """Fail before sending traffic if the service is not ready to serve.

    Checks model_loaded rather than just a reachable process, so a run cannot
    produce a file full of 503s that looks like drift.

    Raises:
        RuntimeError: if the service is unreachable or has no model loaded.
    """
    url = f"{api_url.rstrip('/')}{HEALTH_PATH}"
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"No service at {url}. Start it with "
            f"'uvicorn app.main:app' and try again. ({exc})"
        ) from exc

    if response.status_code != httpx.codes.OK:
        raise RuntimeError(f"{url} returned {response.status_code}, expected 200.")
    if not response.json().get("model_loaded"):
        raise RuntimeError(f"{url} reports the model is not loaded.")


def extract_error_code(response: httpx.Response) -> str:
    """Pull the machine-readable code out of the error envelope.

    Falls back to the status code if the body is not the documented envelope,
    so an unexpected failure is still recorded rather than raising here.
    """
    try:
        body = response.json()
    except ValueError:
        return f"HTTP_{response.status_code}"
    code = body.get("error", {}).get("code") if isinstance(body, dict) else None
    return code or f"HTTP_{response.status_code}"


def send_one(
    client: httpx.Client, api_url: str, category: str, text: str
) -> TrafficRecord:
    """Send one prediction request and record the outcome either way."""
    url = f"{api_url.rstrip('/')}{PREDICT_PATH}"
    started = time.perf_counter()
    try:
        response = client.post(url, json={"text": text})
    except httpx.HTTPError:
        return TrafficRecord(
            category=category,
            split=split_for(category),
            text=text,
            text_length=len(text),
            status_code=TRANSPORT_ERROR_STATUS,
            dialect=None,
            confidence=None,
            error_code=TRANSPORT_ERROR_CODE,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    latency_ms = (time.perf_counter() - started) * 1000
    if response.status_code == httpx.codes.OK:
        body = response.json()
        return TrafficRecord(
            category=category,
            split=split_for(category),
            text=text,
            text_length=len(text),
            status_code=response.status_code,
            dialect=body["dialect"],
            confidence=float(body["confidence"]),
            error_code=None,
            latency_ms=latency_ms,
        )

    return TrafficRecord(
        category=category,
        split=split_for(category),
        text=text,
        text_length=len(text),
        status_code=response.status_code,
        dialect=None,
        confidence=None,
        error_code=extract_error_code(response),
        latency_ms=latency_ms,
    )


def run_category(
    client: httpx.Client,
    api_url: str,
    category: str,
    count: int,
    seed: int,
) -> list[TrafficRecord]:
    """Load one category's samples and send every one of them.

    Requests go out one at a time. The service holds a single model in memory
    and inference is the slow part, so concurrency would queue at the same
    place while making the recorded latencies meaningless.
    """
    texts = load_category(category, count, seed)
    return [send_one(client, api_url, category, text) for text in texts]


def summarize(records: list[TrafficRecord]) -> str:
    """Render a per-category summary, without printing any of the text.

    Console encoding on Windows is not reliably UTF-8, and the Arabic
    categories would raise on the way out. Counts and codes are enough to see
    whether a run behaved.
    """
    frame = records_to_frame(records)
    lines: list[str] = []
    for category, group in frame.groupby("category", sort=False):
        served = group[group["status_code"] == httpx.codes.OK]
        lines.append(
            f"{category}: {len(group)} sent, {len(served)} served, "
            f"{len(group) - len(served)} refused"
        )
        if not served.empty:
            lines.append(f"  mean confidence {served['confidence'].mean():.3f}")
        refused = group[group["error_code"].notna()]
        for code, count in refused["error_code"].value_counts().items():
            lines.append(f"  {code}: {count}")
    return "\n".join(lines)


def records_to_frame(records: list[TrafficRecord]) -> pd.DataFrame:
    """Turn records into the frame generate_report expects."""
    return pd.DataFrame([asdict(record) for record in records])


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface for the simulator."""
    parser = argparse.ArgumentParser(
        description="Send simulated traffic at a running DialectGuard instance."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(ALL_CATEGORIES),
        choices=list(ALL_CATEGORIES),
        help="Categories to send. Defaults to all four.",
    )
    parser.add_argument("--count", type=int, default=SAMPLES_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--timeout", type=float, default=REQUEST_TIMEOUT_SECONDS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the simulation and return a process exit code."""
    args = build_parser().parse_args(argv)

    records: list[TrafficRecord] = []
    with httpx.Client(timeout=args.timeout) as client:
        try:
            check_health(client, args.api_url)
        except RuntimeError as exc:
            print(f"Health check failed: {exc}", file=sys.stderr)
            return 1

        for category in args.categories:
            try:
                records.extend(
                    run_category(client, args.api_url, category, args.count, args.seed)
                )
            except SampleSourceUnavailable as exc:
                print(f"Skipping {category}: {exc}", file=sys.stderr)

    if not records:
        print("No categories produced any traffic.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records_to_frame(records).to_parquet(args.output, index=False)

    print(summarize(records))
    print(f"\n{len(records)} rows written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Configuration for the DialectGuard service.

Two kinds of settings live here, and the split is deliberate.

Values that must stay identical between training and serving (the label
order, the maximum sequence length) are module-level constants. They are not
environment variables, because a mistyped variable in a deployment would
silently produce wrong labels or a train/serve skew rather than an obvious
crash.

Values that legitimately differ per environment (model location, log level,
build metadata) are read from the environment with conservative defaults.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Serving-critical constants. Do not move these to the environment. ---

# Index to ISO 3166-1 alpha-2 country code, in the exact order the model was
# trained with. Changing this breaks compatibility with the evaluated model.
# The shipped config.json only carries generic LABEL_0..LABEL_5 names, so the
# real mapping has to live here.
ID_TO_DIALECT: tuple[str, ...] = ("OM", "SA", "KW", "QA", "BH", "AE")

# Token budget used during fine-tuning, including [CLS] and [SEP]. Inputs
# longer than this are rejected rather than truncated, so that a caller never
# gets a confident prediction over a silently clipped sentence.
MAX_SEQUENCE_LENGTH = 128

# Hard character cap applied before the text reaches the tokenizer. This is a
# resource-exhaustion guard, not a modelling decision, so it is deliberately
# looser than MAX_SEQUENCE_LENGTH implies.
MAX_INPUT_CHARS = 1000

# Hard cap on the raw request body, checked against Content-Length before the
# body is read. MAX_INPUT_CHARS bounds the text field, but that check runs only
# after the whole body has been buffered into memory, so on its own it does not
# stop a caller from making the process hold an arbitrarily large payload. The
# allowance over MAX_INPUT_CHARS covers JSON framing and the worst-case
# encoding of Arabic, six bytes per character when a client escapes it.
MAX_REQUEST_BODY_BYTES = 8 * 1024

# Fraction of alphabetic characters that must be Arabic script for the input
# to be considered in-domain. The model has no notion of "not Arabic", so
# without this gate it would return a confident label for English text.
MIN_ARABIC_CHAR_RATIO = 0.5

# Documented macro F1 on the held-out test set. Reported by /version so
# callers can gate on model quality. See README for the caveats.
MODEL_MACRO_F1 = 0.618

API_PREFIX = "/api/v1"
API_VERSION = "v1"

# --- Environment-dependent settings. ---

MODEL_DIR = Path(
    os.getenv("DIALECTGUARD_MODEL_DIR", str(PROJECT_ROOT / "models" / "dialectguard_model"))
)

MODEL_VERSION = os.getenv("DIALECTGUARD_MODEL_VERSION", "dialectguard-marbertv2-v1")

LOG_LEVEL = os.getenv("DIALECTGUARD_LOG_LEVEL", "INFO").upper()

# When true, prediction logs include the raw input text. Off by default
# because request bodies are user content.
LOG_RAW_TEXT = os.getenv("DIALECTGUARD_LOG_RAW_TEXT", "false").lower() in {"1", "true", "yes"}


def resolve_git_commit() -> str | None:
    """Return the current short git commit, or None if it cannot be determined.

    Deployed containers usually have no .git directory, so the commit is
    normally injected as an environment variable at build time. The subprocess
    call is only a convenience for local development and must never raise.
    """
    injected = os.getenv("DIALECTGUARD_GIT_COMMIT")
    if injected:
        return injected
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


GIT_COMMIT = resolve_git_commit()

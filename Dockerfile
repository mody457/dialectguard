# syntax=docker/dockerfile:1

# Build stage. Dependencies resolve into a virtualenv that the runtime stage
# copies wholesale, so no part of the build toolchain reaches the final image.
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# Torch is installed first, from PyTorch's CPU index. The default PyPI wheel
# bundles CUDA and adds roughly 2GB to an image that will never see a GPU. The
# pin is read out of requirements.txt rather than repeated here so the two
# cannot drift apart. The second install then finds torch already satisfied and
# resolves only the rest.
RUN TORCH_PIN="$(grep -E '^torch==' requirements.txt)" \
 && pip install --index-url https://download.pytorch.org/whl/cpu "$TORCH_PIN" \
 && pip install -r requirements.txt


# Runtime stage.
FROM python:3.13-slim AS runtime

# Injected at build time because a deployed image has no .git directory for
# app.config.resolve_git_commit to read. Surfaced by /version, which is how a
# prediction gets correlated back to the release that produced it.
ARG GIT_COMMIT=unknown

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DIALECTGUARD_MODEL_DIR=/app/models/dialectguard_model \
    DIALECTGUARD_GIT_COMMIT=${GIT_COMMIT}

# Serve unprivileged. The process only reads its own code and the checkpoint,
# so there is nothing it needs that root would have to provide.
RUN useradd --create-home --uid 10001 dialectguard

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=dialectguard:dialectguard app/ ./app/

# The checkpoint is DVC tracked and gitignored, so a fresh clone has no weights
# and CI must run "dvc pull" before building. Baking it in rather than mounting
# it at runtime keeps a container self-contained: the image tag identifies the
# code and the weights together, so a rollback is one tag rather than two.
COPY --chown=dialectguard:dialectguard models/dialectguard_model/ ./models/dialectguard_model/

USER dialectguard

EXPOSE 8000

# Probes /health, which reports whether the weights are in memory rather than
# whether the process is alive. start-period covers the seconds MARBERTv2 takes
# to load, during which a failing check would be a false alarm rather than a
# fault.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker on purpose. Each would hold its own copy of the checkpoint, so
# scaling by workers costs 651MB of memory per unit. Scale with replicas
# instead, and let the orchestrator schedule them against real memory limits.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

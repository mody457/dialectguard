# DialectGuard

Gulf Arabic dialect classification API. Takes a piece of Arabic text and
returns one of six country codes: OM, SA, KW, QA, BH, AE.

Prediction, health and version endpoints run locally, rate limited per client.
The service builds as a Docker image, and CI runs lint, tests and the eval gate
before publishing one. Drift monitoring runs against a local instance.

## The problem it solves

Content routing and moderation on Gulf-region Arabic text. The concrete cases
are routing customer service tickets to a regional desk, filtering or grouping
comments by origin, and selecting a downstream model by dialect.

The six dialects overlap heavily, which is what makes this harder than telling
Arabic from English. A TF-IDF and logistic regression baseline on the same
split gets 44.6% macro F1. The fine-tuned model gets 61.8%. That gap is the
argument for the transformer, and 61.8% is also the reason to read the known
limitations at the bottom before wiring this into anything that matters.

## Architecture

FastAPI serves a fine-tuned MARBERTv2 checkpoint. Requests are validated and
cleaned before they reach the tokenizer, predictions are logged as JSON lines,
and the checkpoint and dataset splits are versioned with DVC rather than git.

### Model

Fine-tuned [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) with a
6-label sequence classification head, trained on the Gulf subset of QADI.

| | Macro F1 |
|---|---|
| MARBERTv2 fine-tuned | 61.8% |
| TF-IDF + logistic regression baseline | 44.6% |

Label order is fixed at `0: OM, 1: SA, 2: KW, 3: QA, 4: BH, 5: AE` and lives
in `app/config.py` as `ID_TO_DIALECT`. The shipped `config.json` only carries
placeholder `LABEL_0..LABEL_5` names, so that constant is the single source of
truth. Changing it breaks compatibility with the evaluated checkpoint.

`app/config.py` also holds `SOURCE_LABEL_NAMES`, the 18 dialect labels of the
source dataset. `ID_TO_DIALECT` is the Gulf six carved out of that space. The
eval harness and the monitoring sample sources both map through it, so it sits
next to the serving label order rather than inside either consumer.

### Preprocessing

Inference applies the same cleaning as training, in the same order: strip
@mentions, strip #hashtags, strip URLs, normalize whitespace. This is not
cosmetic. An earlier version of the model learned account identity from
@mentions instead of dialect, and skipping any step at serving time
reintroduces that train/serve skew.

The rules live in `app/preprocessing.py` and are covered by
`tests/test_preprocessing.py`.

### Layout

```
app/
  config.py          constants and environment settings
  preprocessing.py   training-time text cleaning
  validation.py      input gates applied before the tokenizer
  model.py           checkpoint loading and inference
  schemas.py         Pydantic request and response models
  errors.py          uniform error envelope and handlers
  logging_config.py  JSON line formatter
  main.py            app, routes, lifespan
eval_harness/
  evaluate.py        offline scoring and the CI F1 gate
monitoring/
  samples.py           the four traffic categories and their sources
  simulate_traffic.py  sends them at a running instance
  generate_report.py   builds the Evidently reports
  reports/             per-run output, gitignored
tests/
data/processed/      train, validation and test splits (DVC tracked)
docs/                eval report and confusion matrix, generated
models/dialectguard_model/
```

## Data and model versioning (DVC)

Model artifacts and the dataset splits are versioned with DVC rather than
committed to git. `models/dialectguard_model/` and `data/processed/` are
gitignored; `models/dialectguard_model.dvc` and `data/processed.dvc` hold the
content hashes and are what git tracks.

The splits are the ones the checkpoint was trained and evaluated on, recovered
from Drive. The eval harness needs `data/processed/gulf_test.parquet`, so CI
has to check the artifacts out before it can run the gate.

```bash
dvc status                          # workspace against the tracked version
dvc add models/dialectguard_model   # after replacing the weights
dvc add data/processed              # after changing the splits
dvc checkout                        # restore the version this commit points at
```

The directory is tracked as a single artifact, so weights, config and tokenizer
are versioned together and cannot drift apart.

### Cache location

The cache lives at `~/.dvc-cache/DialectGuard`, deliberately outside the
project. This repo sits inside a OneDrive folder, and DVC's default cache
(`.dvc/cache`) would hand OneDrive a second copy of the weights to sync on
every model version.

That path is set in `.dvc/config.local`, which is gitignored, so no absolute
local path leaks into the repo and nothing breaks when the project is built in
a container. On a fresh clone it has to be set again:

```bash
dvc cache dir --local /path/outside/any/sync/folder
dvc pull
```

### Remote

Backblaze B2, bucket `dialectguard-dvc-mody457`, endpoint
`s3.us-east-005.backblazeb2.com`.

```bash
dvc push    # upload tracked artifacts
dvc pull    # restore them
```

`region` is set explicitly to match the endpoint. B2 signs with SigV4, which
puts the region in the credential scope, so falling back to boto3's `us-east-1`
default produces signature failures.

Restore is verified. The cache was emptied and `dvc pull` rebuilt
`models/dialectguard_model/` byte for byte from the remote, so this is a real
backup rather than an assumed one. That matters more than usual here: the
current weights cannot be regenerated, because they came from a Colab session
that was never saved as a notebook.

Credentials live in `.dvc/config.local`, which is gitignored. Local development
uses a bucket-scoped application key with read and write access. CI gets a
separate read-only key, since the pipeline only ever pulls. The key in use was
rotated on 2026-09-02. The account master key does not work for either: B2's
S3-compatible API rejects it, and the error it returns does not say why.

Google Drive was evaluated first and ruled out on capacity. 15GB shared across
Gmail, Drive and Photos left under 700MB free, against 654MB for one model
version. B2's free tier is 10GB, currently 6.5 percent used, so roughly 15
versions fit.

## Running locally

Requires Python 3.13.

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs.

`requirements.txt` holds runtime dependencies only. The dev file adds the test,
eval, monitoring and lint tooling on top.

The model directory `models/dialectguard_model/` must contain `config.json`,
`model.safetensors` and the tokenizer files. The tokenizer is the unmodified
MARBERTv2 one. If it is ever missing, restore it with:

```bash
python -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('UBC-NLP/MARBERTv2').save_pretrained('models/dialectguard_model')"
```

### Configuration

All environment variables are optional. See `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `DIALECTGUARD_MODEL_DIR` | `models/dialectguard_model` | Where to load the checkpoint from |
| `DIALECTGUARD_MODEL_VERSION` | `dialectguard-marbertv2-v1` | Reported by `/version` |
| `DIALECTGUARD_LOG_LEVEL` | `INFO` | Root log level |
| `DIALECTGUARD_LOG_RAW_TEXT` | `false` | Include request text in prediction logs |
| `DIALECTGUARD_GIT_COMMIT` | unset | Build commit, for environments with no `.git` |
| `DIALECTGUARD_RATE_LIMIT` | `30/minute` | Per-client request allowance |
| `DIALECTGUARD_TRUST_PROXY_HEADER` | `false` | Read the client address from `X-Forwarded-For` |

Sequence length, label order and the Arabic-ratio threshold are deliberately
not configurable. A typo in a deployment variable should not be able to cause
a train/serve skew.

### Logging

One JSON object per line on stdout. Prediction logs carry input length,
predicted label, confidence and latency, but not the request text, since that
is user content. Set `DIALECTGUARD_LOG_RAW_TEXT=true` to include it while
debugging.

```json
{"timestamp": "2026-09-01T18:40:18.905467+00:00", "level": "INFO", "logger": "app.main", "message": "prediction", "event": "prediction", "input_length": 27, "dialect": "BH", "confidence": 0.613, "latency_ms": 116.37}
```

## API

`/health` and `/version` sit outside `/api/v1/` on purpose. They describe
the deployment rather than the classification contract, so they stay put when
the API version moves.

### POST /api/v1/predict

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "شلونك اليوم شخبارك عساك طيب"}'
```

```json
{"dialect": "BH", "confidence": 0.613}
```

`confidence` is the softmax probability of the winning class. It is not
calibrated, but it is usable as a routing threshold.

### GET /health

Readiness probe. Returns 200 only when the model is loaded into memory, and
503 otherwise. Use this rather than a plain TCP check.

### GET /version

API version, model version, base model, git commit and the documented macro
F1. Useful for correlating a prediction with the release that produced it.

### Errors

Every failure returns the same envelope, with no stack traces:

```json
{
  "error": {
    "code": "NOT_ARABIC_DOMINANT",
    "message": "Field 'text' is 0% Arabic script after cleaning, the minimum is 50%. This model only classifies Arabic text.",
    "status_code": 422
  }
}
```

| Code | Meaning |
|---|---|
| `INVALID_REQUEST_BODY` | Body failed schema validation, for example `text` missing |
| `REQUEST_BODY_TOO_LARGE` | Request body exceeded 8192 bytes (413) |
| `INVALID_CONTENT_LENGTH` | `Content-Length` header was not an integer (400) |
| `EMPTY_TEXT` | `text` was empty or whitespace only |
| `TEXT_TOO_LONG` | `text` exceeded 1000 characters |
| `TEXT_EMPTY_AFTER_PREPROCESSING` | `text` was nothing but mentions, hashtags or URLs |
| `NOT_ARABIC_DOMINANT` | Fewer than 50% of the letters were Arabic script |
| `TEXT_TOO_MANY_TOKENS` | Over 128 tokens after cleaning |
| `RATE_LIMIT_EXCEEDED` | Client is over its request allowance (429) |
| `MODEL_NOT_LOADED` | Model is not in memory (503) |
| `INTERNAL_ERROR` | Unexpected server-side failure (500) |

### Oversized input is rejected, not truncated

Training tokenized with `truncation=True`, so anything past 128 tokens was
silently clipped. The API deliberately does not do this. Input over the token
budget returns `TEXT_TOO_MANY_TOKENS` instead.

This is an intentional deviation from training behaviour, not an oversight. A
truncated sequence still yields a confident-looking label, and the caller has
no way to tell that the tail of their text was dropped. For an API whose output
drives routing decisions, failing loudly beats answering quietly on partial
input. Worth revisiting if a real workload turns out to send long text often.

The 1000 character cap is a field-level check, which by definition runs after
the whole body is already in memory. So the body itself is capped separately at
8192 bytes from `Content-Length`, before it is read. The two limits are not
redundant: the first bounds what reaches the model, the second bounds what
reaches the process. Chunked requests carry no `Content-Length` and bypass the
second, so a reverse proxy in front of the service should set its own body
limit.

Padding also differs between training and serving, and that one is a non-issue.
Training padded every sequence to a full 128 tokens with
`padding='max_length'`; the API pads nothing, since it classifies one text at a
time. The attention mask makes the two paths equivalent. Measured worst-case
difference in output probability is 4.3e-07, which is float noise.

## Rate limiting

Thirty requests per minute per client, refused with 429 and
`RATE_LIMIT_EXCEEDED` once exceeded. Configurable through
`DIALECTGUARD_RATE_LIMIT`.

The number comes from what one process can serve. A forward pass costs about
120ms on CPU and the container runs a single uvicorn worker on purpose, so
sustained capacity is roughly 8 requests per second. Thirty a minute is one
request every two seconds, which leaves any single caller far short of
occupying the process while still being generous for ticket routing or comment
filtering. A scraping loop hits it in about two seconds.

Refusals carry `Retry-After` in seconds, so a client can back off without
parsing the message:

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit of 30 per 1 minute exceeded. Retry after 58 seconds.",
    "status_code": 429
  }
}
```

`/health` and `/version` are exempt. The container HEALTHCHECK polls `/health`
on an interval, so throttling it would let load make a healthy container look
unhealthy. Everything else is covered, including `/docs`.

### Identifying the caller

By default the bucket key is the socket peer address. `X-Forwarded-For` is
ignored, which matters more than it sounds: the header is caller supplied, so
honouring it on a directly exposed service lets anyone mint a fresh bucket per
request. That is worse than having no limit, because it still looks like one.

Set `DIALECTGUARD_TRUST_PROXY_HEADER=true` only where a reverse proxy you
control is guaranteed to overwrite the header. Without it, a service behind a
proxy sees every request coming from the proxy address and throttles all
clients against one shared bucket.

An API key would be the better identifier. This service has no authentication
at all, so that is a larger change than a limit, and it is not built.

## Development

```bash
ruff check .
pytest
```

Lint is configured in `pyproject.toml`. The `ARG` rules are deliberately off:
test code legitimately takes arguments it does not use, such as a fixture
requested only for ordering or a stub whose signature has to match what it
replaces. Turning them on would mean scattering `noqa` over correct code.

116 tests. `tests/test_api.py` exercises the full HTTP path with the real model
loaded. It asserts status codes and response shape, not which dialect comes
back, because at 61.8% macro F1 the label for any single example is not a
stable contract.

`tests/test_eval_harness.py` never loads the checkpoint. Scoring the real split
takes minutes, so the model call is stubbed and what is tested is the wiring
around it: the label mapping, the row accounting and the threshold gate. Those
are the parts that fail quietly.

`tests/test_monitoring_samples.py` stubs the Hub fetch the same way, so the
suite never reaches the network. What it covers is the selection logic: the
counts, the determinism, the even draw across dialects, and the guard that
keeps a dialect the model predicts out of the shifted set.

## Evaluation

```bash
python -m eval_harness.evaluate
```

Scores the checkpoint on the held-out test split and exits non-zero if macro F1
falls below the floor. This is the gate CI runs before any Docker build.

Current result on the 3366 row test split:

| | macro F1 | accuracy |
|---|---|---|
| MARBERTv2 fine-tuned | 0.618 | 0.628 |

Per-country F1 ranges from 0.561 for Bahrain to 0.697 for Kuwait. The full
breakdown and the confusion matrix are in `docs/`, regenerated on every run.

The gate floor is 0.60, set below the documented 0.618 so that library drift
cannot fail a build over noise. A real regression costs far more than two
points: a reordered label map or a dropped preprocessing step is catastrophic,
not marginal.

Useful flags: `--split validation`, `--limit N` for a quick smoke run,
`--no-plot` to skip the heatmap, `--min-macro-f1` to override the floor.

### Why the harness is also a preprocessing check

The cleaning code used by the clean retrain was never saved, so it cannot be
diffed against `app/preprocessing.py`. Running the raw test split through
`app/preprocessing.py` and landing at the documented 0.618 is the evidence that
the two agree. A large gap would mean a train/serve skew that no unit test
would catch.

One deliberate difference from the serving path: training tokenized with
`truncation=True`, so long examples were clipped. The API rejects them instead.
The harness follows training rather than serving, because it measures the
checkpoint against the number training produced. Rejecting would change the
denominator.

## Drift monitoring

Two steps. The first needs the service already running.

```bash
uvicorn app.main:app                       # in another shell
python -m monitoring.simulate_traffic
python -m monitoring.generate_report
```

`simulate_traffic` sends four categories of text at a running instance and
records what came back, refusals included. It goes over HTTP rather than
calling the model directly because the validation gate is part of what is being
measured: English never reaches the tokenizer, so the only way to observe that
is to watch the API refuse it. `generate_report` then builds one Evidently
report per shifted category, each against the same reference. A blended current
set would average an MSA shift together with an outright refusal and hide both.

| Category | Source | Question |
|---|---|---|
| `in_distribution` | held-out Gulf test split | the reference the rest are compared against |
| `msa` | BBC Arabic headlines | does Modern Standard Arabic pass as a Gulf dialect |
| `non_gulf_dialect` | Egyptian, Levantine and Maghrebi rows of the source dataset | does another dialect pass as a Gulf one |
| `english` | generated support-desk sentences | does the validation gate hold |

Only `in_distribution` is versioned. The other three exist to produce a report,
so they are fetched from the Hub or generated on demand, and are neither
committed nor DVC tracked. Output lands in `monitoring/reports/`, which is
gitignored: the HTML runs to about 3.8MB per report and is regenerated by the
two commands above.

### What the first run showed

300 requests per category at seed 42, against the shipped checkpoint.

| Category | Served | Mean confidence | Columns drifted |
|---|---|---|---|
| `in_distribution` | 299 / 300 | 0.780 | reference |
| `msa` | 300 / 300 | 0.708 | 3 of 5 |
| `non_gulf_dialect` | 299 / 300 | 0.644 | 4 of 5 |
| `english` | 0 / 300 | none served | 3 of 3 |

Five columns are compared: `dialect`, `confidence`, `status_code`, `error_code`
and `text_length`. Numerical columns get a K-S test, categorical ones a Z-test
at two categories and chi-square above that.

MSA is the finding worth acting on. Every request was served, and neither
`status_code` nor `error_code` drifts (p = 0.32 on both), so from the outside
the traffic looks ordinary. The predictions underneath moved hard: `dialect`
drifts at p = 0, `confidence` at p = 3e-18. The service returns a confident
Gulf country code for text that belongs to no Gulf country, and nothing in the
response says otherwise. A confidence threshold would not catch it either,
since mean confidence only falls from 0.780 to 0.708.

`non_gulf_dialect` reads as 4 of 5, but that fourth column is not a finding.
One row of 300 failed the Arabic ratio gate. That single `NOT_ARABIC_DOMINANT`
is enough to move `error_code` by itself, because the reference's own single
refusal carried a different code (`TEXT_EMPTY_AFTER_PREPROCESSING`). Both sides
served 299 of 300 and disagree on one row each. Read the count as 3 of 5 with a
rounding error attached.

The real signal is the same shape as MSA. The service served 299 of 300 and
`status_code` did not move at all (p = 1.0), so the traffic looks ordinary from
the outside, while `dialect` drifts at p = 1e-19 and `confidence` at p = 2e-13.
Mean confidence falls to 0.644, lower than MSA's 0.708 but still nowhere near
low enough for a threshold to separate it from ordinary traffic.

English is the opposite case and the reassuring one. Nothing reached the model,
all 300 came back 422 `NOT_ARABIC_DOMINANT`, and the report compares behavior
only. With no served rows there is no prediction distribution to compare, so
`confidence` and `dialect` are dropped rather than reported as drift in an
empty column.

### Caveats on the sources

`non_gulf_dialect` comes from the test split of the dataset the Gulf splits
under `data/processed` were carved out of, so the reference and the shifted set
differ only in dialect. Same corpus, same collection, same split.

`msa` cannot do that, because no MSA corpus shares a source with the training
data. QADI's 18 labels are all country dialects with no MSA class. So the MSA
comparison carries a register and subject difference along with the variety
one: it is edited news headlines against tweets, not MSA against dialect in the
abstract. The `text_length` drift in that report is partly this, headlines
running a median 55 characters against the reference's 71.

Neither category is stratified the same way. The reference keeps the class
imbalance the documented 61.8% was measured against. `non_gulf_dialect` is
drawn evenly across its nine dialects instead, because its per-dialect counts
only reflect how much each country tweets, and an unstratified draw would be
mostly Egyptian.

## Deployment

### Docker

```bash
dvc pull                                  # a fresh clone has no weights
docker build --build-arg GIT_COMMIT="$(git rev-parse --short HEAD)" -t dialectguard:local .
docker run -d -p 8000:8000 dialectguard:local
```

Multi-stage. Dependencies resolve into a virtualenv in the build stage which the
runtime stage copies wholesale, so no build toolchain ships in the image.

Torch is installed from PyTorch's CPU index, not PyPI. The default wheel bundles
CUDA for a service that will never see a GPU. The pin is read out of
`requirements.txt` at build time rather than repeated in the Dockerfile, so the
two cannot drift apart. A built image reports `torch 2.11.0+cpu` and ships no
nvidia packages.

Size depends on which number you mean: about 2.0GB unpacked on a node, 0.89GB
compressed to pull. The virtualenv is 1.18GB of that and the checkpoint 654MB.

The checkpoint is baked in rather than mounted at runtime, so an image tag
identifies code and weights together and a rollback is one tag instead of two
things kept in sync. The cost is that CI has to `dvc pull` before `docker
build`, since `models/` is gitignored.

`.dockerignore` denies everything and re-includes only `app`, `models` and
`requirements.txt`. An allowlist keeps the next dataset or notebook out of the
build context by default instead of by someone remembering to add it.

The container runs as uid 10001. `HEALTHCHECK` probes `/health`, which reports
whether the weights are in memory rather than whether the process is alive, with
a start period covering model load.

One uvicorn worker on purpose. Each worker holds its own copy of the checkpoint,
so scale with replicas rather than workers.

Measured on a first run: healthy after model load, 366MB resident, first
prediction 393ms.

### CI

`.github/workflows/ci.yml` runs lint, tests, the eval gate, then the Docker build
and push, as four sequential jobs. They are separate jobs rather than steps so
the stage order is visible in the Actions UI and a lint failure costs seconds
instead of waiting on a 654MB model download.

Images publish to `ghcr.io/mody457/dialectguard`, tagged with the commit SHA and
`latest`, only on a push to main. Pull requests run everything through the eval
gate but publish nothing.

#### Required secrets

| Secret | Used for |
|---|---|
| `B2_ACCESS_KEY_ID` | pulling the checkpoint and splits from DVC |
| `B2_SECRET_ACCESS_KEY` | the same |

Use a read-only, bucket-scoped B2 application key. CI only ever reads from the
remote, so a read-write key would hand the pipeline more authority than it needs.
`GITHUB_TOKEN` is supplied by Actions and needs no setup; the docker job requests
`packages: write` against it.

#### Why the DVC cache matters

The checkpoint is restored from the Actions cache, keyed on the DVC pointer
files, so only the first run after the model or data changes pays B2 egress.
The bucket's free allowance is roughly 1GB per day and three uncached heavy jobs
would spend twice that in a single run.

## Known limitations

### Model and data

61.8% macro F1 means roughly one prediction in three is wrong. This is not
suitable for high-stakes routing without a confidence threshold or a human in
the loop.

Training data is Twitter/X only. Performance on WhatsApp messages, product
reviews and forum posts has not been measured.

The training set is imbalanced, with Kuwait overrepresented relative to Oman.
Per-country F1 varies accordingly, from 0.561 for Bahrain to 0.697 for Kuwait.

Residual user-level leakage in the train/test split cannot be ruled out. The
source dataset has no `user_id`, so only exact-duplicate text leakage could be
eliminated. The real figure may be below 61.8%.

Confidence does not identify out-of-domain input. The drift run above puts mean
confidence at 0.708 on Modern Standard Arabic and 0.644 on non-Gulf dialects,
against 0.780 on in-distribution traffic. Those are not separable by a
threshold. Arabic text that is not a Gulf dialect comes back with a confident
country code and nothing to flag it.

### Operational gaps

The rate limit counts per process. The container runs one worker and scales by
replicas, so N replicas allow N times the limit, and a client whose requests
land on different replicas gets a proportionally larger allowance. At three
replicas the effective ceiling is 90 a minute per client rather than 30. Fixing
it means a shared store: slowapi takes a `storage_uri`, so pointing it at Redis
is configuration rather than a rewrite, but nothing here runs Redis today.

The monitoring harness is not covered by CI. Two of the four sample sources
fetch from the HuggingFace Hub at run time, so running it in CI would put a
network dependency in the pipeline. The unit tests stub that fetch, which means
a Hub-side change to either dataset goes unnoticed until someone runs the
simulator by hand.

The reports are a one-off snapshot rather than a scheduled job. Nothing reruns
the simulator or alerts on a threshold, so reading a report is a manual step.

The checkpoint cannot be regenerated. It came from a Colab session that was
never saved as a notebook, so the DVC remote is the only copy. Restore from B2
is verified, which is what makes that acceptable.

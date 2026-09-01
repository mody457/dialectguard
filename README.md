# DialectGuard

Gulf Arabic dialect classification API. Takes a piece of Arabic text and
returns one of six country codes: OM, SA, KW, QA, BH, AE.

Intended for content routing and moderation on Gulf-region Arabic text, for
example routing customer service tickets or filtering comments by origin.

## Status

Minimum working version. Prediction, health and version endpoints run
locally. Docker, CI, drift monitoring and rate limiting are not built yet.

## Model

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

### Limitations

61.8% macro F1 means roughly one prediction in three is wrong. This is not
suitable for high-stakes routing without a confidence threshold or a human in
the loop.

Training data is Twitter/X only. Performance on WhatsApp messages, product
reviews and forum posts has not been measured. The training set is imbalanced,
with Kuwait overrepresented relative to Oman, so per-country performance
varies. Residual user-level leakage in the train/test split cannot be ruled
out, because the source dataset has no user_id and only exact-duplicate text
leakage was eliminated.

## Setup

Requires Python 3.13.

```bash
pip install -r requirements-dev.txt
```

`requirements.txt` holds runtime dependencies only. The dev file adds pytest
and httpx on top.

The model directory `models/dialectguard_model/` must contain `config.json`,
`model.safetensors` and the tokenizer files. The tokenizer is the unmodified
MARBERTv2 one. If it is ever missing, restore it with:

```bash
python -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('UBC-NLP/MARBERTv2').save_pretrained('models/dialectguard_model')"
```

## Running

```bash
uvicorn app.main:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs.

## Endpoints

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

## Errors

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
| `EMPTY_TEXT` | `text` was empty or whitespace only |
| `TEXT_TOO_LONG` | `text` exceeded 1000 characters |
| `TEXT_EMPTY_AFTER_PREPROCESSING` | `text` was nothing but mentions, hashtags or URLs |
| `NOT_ARABIC_DOMINANT` | Fewer than 50% of the letters were Arabic script |
| `TEXT_TOO_MANY_TOKENS` | Over 128 tokens after cleaning |
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

Padding also differs between training and serving, and that one is a non-issue.
Training padded every sequence to a full 128 tokens with
`padding='max_length'`; the API pads nothing, since it classifies one text at a
time. The attention mask makes the two paths equivalent. Measured worst-case
difference in output probability is 4.3e-07, which is float noise.

## Preprocessing

Inference applies the same cleaning as training, in the same order: strip
@mentions, strip #hashtags, strip URLs, normalize whitespace. This is not
cosmetic. An earlier version of the model learned account identity from
@mentions instead of dialect, and skipping any step at serving time
reintroduces that train/serve skew.

The rules live in `app/preprocessing.py` and are covered by
`tests/test_preprocessing.py`.

## Configuration

All environment variables are optional. See `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `DIALECTGUARD_MODEL_DIR` | `models/dialectguard_model` | Where to load the checkpoint from |
| `DIALECTGUARD_MODEL_VERSION` | `dialectguard-marbertv2-v1` | Reported by `/version` |
| `DIALECTGUARD_LOG_LEVEL` | `INFO` | Root log level |
| `DIALECTGUARD_LOG_RAW_TEXT` | `false` | Include request text in prediction logs |
| `DIALECTGUARD_GIT_COMMIT` | unset | Build commit, for environments with no `.git` |

Sequence length, label order and the Arabic-ratio threshold are deliberately
not configurable. A typo in a deployment variable should not be able to cause
a train/serve skew.

## Logging

One JSON object per line on stdout. Prediction logs carry input length,
predicted label, confidence and latency, but not the request text, since that
is user content. Set `DIALECTGUARD_LOG_RAW_TEXT=true` to include it while
debugging.

```json
{"timestamp": "2026-09-01T18:40:18.905467+00:00", "level": "INFO", "logger": "app.main", "message": "prediction", "event": "prediction", "input_length": 27, "dialect": "BH", "confidence": 0.613, "latency_ms": 116.37}
```

## Model versioning (DVC)

Model artifacts are versioned with DVC rather than committed to git.
`models/dialectguard_model/` is gitignored. `models/dialectguard_model.dvc`
holds the content hash and is what git tracks.

```bash
dvc status                          # workspace against the tracked version
dvc add models/dialectguard_model   # after replacing the weights
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

Credentials live in `.dvc/config.local`, which is gitignored. They are a
bucket-scoped application key with read and write access. The account master
key does not work here: B2's S3-compatible API rejects it, and the error it
returns does not say why.

Google Drive was evaluated first and ruled out on capacity. 15GB shared across
Gmail, Drive and Photos left under 700MB free, against 654MB for one model
version. B2's free tier is 10GB, currently 6.5 percent used, so roughly 15
versions fit.

A restore has not been verified yet. `dvc push` has run and the remote holds all
five objects, but nothing has been pulled back from it into an empty cache. Keep
an independent copy of `models/dialectguard_model/` until that test passes. The
current weights are unreproducible: they came from a Colab session that was
never saved as a notebook.

## Tests

```bash
pytest
```

34 tests. `tests/test_api.py` exercises the full HTTP path with the real model
loaded. It asserts status codes and response shape, not which dialect comes
back, because at 61.8% macro F1 the label for any single example is not a
stable contract.

## Layout

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
tests/
models/dialectguard_model/
```

## Not built yet

Rate limiting on `/api/v1/predict`, the eval harness with its F1 gate, Docker,
CI and drift monitoring.

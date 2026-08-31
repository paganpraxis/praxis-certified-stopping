# Praxis Certified Stopping

Finite-sample certification and protocol diagnosis for round-level early
stopping in rolling LLM self-revision.

This project is the corrected successor to the PX-057 discovery experiment. It
does not propose answer stability as a new stopping method. It asks whether a
frozen stopping policy can receive a rigorous certificate on the correctness
harm it causes, while paired inference separately determines whether continued
rolling revision is beneficial or whether the reasoning protocol itself is
inadmissible.

## Scientific design

The primary analyses are:

1. A one-sided exact upper confidence bound on
   `Pr(full correct, stopped wrong)`.
2. A paired lower confidence bound on
   `accuracy(stopped) - accuracy(full)`.
3. A dual certificate with a shared error budget.
4. A protocol-admissibility comparison between frozen round 2 and round 8.
5. A matched context-reset arm separating dependent rolling revision from
   independent fresh generations.
6. An extraction-validity audit across explicit, boxed, and fallback numeric
   answer parsers.

See [the preregistration draft](docs/PREREGISTRATION.md) for the hypotheses,
claim boundaries, gates, and confirmatory requirements.

## Current evidence

The deterministic harness smoke passes all software checks. It supports no
scientific claim.

Replaying the archived 200-item PX-057 discovery data produces:

- Stopping-harm point estimate: `0.005`.
- One-sided 95% exact harm upper bound: `0.02350`.
- Harm certificate at the `0.02` ceiling: **fail**.
- Paired net-effect point estimate: `+0.295`.
- One-sided paired lower bound: `+0.25899`.
- Paired net-effect certificate: **pass**.
- Dual certificate: **fail** because stopping harm is not certified.
- Round-2 versus round-8 lower bound: `+0.20951`.
- Protocol inadmissibility at the `0.05` margin: **yes**, pending causal
  confirmation with the context-reset arm.
- Cross-template extraction disagreements: `307`, requiring adjudication.

These are discovery results, not confirmatory evidence.

## Repository layout

```text
configs/   Smoke, discovery-replay, and confirmatory-template configurations
data/      Archived discovery generations and deterministic smoke fixture
docs/      Confirmatory preregistration draft
results/   Reproducible smoke and discovery summaries
scripts/   Collector, analyzer, and deterministic smoke runner
tests/     Statistical, parser, prompt, and end-to-end tests
```

## Reproduce the deterministic smoke

The smoke uses only the Python standard library:

```bash
python3 scripts/run_px057_certified_stopping_v2_smoke.py \
  --config configs/px057_certified_stopping_v2_smoke_20260831.json \
  --output-dir results/smoke-replay
```

## Reproduce the archived discovery analysis

```bash
python3 scripts/analyze_px057_certified_stopping_v2.py \
  --config configs/px057_certified_stopping_v2_discovery_replay_20260831.json \
  --raw-generations data/discovery/raw_generations.jsonl \
  --output results/discovery-replay/summary.json
```

## Run tests

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
```

## Prospective model smoke and confirmatory run

Install the model extras on a CUDA-capable host:

```bash
python3 -m pip install -e '.[model]'
```

Before generating confirmatory outcomes, copy the configuration template,
replace every placeholder, pin the model to an immutable revision, choose a
fresh non-overlapping sample, and timestamp the config and code commit. The
template itself is deliberately not frozen.

The prospective collector records both rolling and context-reset arms, along
with prompt, generated, and total tokens. Do not treat a successful smoke run
as a scientific gate.

## Claim boundary

Certificates are scoped to one declared exchangeable
model–dataset–prompt–decoding distribution. They are not universal safety or
deployment guarantees.
